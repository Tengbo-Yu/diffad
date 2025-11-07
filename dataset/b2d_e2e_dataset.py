import copy
from collections import defaultdict
import numpy as np
import logging
import traceback
from os import path as osp
import torch
import pickle
from pyquaternion import Quaternion
from mmdet.datasets.pipelines import to_tensor
from mmdet.parallel.data_container import DataContainer as DC
from mmdet.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet.datasets.custom_3d import Custom3DDataset
from nuscenes.eval.common.utils import Quaternion
import random
from dataset.util.rasterize_bev import Rasterize

NameMapping = {
    # =================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    # =========================================

    # =================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',

    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    # =========================================

    # ===================Construction===========
    "static.prop.warningconstruction": 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    # ===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    # =================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}


class BaseB2DE2EDataset(Custom3DDataset):
    def __init__(self,
                 queue_length=4,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 sample_interval=1,
                 name_mapping=NameMapping,
                 map_root=None,
                 map_file=None,
                 past_frames=4,
                 future_frames=4,
                 predict_frames=12,
                 planning_frames=6,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length if not self.test_mode else 1
        self.point_cloud_range = np.array(point_cloud_range)

        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.predict_frames = predict_frames
        self.planning_frames = planning_frames

        self.NameMapping = name_mapping
        self.map_root = map_root
        self.map_file = map_file
        self.map_element_class = {'Broken': 0, 'Solid': 1, 'SolidSolid': 2, 'Center': 3, 'TrafficLight': 4,
                                  'StopSign': 5}
        with open(self.map_file, 'rb') as f:
            self.map_infos = pickle.load(f)

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """
        data_queue = []
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_queue.insert(0, example)
        data_queue = self.union2one(data_queue)
        return data_queue

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length * self.sample_interval, index, self.sample_interval))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        map_gt_labels_list = [each['map_gt_labels'] for each in queue]
        map_gt_instance_list = [each['map_gt_instance'] for each in queue]
        gt_fut_traj_list = [each['gt_fut_traj'] for each in queue]
        gt_fut_traj_mask_list = [each['gt_fut_traj_mask'] for each in queue]
        sdc_planning_list = [to_tensor(each['sdc_planning']) for each in queue]
        sdc_planning_mask_list = [to_tensor(each['sdc_planning_mask']) for each in queue]
        command_list = [to_tensor(each['command']) for each in queue]
        ego2global_list = [to_tensor(each['ego2global']) for each in queue]
        ego_status_list = [to_tensor(each['ego_status']) for each in queue]

        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['folder'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['folder']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        queue['gt_labels_3d'] = gt_labels_3d_list
        queue['gt_bboxes_3d'] = gt_bboxes_3d_list
        queue['map_gt_labels'] = map_gt_labels_list
        queue['map_gt_instance'] = map_gt_instance_list
        queue['gt_fut_traj'] = gt_fut_traj_list
        queue['gt_fut_traj_mask'] = gt_fut_traj_mask_list
        queue['sdc_planning'] = torch.stack(sdc_planning_list)
        queue['sdc_planning_mask'] = torch.stack(sdc_planning_mask_list)
        queue['command'] = torch.stack(command_list)
        queue['ego2global'] = torch.stack(ego2global_list)
        queue['ego_status'] = torch.stack(ego_status_list)
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]

        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=np.nan_to_num(info['ego_yaw'], nan=np.pi / 2),
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel=info['ego_vel'],
            ego_accel=info['ego_accel'],
            ego_rotation_rate=info['ego_rotation_rate'],
            npc2world=info['npc2world'],
            timestamp=info['frame_idx'] / 10

        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root, cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)

            ego2world = np.eye(4)
            ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=input_dict['ego_yaw']).rotation_matrix
            ego2world[0:3, 3] = input_dict['ego_translation']
            lidar2global = ego2world @ lidar2ego
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    l2g_r_mat=lidar2global[0:3, 0:3],
                    l2g_t=lidar2global[0:3, 3]

                ))

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        if yaw < 0:
            yaw += 2 * np.pi
        yaw_in_degree = yaw / np.pi * 180

        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus

        # mapping
        input_dict['map_gt_instance'] = annos['map_gt_instance']
        input_dict['map_gt_labels'] = annos['map_gt_labels']

        # prediction
        input_dict['gt_fut_traj'] = annos['gt_fut_traj']
        input_dict['gt_sdc_fut_traj'] = annos['gt_sdc_fut_traj']
        input_dict['gt_fut_traj_mask'] = annos['gt_fut_traj_mask']
        input_dict['gt_sdc_fut_traj_mask'] = annos['gt_sdc_fut_traj_mask']

        # planning
        sdc_planning, sdc_planning_mask = self.get_ego_future_xy(index,
                                                                 self.sample_interval,
                                                                 self.planning_frames)

        # to avoid unexpected turn when ego is braking by eastimating a near-zero negtive y
        sdc_planning = sdc_planning.squeeze(0)[..., :2]  # T, 2
        sdc_planning_mask = sdc_planning_mask.squeeze(0)[..., :2]  # T, 2
        start_pos = np.array([[0., 0.]])
        input_dict['sdc_planning'] = np.concatenate([start_pos, sdc_planning], axis=0)
        input_dict['sdc_planning_mask'] = np.concatenate([np.ones_like(start_pos), sdc_planning_mask], axis=0)

        command = info['command_near']
        if command < 0:
            command = 4
        command -= 1

        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        input_dict['command'] = command_onehot

        # ego status
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = input_dict['ego_translation'][0:2]
        ego_lcf_feat[2:4] = input_dict['ego_accel'][2:4]
        ego_lcf_feat[4] = input_dict['ego_rotation_rate'][-1]
        ego_lcf_feat[5] = info['ego_size'][1]
        ego_lcf_feat[6] = info['ego_size'][0]
        ego_lcf_feat[7] = np.sqrt(input_dict['ego_translation'][0] ** 2 + input_dict['ego_translation'][1] ** 2)
        ego_lcf_feat[8] = info['steer']
        input_dict['ego_status'] = ego_lcf_feat

        # ego2lidar and ego2global
        lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
        world2ego = info['world2ego']
        input_dict['ego2lidar'] = np.linalg.inv(lidar2ego)
        input_dict['ego2global'] = np.linalg.inv(world2ego)
        return input_dict

    def get_map_info(self, index):
        gt_points = []
        gt_labels = []

        ann_info = self.data_infos[index]
        town_name = ann_info['town_name']
        map_info = self.map_infos[town_name]
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']
        lane_types = map_info['lane_types']
        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2, 3]

        # 1st search
        max_distance = 100
        chosed_idx = []
        for idx in range(len(lane_sample_points)):
            single_sample_points = lane_sample_points[idx]
            distance = np.linalg.norm((single_sample_points[:, 0:2] - ego_xy), axis=-1)
            if np.min(distance) < max_distance:
                chosed_idx.append(idx)

        for idx in chosed_idx:
            if not lane_types[idx] in self.map_element_class.keys():
                continue
            points = lane_points[idx]
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            # print(points_in_ego)
            mask = (points_in_ego[:, 0] > self.point_cloud_range[0]) & \
                   (points_in_ego[:, 0] < self.point_cloud_range[3]) & \
                   (points_in_ego[:, 1] > self.point_cloud_range[1]) & \
                   (points_in_ego[:, 1] < self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask, 0:2]

            if len(points_in_ego_range) > 1:
                gt_points.append(points_in_ego_range)
                gt_label = self.map_element_class[lane_types[idx]]
                gt_labels.append(gt_label)

        for idx in range(len(trigger_volumes_points)):
            if not trigger_volumes_types[idx] in self.map_element_class.keys():
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            mask = (points_in_ego[:, 0] > self.point_cloud_range[0]) & (
                    points_in_ego[:, 0] < self.point_cloud_range[3]) & (
                           points_in_ego[:, 1] > self.point_cloud_range[1]) & (
                           points_in_ego[:, 1] < self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask, 0:2]
            if mask.all():
                gt_label = self.map_element_class[trigger_volumes_types[idx]]
                gt_points.append(points_in_ego_range)
                gt_labels.append(gt_label)

        return gt_points, gt_labels

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]

        mask = (info['num_points'] >= -1)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_ids']
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # mappping
        map_gt_instance, map_gt_labels = self.get_map_info(index)

        # prediction
        ego_future_track, ego_future_mask = self.get_ego_future_xy(index, self.sample_interval, self.predict_frames)
        past_track, past_mask = self.get_past_or_future_xy(index, self.sample_interval, self.past_frames,
                                                           past_or_future='past', local_xy=True)
        predict_track, predict_mask = self.get_past_or_future_xy(index, self.sample_interval, self.predict_frames,
                                                                 past_or_future='future', local_xy=False)
        mask = (past_mask.sum((1, 2)) > 0).astype(np.int32)
        future_track = predict_track[:, 0:self.future_frames, :] * mask[:, None, None]
        future_mask = predict_mask[:, 0:self.future_frames, :] * mask[:, None, None]
        full_past_track = np.concatenate([past_track, future_track], axis=1)
        full_past_mask = np.concatenate([past_mask, future_mask], axis=1)

        # planning
        gt_sdc_bbox, gt_sdc_label = self.generate_sdc_info(index)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            map_gt_instance=map_gt_instance,
            map_gt_labels=map_gt_labels,
            gt_inds=gt_inds,
            gt_fut_traj=predict_track,
            gt_fut_traj_mask=predict_mask,
            gt_past_traj=full_past_track,
            gt_past_traj_mask=full_past_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=ego_future_track[..., :2],
            gt_sdc_fut_traj_mask=ego_future_mask,
        )
        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def generate_sdc_info(self, idx):
        info = self.data_infos[idx]
        ego_size = info['ego_size']
        ego_vel = info['ego_vel']
        psudo_sdc_bbox = np.array(
            [0.0, 0.0, 0.0, ego_size[0], ego_size[1], ego_size[2], -np.pi, ego_vel[1], ego_vel[0]])
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d

    def get_past_or_future_xy(self, idx, sample_rate, frames, past_or_future, local_xy=False):

        assert past_or_future in ['past', 'future']
        if past_or_future == 'past':
            adj_idx_list = range(idx - sample_rate, idx - (frames + 1) * sample_rate, -sample_rate)
        else:
            adj_idx_list = range(idx + sample_rate, idx + (frames + 1) * sample_rate, sample_rate)

        cur_frame = self.data_infos[idx]
        box_ids = cur_frame['gt_ids']
        adj_track = np.zeros((len(box_ids), frames, 2))
        adj_mask = np.zeros((len(box_ids), frames, 2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_ego_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2, 3]
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx < 0 or adj_idx >= len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame['folder']:
                    break
                if len(np.where(adj_frame['gt_ids'] == box_id)[0]) == 0:
                    continue
                assert len(np.where(adj_frame['gt_ids'] == box_id)[0]) == 1, np.where(adj_frame['gt_ids'] == box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids'] == box_id)[0][0]
                adj_box2lidar = world2lidar_ego_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2, 3]
                if local_xy:
                    adj_xy -= cur_xy
                adj_track[i, j, :] = adj_xy
                adj_mask[i, j, :] = 1
        return adj_track, adj_mask

    def get_ego_future_xy(self, idx, sample_rate, frames):

        adj_idx_list = range(idx + sample_rate, idx + (frames + 1) * sample_rate, sample_rate)
        cur_frame = self.data_infos[idx]
        adj_track = np.zeros((1, frames, 3))
        adj_mask = np.zeros((1, frames, 2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx < 0 or adj_idx >= len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_ego_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2, 3]
            yaw = np.arctan2(adj2cur_lidar[1, 0], adj2cur_lidar[0, 0])
            yaw = -yaw - np.pi
            while yaw > np.pi:
                yaw -= np.pi * 2
            while yaw < -np.pi:
                yaw += np.pi * 2
            adj_track[0, j, 0:2] = xy
            adj_track[0, j, 2] = yaw
            adj_mask[0, j, :] = 1

        return adj_track, adj_mask


class B2DE2EDataset(BaseB2DE2EDataset):
    """
        use_nonlinear_optimizer(bool): return scene-centric trajectory
    """

    def __init__(self,
                 sample_interval=1,
                 queue_length=4,
                 point_cloud_range=[-50., -50., -5., 50., 50., 3.],
                 bev_reso=(0.125, 0.125),
                 map_root=None,
                 map_file=None,
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 # For debug
                 is_debug=False,
                 len_debug=30,
                 *args, **kwargs):

        self.is_debug = is_debug
        self.len_debug = len_debug
        self.bev_reso = bev_reso
        patch_h = point_cloud_range[4] - point_cloud_range[1]
        patch_w = point_cloud_range[3] - point_cloud_range[0]
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (int(patch_h / bev_reso[1]), int(patch_w / bev_reso[0]))
        super().__init__(sample_interval=sample_interval,
                         queue_length=queue_length,
                         point_cloud_range=point_cloud_range,
                         map_root=map_root,
                         map_file=map_file,
                         predict_frames=predict_steps,
                         planning_frames=planning_steps,
                         past_frames=past_steps,
                         future_frames=fut_steps,
                         *args, **kwargs)

    def __len__(self):
        if not self.is_debug:
            return len(self.data_infos)
        else:
            return self.len_debug

    def __getitem__(self, idx):
        max_retry_count = 100
        retry_count = 0
        while retry_count < max_retry_count:
            try:
                data = super().__getitem__(idx)
                data = self._parse_data_container(data)
                return data
            except Exception as error:
                retry_count += 1
                logging.warning("resample warning: %s", error)
                traceback.print_exc()
                idx = random.randint(0, len(self) - 1)

        raise Exception('read data error')

    def _parse_data_container(self, data):
        cam_names = ['cam_front', 'cam_front_right',
                     'cam_front_left', 'cam_back',
                     'cam_back_left', 'cam_back_right']

        img_metas = data.pop('img_metas')

        if isinstance(img_metas, DC):
            img_metas = img_metas.data
        else:
            assert 'invalid img_meta datatype'

        img = data.pop('img')
        if isinstance(img, list):
            img = img[0].data
        else:
            img = img.data

        # metadata
        metadata = {'frame_metadata': [
            {
                'frame_id': img_metas[i]['frame_idx'],
                'timestamp': img_metas[i]['timestamp'],
                'clip_id': img_metas[i]['scene_token'].replace('v1/', ''),
                'filename': img_metas[i]['filename']
            }
            for i in range(self.queue_length)]}
        data['metadata'] = metadata

        # image and lidar2img
        data['image'] = {}
        lidar2img = defaultdict(dict)
        filename_list = img_metas[0]['filename']
        for i, filename in enumerate(filename_list):
            cam_name = filename.split('/')[-2]  # rgb_xxx
            assert 'rgb' in cam_name
            cam_name = cam_name.replace('rgb', 'cam')
            data['image'][cam_name] = img[:, i]  # T, n_cam, c, H, W
            lidar2img[cam_name] = np.array([img_metas[ts_idx]['lidar2img'][i] for ts_idx in range(self.queue_length)])

        data['lidar2img'] = lidar2img
        lidar2ego = np.linalg.inv(data['ego2lidar'])
        lidar2global = data['ego2global'] @ lidar2ego
        data['lidar2global'] = lidar2global

        # rasterized
        self._rasterized_bev(data)

        sample_dict = {
            'image': data['image'],
            'lidar2img': data['lidar2img'],
            'lidar2global': data['lidar2global'],
            'metadata': data['metadata'],
            'task_label': data['task_label'],
            'perc_map': data['perc_map'],
            'motion_map': data['motion_map'],
            'plan_map': data['plan_map'],
            'sdc_planning': data['sdc_planning'],
            'sdc_planning_mask': data['sdc_planning_mask'],
            "ego_status": data['ego_status'],
            "command": data['command']
        }
        return sample_dict

    def _rasterized_bev(self, data):
        perc_canvas_list = []
        motion_canvas_list = []
        plan_canvas_list = []
        task_label = np.ones((self.queue_length, 4), dtype=np.int64)

        gt_bboxes_3d = data['gt_bboxes_3d']  # list of LiDARInstance3DBoxes, each has a shape of (N,9) tensor
        gt_labels_3d = data['gt_labels_3d']  # list of tensor
        map_gt_instance = data['map_gt_instance']  # list of list tensor
        map_gt_labels = data['map_gt_labels']  # list of list
        gt_fut_traj = data['gt_fut_traj']  # list of tensor, each has a shape of (N,12,2)
        gt_fut_traj_mask = data['gt_fut_traj_mask']
        sdc_planning = data['sdc_planning']
        sdc_planning_mask = data['sdc_planning_mask']  # list of tensor, each has a shape of (T,2)

        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range  # --> x
        bev_canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
        for ts_idx in range(self.queue_length):
            perc_canvas = copy.deepcopy(bev_canvas)
            motion_canvas = copy.deepcopy(bev_canvas)
            plan_canvas = copy.deepcopy(bev_canvas)

            # mapping
            cur_map_gt_instance = map_gt_instance[ts_idx]
            cur_map_gt_labels = map_gt_labels[ts_idx]
            cur_map_gt_classes = [next(key for key, val in self.map_element_class.items() if val == value_to_find)
                                  for value_to_find in cur_map_gt_labels]
            perc_canvas = Rasterize.draw_bev_static_map(cur_map_gt_instance, cur_map_gt_classes,
                                                        x_max, y_max, self.bev_reso, perc_canvas)

            # detection
            cur_gt_bboxes_3d = gt_bboxes_3d[ts_idx]
            cur_gt_labels_3d = gt_labels_3d[ts_idx]
            cur_gt_classes_3d = [next(key for key, val in self.cat2id.items() if val == value_to_find)
                                 for value_to_find in cur_gt_labels_3d]
            perc_canvas = Rasterize.draw_bev_obstacle_map(cur_gt_bboxes_3d, cur_gt_classes_3d,
                                                          x_max, y_max, self.bev_reso, perc_canvas)

            if (ts_idx + 1) >= self.past_frames:
                # motion
                cur_gt_fut_traj = gt_fut_traj[ts_idx]
                cur_gt_fut_traj_mask = gt_fut_traj_mask[ts_idx]
                # add start point
                cur_gt_bbox_center = cur_gt_bboxes_3d.tensor[:, :2].numpy()[:, None]  # N,1,2
                cur_gt_fut_traj = np.concatenate([cur_gt_bbox_center, cur_gt_fut_traj], axis=1)
                cur_gt_fut_traj_mask = np.concatenate([np.ones_like(cur_gt_bbox_center), cur_gt_fut_traj_mask], axis=1)
                motion_canvas = Rasterize.draw_prediction_map(cur_gt_fut_traj, cur_gt_fut_traj_mask,
                                                              x_max, y_max, self.bev_reso, self.predict_frames,
                                                              motion_canvas)

                # planning
                cur_sdc_planning = sdc_planning[ts_idx]
                plan_canvas = Rasterize.draw_planning_map(cur_sdc_planning, x_max, y_max, self.bev_reso, plan_canvas)

            else:
                gt_fut_traj_mask[ts_idx][:] = 0
                sdc_planning_mask[ts_idx][:] = 0
                task_label[ts_idx, 2:] = 0

            perc_canvas = perc_canvas.transpose(2, 0, 1)  # c, h, w
            motion_canvas = motion_canvas.transpose(2, 0, 1)
            plan_canvas = plan_canvas.transpose(2, 0, 1)
            perc_canvas_list.append(perc_canvas)
            motion_canvas_list.append(motion_canvas)
            plan_canvas_list.append(plan_canvas)

        data['perc_map'] = np.stack(perc_canvas_list)
        data['motion_map'] = np.stack(motion_canvas_list)
        data['plan_map'] = np.stack(plan_canvas_list)
        data['task_label'] = task_label
