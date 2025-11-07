from collections import defaultdict
# ---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# ---------------------------------------------------------------------------------#
import logging
import traceback
import copy
import numpy as np
import torch
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.nuscenes_dataset import NuScenesDataset
from mmdet.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet.fileio.file_client import FileClient
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet.parallel.data_container import DataContainer as DC
import random
from nuscenes import NuScenes
from mmdet.datasets.data_utils.vector_map import VectorizedLocalMap, CLASS2LABEL
from mmdet.datasets.eval_utils.map_api import NuScenesMap
from mmdet.datasets.data_utils.trajectory_api import NuScenesTraj
from dataset.util.rasterize_bev import Rasterize


class BaseNuScenesE2EDataset(NuScenesDataset):
    r"""NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 queue_length=4,
                 point_cloud_range=(-50., -50., -5., 50., 50., 3.),
                 bev_reso=(0.125, 0.125),
                 overlap_test=False,
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 padding_value=-10000,
                 map_fixed_ptsnum_per_line=20,
                 use_nonlinear_optimizer=False,
                 file_client_args=dict(backend='disk'),
                 *args,
                 **kwargs):
        # init before super init since it is called in parent class
        self.file_client_args = file_client_args
        self.file_client = FileClient(**file_client_args)

        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.scene_token = None

        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        self.nusc = NuScenes(version=self.version,
                             dataroot=self.data_root, verbose=True)

        self.nusc_maps = {
            'boston-seaport': NuScenesMap(dataroot=self.data_root, map_name='boston-seaport'),
            'singapore-hollandvillage': NuScenesMap(dataroot=self.data_root, map_name='singapore-hollandvillage'),
            'singapore-onenorth': NuScenesMap(dataroot=self.data_root, map_name='singapore-onenorth'),
            'singapore-queenstown': NuScenesMap(dataroot=self.data_root, map_name='singapore-queenstown'),
        }

        self.bev_reso = bev_reso
        self.point_cloud_range = point_cloud_range
        patch_h = point_cloud_range[4] - point_cloud_range[1]
        patch_w = point_cloud_range[3] - point_cloud_range[0]
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (int(patch_h / bev_reso[0]), int(patch_w / bev_reso[0]))
        self.vector_map = VectorizedLocalMap(
            self.data_root,
            patch_size=self.patch_size)
        self.padding_value = padding_value
        self.fixed_num = map_fixed_ptsnum_per_line
        self.traj_api = NuScenesTraj(self.nusc,
                                     self.predict_steps,
                                     self.planning_steps,
                                     self.past_steps,
                                     self.fut_steps,
                                     self.with_velocity,
                                     self.CLASSES,
                                     self.box_mode_3d,
                                     self.use_nonlinear_optimizer)

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_globals_3d: gt_globals of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame (list)
        """
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index - self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)

        example = self.pipeline(input_dict)
        assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]

        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        data_queue.insert(0, example)

        # retrieve previous infos
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            assert example['gt_labels_3d'].data.shape[0] == example['gt_fut_traj'].shape[0]
            data_queue.insert(0, copy.deepcopy(example))
        data_queue = self.union2one(data_queue)
        return data_queue

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

    def union2one(self, queue):
        """
        convert sample dict into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        gt_labels_3d_list = [each['gt_labels_3d'].data for each in queue]
        gt_bboxes_3d_list = [each['gt_bboxes_3d'].data for each in queue]
        map_gt_labels_list = [each['map_gt_labels'] for each in queue]
        map_gt_instance_list = [each['map_gt_instance'] for each in queue]
        gt_fut_traj_list = [each['gt_fut_traj'] for each in queue]
        gt_fut_traj_mask_list = [each['gt_fut_traj_mask'] for each in queue]
        gt_sdc_fut_traj_list = [to_tensor(each['gt_sdc_fut_traj']) for each in queue]
        gt_sdc_fut_traj_mask_list = [to_tensor(each['gt_sdc_fut_traj_mask']) for each in queue]
        sdc_planning_list = [to_tensor(each['sdc_planning']) for each in queue]
        sdc_planning_mask_list = [to_tensor(each['sdc_planning_mask']) for each in queue]
        command_list = [to_tensor(each['command']) for each in queue]
        ego2global_list = [to_tensor(each['ego2global']) for each in queue]
        ego_status_list = [to_tensor(each['ego_status']) for each in queue]

        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        queue['gt_labels_3d'] = gt_labels_3d_list
        queue['gt_bboxes_3d'] = gt_bboxes_3d_list
        queue['map_gt_labels'] = map_gt_labels_list
        queue['map_gt_instance'] = map_gt_instance_list
        queue['gt_fut_traj'] = gt_fut_traj_list
        queue['gt_fut_traj_mask'] = gt_fut_traj_mask_list
        queue['gt_sdc_fut_traj'] = gt_sdc_fut_traj_list
        queue['gt_sdc_fut_traj_mask'] = gt_sdc_fut_traj_mask_list
        queue['sdc_planning'] = torch.stack(sdc_planning_list)
        queue['sdc_planning_mask'] = torch.stack(sdc_planning_mask_list)
        queue['command'] = torch.stack(command_list)
        queue['ego2global'] = torch.stack(ego2global_list)
        queue['ego_status'] = torch.stack(ego_status_list)
        return queue

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
                - gt_inds (np.ndarray): Instance ids of ground truths.
                - gt_fut_traj (np.ndarray): .
                - gt_fut_traj_mask (np.ndarray): .
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        sample = self.nusc.get('sample', info['token'])
        ann_tokens = np.array(sample['anns'])[mask]
        assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]

        gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = self.traj_api.get_traj_label(
            info['token'], ann_tokens)
        sdc_vel = self.traj_api.sdc_vel_info[info['token']]
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = self.traj_api.get_sdc_traj_label(
            info['token'])

        sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(
            info['token'])

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # 
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = info['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        # panoptic format
        location = self.nusc.get('log', self.nusc.get(
            'scene', info['scene_token'])['log_token'])['location']

        filtered_vectors = self.vector_map.gen_vectorized_samples(
            location, lidar2global_translation, lidar2global_rotation
        )
        map_gt_labels = []
        map_gt_instance = []
        for instance_dict in filtered_vectors:
            instance = instance_dict['pts']
            type = instance_dict['type']
            if type != -1:
                map_gt_instance.append(instance)
                map_gt_labels.append(type)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            map_gt_instance=map_gt_instance,
            map_gt_labels=map_gt_labels,
            gt_names=gt_names_3d,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=gt_sdc_fut_traj,
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
        )
        assert gt_fut_traj.shape[0] == gt_labels_3d.shape[0]
        assert gt_past_traj.shape[0] == gt_labels_3d.shape[0]
        return anns_results

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

        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        yaw = quaternion_yaw(rotation) / np.pi * 180
        if yaw < 0:
            yaw += 2 * np.pi
        yaw_in_degree = yaw / np.pi * 180

        can_bus[-2] = yaw_in_degree
        can_bus[-1] = yaw

        # mapping
        input_dict['map_gt_instance'] = annos['map_gt_instance']
        input_dict['map_gt_labels'] = annos['map_gt_labels']

        # prediction
        input_dict['gt_fut_traj'] = annos['gt_fut_traj']
        input_dict['gt_sdc_fut_traj'] = annos['gt_sdc_fut_traj']
        input_dict['gt_fut_traj_mask'] = annos['gt_fut_traj_mask']
        input_dict['gt_sdc_fut_traj_mask'] = annos['gt_sdc_fut_traj_mask']

        # planning
        sdc_planning = annos['sdc_planning'].squeeze(0)[..., :2]  # T, 2
        sdc_planning_mask = annos['sdc_planning_mask'].squeeze(0)[..., :2]  # T, 2

        start_pos = np.array([[0., 0]])
        input_dict['sdc_planning'] = np.concatenate([start_pos, sdc_planning], axis=0)
        input_dict['sdc_planning_mask'] = np.concatenate([np.ones_like(start_pos), sdc_planning_mask], axis=0)

        command_onehot = np.zeros(6)
        command_onehot[annos['command']] = 1
        input_dict['command'] = command_onehot

        # ego status
        ego_status = np.zeros(18)
        ego_status[:3] = translation
        ego_status[3:7] = rotation
        ego_status[7:10] = 0
        ego_status[10:13] = 0
        ego_status[13:16] = 0
        ego_status[16] = yaw
        ego_status[17] = yaw_in_degree

        # ego status
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = translation[0:2]
        ego_lcf_feat[7] = np.sqrt(translation[0] ** 2 + translation[1] ** 2)
        ego_lcf_feat[8] = 0
        input_dict['ego_status'] = ego_lcf_feat

        # ego2lidar and ego2global
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']

        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        ego2lidar = np.eye(4)
        ego2lidar[:3, :3] = l2e_r_mat
        ego2lidar[:3, 3] = l2e_t

        ego2global = np.eye(4)
        ego2global[:3, :3] = e2g_r_mat
        ego2global[:3, 3] = e2g_t

        input_dict['ego2lidar'] = ego2lidar
        input_dict['ego2global'] = ego2global
        return input_dict

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


class NuScenesE2EDataset(BaseNuScenesE2EDataset):
    """

        use_nonlinear_optimizer(bool): return scene-centric trajectory
    """

    def __init__(self,
                 queue_length=4,
                 point_cloud_range=[-50., -50., -5., 50., 50., 3.],
                 bev_reso=(0.125, 0.125),
                 overlap_test=False,
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 use_nonlinear_optimizer=True,
                 is_debug=False,
                 len_debug=30,
                 file_client_args=dict(backend='disk'),
                 *args, **kwargs):

        self.is_debug = is_debug
        self.len_debug = len_debug
        super().__init__(queue_length=queue_length,
                         bev_reso=bev_reso,
                         point_cloud_range=point_cloud_range,
                         overlap_test=overlap_test,
                         predict_steps=predict_steps,
                         planning_steps=planning_steps,
                         past_steps=past_steps,
                         fut_steps=fut_steps,
                         use_nonlinear_optimizer=use_nonlinear_optimizer,
                         file_client_args=file_client_args, *args, **kwargs)

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
                'clip_id': img_metas[i]['scene_token'],
                'filename': img_metas[i]['filename']
            }
            for i in range(self.queue_length)]}
        data['metadata'] = metadata

        # image and calibration
        data['image'] = {}
        lidar2img = defaultdict(dict)
        for i, cam_name in enumerate(cam_names):
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
        sdc_planning_mask = data['sdc_planning_mask']  # list of tensor, each has a shape of (1,T,2)

        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range  # --> x
        bev_canvas = np.zeros((*self.canvas_size, 3), dtype=np.uint8)
        for ts_idx in range(self.queue_length):
            perc_canvas = copy.deepcopy(bev_canvas)
            motion_canvas = copy.deepcopy(bev_canvas)
            plan_canvas = copy.deepcopy(bev_canvas)

            # mapping
            cur_map_gt_instance = map_gt_instance[ts_idx]
            cur_map_gt_labels = map_gt_labels[ts_idx]
            cur_map_gt_classes = [next(key for key, val in CLASS2LABEL.items() if val == value_to_find)
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
            if (ts_idx + 1) >= self.past_steps:
                # motion
                cur_gt_fut_traj = gt_fut_traj[ts_idx]
                cur_gt_fut_traj_mask = gt_fut_traj_mask[ts_idx]
                # add start point
                cur_gt_bbox_center = cur_gt_bboxes_3d.tensor[:, :2].numpy()[:, None]  # N,1,2
                cur_gt_fut_traj = np.concatenate([cur_gt_bbox_center, cur_gt_fut_traj], axis=1)
                cur_gt_fut_traj_mask = np.concatenate([np.ones_like(cur_gt_bbox_center), cur_gt_fut_traj_mask], axis=1)
                motion_canvas = Rasterize.draw_prediction_map(cur_gt_fut_traj, cur_gt_fut_traj_mask,
                                                              x_max, y_max, self.bev_reso, self.predict_steps,
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
