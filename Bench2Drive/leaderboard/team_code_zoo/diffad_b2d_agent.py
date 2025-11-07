import os
import sys
import json
import datetime
import pathlib
import time
from collections import defaultdict
import cv2
import carla
import math
from scipy.optimize import fsolve
import torch
import carla
import numpy as np
from PIL import Image
from diffusers.models import AutoencoderKL

from team_code_zoo.pid_controller import PIDController
from team_code_zoo.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent

from model.ddpm_diffad import DDPMDiffad
from model.fm_diffad import FMDiffad
from util.load_and_save_callback import LoadAndSaveCallback
from util.config_util import load_yaml, update_config
from util.visualize_util import draw_ego_car, draw_ego_planning, proj_traj_to_pv_wide_path, draw_faded_path
from mmdet.datasets.pipelines import Compose
from mmdet.core.bbox import get_box_type
from pyquaternion import Quaternion

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

def get_entry_point():
    return 'DiffadAgent'
class DiffadAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steer = 0
        self.pidcontroller = PIDController() 
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        # build diffad model
        config_path = path_to_conf_file.split('+')[0]
        ckpt_path = path_to_conf_file.split('+')[1]
        self.config = load_yaml(config_path)
        update_config(self.config)

        self.global_config = self.config['Global']
        torch.manual_seed(self.global_config['global_seed'])

         # init diffad model
        if self.global_config['diffusion_type'] == 'dpm':
            diffad = DDPMDiffad(self.config, num_inference_steps=10)
        elif self.global_config['diffusion_type'] == 'fm':
            diffad = FMDiffad(self.config, num_inference_steps=4)
        else:
            raise ValueError('invalid diffusion type!')

        diffad.build_diffad()
        load_save_func = LoadAndSaveCallback(ckpt_path, ignore_keys=[])
        load_save_func.load_diffad(diffad)
        self.model = diffad
        self.model.cuda()
        self.model.eval()

        vae_model_path = self.config["VAE"]['model_path']
        self.vae = AutoencoderKL.from_pretrained(vae_model_path)
        self.vae.cuda()
        self.vae.eval()

        self.vae_in_channels = self.config["VAE"]['latent_channels'] * 3 # bev+obstacle+motion RGB map
        self.vae_latent_size = self.config["VAE"]['latent_size']

        # buffer
        self.decoding_bev_flag = True
        self.prev_lidar2global = None
        self.prev_h = None
        self.prev_c = None
        self.prev_x_start = None
        self.decoded_bev = None

        inference_only_pipeline = [
            dict(type='NormalizeMultiviewImage', **dict(mean=[ 103.530, 116.280, 123.675 ], std=[ 1.0, 1.0, 1.0 ], to_rgb=False)),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='CustomCollect3D', keys=[ 'img'])
        ]

        self.inference_only_pipeline = Compose(inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.save_path = None
        # self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control = control
        self.prev_control_cache = []
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
            (self.save_path / 'pred_bev').mkdir()
   
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                     [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                     [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        
        topdown_extrinsics =  np.array([[0.0, -0.0, -1.0, 50.0], [0.0, 1.0, -0.0, 0.0], [1.0, -0.0, 0.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0      
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def sensors(self):
        sensors =[
                # camera rgb
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                # imu
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                # gps
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
            ]
        if IS_BENCH2DRIVE:
            sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
  
        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node
                }
        
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)
        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = self.step / 20
        results['box_type_3d'], _ = get_box_type('LiDAR')
  
        for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
            # cam = cam.lower()
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
        raw_theta = tick_data['compass']   if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = can_bus[0:2].copy()
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = -tick_data['angular_velocity'][-1] # TODO: check this
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = np.sqrt(can_bus[0]**2+can_bus[1]**2)

        if len(self.prev_control_cache)<10:
            ego_lcf_feat[8] = 0
        else:
            ego_lcf_feat[8] = self.prev_control_cache[0].steer

        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1

        command_onehot = np.zeros(6)
        command_onehot[command] = 1
        results['command'] = command_onehot
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_xy = rotation_matrix @ command_near_xy
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.inference_only_pipeline(results)
        self.device="cuda"


        cur_image = {}
        cur_lidar2img = defaultdict(dict)
        for i, cam in enumerate(['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']):
            cam_name = cam.lower()
            cur_image[cam_name] = results['img'].data[i].unsqueeze(0).to(self.device)  # 1,c, H, W
            cur_lidar2img[cam_name] = torch.from_numpy(self.lidar2img[cam]).unsqueeze(0).to(self.device)

        cur_lidar2global = torch.from_numpy(lidar2global).unsqueeze(0).unsqueeze(0).to(self.device) # 1,1,4,4
        if self.prev_lidar2global is not None:
            prev_lidar2global = torch.from_numpy(self.prev_lidar2global).unsqueeze(0).unsqueeze(0).to(self.device) # 1,1,4,4
            cur_lidar2global = torch.cat([prev_lidar2global, cur_lidar2global], dim=1).to(torch.float32)  # 1,2,4,4

        else:
            cur_lidar2global = None

        cur_task_label = torch.ones((4)).unsqueeze(0).to(self.device)
        cur_ego_status = torch.from_numpy(ego_lcf_feat).unsqueeze(0).to(self.device)
        cur_command = torch.from_numpy(command_onehot).unsqueeze(0).to(self.device)

        model_kwargs = {
                'task_label': cur_task_label,
                'ego_status': cur_ego_status,
                'command': cur_command
            }

        # Create sampling noise
        bs = 1
        cur_z = torch.randn((bs, self.vae_in_channels, *self.vae_latent_size[1:]), device=self.device)

        if self.prev_x_start is None:
            self.prev_x_start = torch.zeros(bs, *cur_z.shape[1:]).to(self.device) # bs,c,h,w

        cur_samples, cur_pred_plan_traj_vcs, prev_h, prev_c = self.model.sample_single_step(
                cur_image, cur_lidar2img, cur_lidar2global, self.prev_h, self.prev_c, self.prev_x_start, 
                cur_z, model_kwargs
            )

        self.prev_lidar2global = lidar2global
        self.prev_x_start = cur_samples
        self.prev_h = prev_h
        self.prev_c = prev_c
        cur_pred_plan_traj = cur_pred_plan_traj_vcs.squeeze(1).squeeze(0).cpu().numpy()

        if self.step % 10 == 0 :
            # # Channel to bs
            num_chunks = 3
            chunks = torch.chunk(cur_samples, chunks=num_chunks, dim=1) # 1,c*3,h,w
            cur_samples = torch.cat(chunks, dim=0)  # 3,c,h,w
            cur_samples = self.vae.decode(cur_samples / 0.18215).sample
            fake_bev, fake_mot, fake_plan = torch.chunk(cur_samples, chunks=num_chunks, dim=0)
            pred_image = torch.cat([fake_bev, fake_mot, fake_plan], dim=-1)[0].cpu().numpy()  # c,h,w*3

            pred_image = np.clip((pred_image + 1) * 128, 0, 255)
            pred_image = np.transpose(pred_image, (1, 2, 0)).astype(np.uint8)
            pred_image = np.ascontiguousarray(pred_image)  # transpose return non-contiguous array, and cv2 only supports contiguous arrays
            pred_bevmap_image, pred_motion_image, pred_planning_image = np.split(pred_image, 3, axis=1)
            pred_wrap_image = pred_bevmap_image

            ### draw ego
            bev_reso = self.config['Global']['bev_reso']
            x_min, y_min, _, x_max, y_max, _ = self.config['Global']['point_cloud_range']

            bev_old_h, bev_old_w = (int((y_max - y_min) / bev_reso[0]), int((x_max - x_min) / bev_reso[1]))
            bev_new_h, bev_new_w = pred_wrap_image.shape[:2]
            grid_ratio = (bev_new_h / bev_old_h, bev_new_w / bev_old_w)

            draw_ego_car(pred_wrap_image, x_max, y_max, bev_reso, grid_ratio)

            ### wrap motion
            threshold = 10
            front_mask = np.mean(pred_motion_image, axis=-1, keepdims=True)
            front_mask = (front_mask >= threshold).astype(np.uint8)
            pred_wrap_image = pred_motion_image * front_mask + pred_wrap_image * (1 - front_mask)

            ### wrap planning
            threshold = 10
            front_mask = np.mean(pred_planning_image, axis=-1, keepdims=True)
            front_mask = (front_mask >= threshold).astype(np.uint8)
            pred_wrap_image = pred_planning_image * front_mask + pred_wrap_image * (1 - front_mask)

            # draw planning on frontview
            ori_img_h, ori_img_w = self.global_config['input_size']['cam_front']
            camera_front = tick_data['imgs']['CAM_FRONT']
            cam_pts_3d = proj_traj_to_pv_wide_path(cur_pred_plan_traj, self.lidar2img, ori_img_h, ori_img_w, traj_width=2., camera_name='CAM_FRONT')
            if len(cam_pts_3d) != 0:
                # recale
                # cam_pts_3d[..., 0] = cam_pts_3d[..., 0] / ori_img_w * img_w
                # cam_pts_3d[..., 1] = cam_pts_3d[..., 1] / ori_img_h * img_h
                cam_pts_2d= cam_pts_3d[..., :2].astype(np.int32)

                # draw trajectory in faded away
                camera_front = tick_data['imgs']['CAM_FRONT']
                draw_faded_path(cam_pts_2d, camera_front, (255, 0, 0), min_alpha=0.)

            # draw planning on bev
            draw_ego_planning(cur_pred_plan_traj, pred_wrap_image, x_max, y_max, bev_reso, grid_ratio)

            command_text = ['Left', 'Right', 'Straight', 'LaneFollow', 'ChangeLaneLeft', 'ChangeLaneRight'][command]
            cv2.putText(pred_wrap_image, command_text, (bev_new_w // 2 - 50, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            tick_data['pred_bev'] = pred_wrap_image

        delta_t = self.config['Global']['sample_interval'] * 0.1
        steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(cur_pred_plan_traj, tick_data['speed'], local_command_xy, delta_t)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        self.pid_metadata = metadata_traj
        self.pid_metadata['agent'] = 'only_traj'
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)     
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan'] = cur_pred_plan_traj.tolist()
        self.pid_metadata['command'] = command
        metric_info = self.get_metric_info()
        self.metric_info[self.step] = metric_info
        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)
        self.prev_control = control
        
        if len(self.prev_control_cache)==10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control


    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
        # Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        Image.fromarray(tick_data['pred_bev']).save(self.save_path / 'pred_bev' / ('%04d.png' % frame))

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()
        
        # metric info  (add for efficiency and smoothness benchmarks)
        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
