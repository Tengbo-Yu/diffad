import torch
import torch.nn as nn
from model.builder import build_backbone
from model.builder import build_neck
from model.builder import build_view_transformation
from model.builder import build_temporal
from model.builder import LDM_MODEL
from model.temporal.action_prop import ActionProp
from model.dit_modules.postprocess_net import PostProcessNet
from util.flatten_util import flatten_sequential_data
from util.flatten_util import unflatten_sequential_data
from util.flatten_util import select_data_dict_by_t_idx
from model.base_module import BaseModule


class BaseDiffad(BaseModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.global_config = config['Global']
        self.vae_config = self.config["VAE"]

        self._freeze_module_dict: dict = {}
        self._load_module_dict: dict = {}

        self.camera_to_group = {}
        self.model = nn.ModuleDict()

    def build_diffad(self):
        self._build_backbone()
        self._build_neck()
        self._build_view_transformation()
        self._build_temporal()
        self._build_ldm()
        self._build_action_prop()
        self._build_postprocess()

    def _build_backbone(self):
        module_name = 'backbone_'
        backbone_config = self.config['Backbone']
        for group in backbone_config:
            group_config = backbone_config[group]
            for input_camera in group_config['input_cameras']:
                self.camera_to_group[input_camera] = group

            self._freeze_module_dict[module_name + group] = group_config.get('freeze', False)
            self._load_module_dict[module_name + group] = group_config.get('load', False)
            self.model[module_name + group] = build_backbone(group_config['type'], group_config)

    def _build_neck(self):
        module_name = 'neck_'
        neck_config = self.config['Neck']
        for group in neck_config:
            group_config = neck_config[group]
            backbone_type = self.config['Backbone'][group]['type']
            backbone_subtype = self.config['Backbone'][group]['sub_type']
            group_config['backbone_type'] = backbone_type + backbone_subtype
            self._freeze_module_dict[module_name + group] = group_config.get('freeze', False)
            self._load_module_dict[module_name + group] = group_config.get('load', False)
            self.model[module_name + group] = build_neck(group_config['type'], group_config)

    def _build_view_transformation(self):
        module_name = 'view_transformation'
        view_transformation_config = self.config['ViewTransformation']
        view_transformation_config['Global'] = self.global_config
        view_transformation_type = view_transformation_config['type']
        self._freeze_module_dict[module_name] = view_transformation_config.get('freeze', False)
        self._load_module_dict[module_name] = view_transformation_config.get('load', False)
        view_transformation_obj = build_view_transformation(view_transformation_type, view_transformation_config)
        self.model[module_name] = view_transformation_obj

    def _build_temporal(self):
        if 'Temporal' in self.config:
            module_name = 'temporal'
            temporal_config = self.config['Temporal']
            self._freeze_module_dict[module_name] = temporal_config.get('freeze', False)
            self._load_module_dict[module_name] = temporal_config.get('load', False)
            temporal_type = temporal_config['type']
            temporal_obj = build_temporal(temporal_type, temporal_config)
            self.model[module_name] = temporal_obj

    def _build_ldm(self):
        module_name = 'ldm'
        ldm_config = self.config["LDM"]
        dit_model_class = LDM_MODEL.get(ldm_config['model_type'])
        latent_size = self.vae_config['latent_size']
        in_channels = self.vae_config['latent_channels'] * 3  # bev+obstacle+motion RGB map
        bevfeat_dim = self.config['ViewTransformation']['hidden_dim']
        use_gradient_checkpointing = ldm_config["use_gradient_checkpointing"]
        ldm = dit_model_class(
            input_size=latent_size[1:],
            in_channels=in_channels,
            bevfeat_dim=bevfeat_dim,
            learn_sigma=False,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        self._freeze_module_dict[module_name] = ldm_config.get('freeze', False)
        self._load_module_dict[module_name] = ldm_config.get('load', False)
        self.model[module_name] = ldm

    def _build_action_prop(self):
        if 'Temporal' in self.config and 'ActionProp' in self.config:
            ldm = self.model['ldm']
            module_name = 'action_prop'
            action_prop_config = self.config["ActionProp"]
            prev_x_model = ActionProp(
                pc_range=self.global_config['point_cloud_range'],
                grid_reso=self.global_config['grid_reso'],
                hidden_size=ldm.hidden_size,
                patch_size=ldm.patch_size,
                input_size=ldm.input_size,
                in_channels=ldm.in_channels,
                dropout=action_prop_config['dropout']
            )
            self._freeze_module_dict[module_name] = action_prop_config.get('freeze', False)
            self._load_module_dict[module_name] = action_prop_config.get('load', False)
            self.model[module_name] = prev_x_model

    def _build_postprocess(self):
        if 'Temporal' in self.config and 'PostProcessNet' in self.config:
            ldm = self.model['ldm']
            module_name = 'postprocess'
            postprocess_config = self.config["PostProcessNet"]
            postprocess_net = PostProcessNet(
                postprocess_config,
                input_size=ldm.input_size,
                in_channels=ldm.in_channels,
                patch_size=ldm.patch_size,
                hidden_size=ldm.hidden_size,
                pc_range=self.global_config['point_cloud_range']
            )
            self._freeze_module_dict[module_name] = postprocess_config.get('freeze', False)
            self._load_module_dict[module_name] = postprocess_config.get('load', False)
            self.model[module_name] = postprocess_net

    def should_freeze(self, module_name):
        res = self._freeze_module_dict.get(module_name, False)
        assert isinstance(res, bool)
        return res

    def should_load(self, module_name):
        res = self._load_module_dict.get(module_name, False)
        assert isinstance(res, bool)
        return res

    def bevnet_forward(self, image, lidar2img, lidar2global, x_start):
        """
            image ({'cam_id': [b*t,c,h,w]})
            lidar2global (bs,ts, 4x4)
        """
        bs, ts = lidar2global.shape[:2]
        image = flatten_sequential_data(image)  # {'cam_id': [b*t,c,h,w]}
        # forward backbone+fpn
        multiview_feature_dict = {}
        for cam_name, image_data in image.items():
            group_name = self.camera_to_group[cam_name]
            backbone_out = self.model['backbone_' + group_name](image_data)
            neck_out = self.model['neck_' + group_name](backbone_out)
            multiview_feature_dict[cam_name] = neck_out

        lidar2img = flatten_sequential_data(lidar2img)  # # {'cam_id': [b*t,4,4]}
        bev_feature = self.model['view_transformation'](multiview_feature_dict, lidar2img)  # bs*ts,c,h,w

        # forward temporal module
        bev_feature, prop_prev_x_start = self._process_temporal_module(bev_feature, lidar2global, x_start, bs, ts)
        return bev_feature, prop_prev_x_start

    def _process_temporal_module(self, bev_feature, lidar2global, x_start, bs, ts):
        if 'temporal' in self.model:
            bev_feature = unflatten_sequential_data(bev_feature, bs, ts)  # bs,ts,c,h,w
            bev_feature = self.model['temporal'](bev_feature, prev_hs=None, prev_cs=None,
                                                 ego_pose=lidar2global)  # bs,ts,c,h,w
            bev_feature = flatten_sequential_data(bev_feature)  # bs*ts,c,h,w

            # action prop
            if 'action_prop' in self.model:
                x_start = unflatten_sequential_data(x_start, bs, ts)  # bs,ts,c,h,w
                # padding zero for the first frame
                init_prev_x = torch.zeros_like(x_start[:, :1])
                prev_x_start = torch.cat([init_prev_x, x_start[:, :-1]], dim=1)
                prop_prev_x_start = self.model['action_prop'](prev_x_start, lidar2global=lidar2global)  # bs,ts,c,h,w
                prop_prev_x_start = flatten_sequential_data(prop_prev_x_start)  # bs*ts,c,h,w
            else:
                prop_prev_x_start = None
        else:
            prop_prev_x_start = None
        return bev_feature, prop_prev_x_start

    def bevnet_recurrent_forward(self, image, lidar2img, lidar2global, prev_h, prev_c, prev_x_start):
        # forward backbone+fpn
        multiview_feature_dict = {}
        # 'camera_front', 'fisheye_front', 'fisheye_rear', 'fisheye_left', 'fisheye_right', 'camera_rear'
        for cam_name, image_data in image.items():
            group_name = self.camera_to_group[cam_name]
            backbone_out = self.model['backbone_' + group_name](image_data)
            neck_out = self.model['neck_' + group_name](backbone_out)
            multiview_feature_dict[cam_name] = neck_out

        # forward view transformation
        bev_feature = self.model['view_transformation'](multiview_feature_dict, lidar2img)  # bs,c,h,w

        # forward temporal module
        if 'temporal' in self.model:
            prev_h, prev_c = self.model['temporal'](bev_feature, prev_hs=prev_h, prev_cs=prev_c,
                                                    ego_pose=lidar2global)  # bs,c,h,w
            prop_prev_x_start = self.model['action_prop'](prev_x_start, lidar2global=lidar2global) \
                if 'action_prop' in self.model else None
        else:
            prev_h = bev_feature
            prev_c = bev_feature
            prop_prev_x_start = None

        return prev_h, prev_c, prop_prev_x_start

    def recurrent_sample(self, z, image, lidar2img, lidar2global,
                         task_label, ego_status, command):
        """
            z (bs*ts, c, h, w)
            image ({'cam_id': [b,t,c,h,w]})
        """
        sample_list = []
        pred_trajs_list = []
        bs, ts = lidar2global.shape[:2]
        prev_h, prev_c = None, None
        prev_x_start = torch.zeros(bs, *z.shape[1:], device=z.device)  # bs,c,h,w
        for t_idx in range(ts):
            # select data for bevformer
            cur_image = select_data_dict_by_t_idx(image, t_idx, to_cpu=False)
            cur_lidar2img = select_data_dict_by_t_idx(lidar2img, t_idx, to_cpu=False)
            cur_lidar2global = lidar2global[:, t_idx - 1: t_idx + 1, ...] if t_idx > 0 else None

            # select data for diffusion
            idx = bs * t_idx
            cur_z = z[idx: idx + bs]  # bs c h w
            cur_task_label = task_label[idx: idx + bs]
            cur_ego_status = ego_status[idx: idx + bs]
            cur_command = command[idx: idx + bs]

            model_kwargs = {
                'task_label': cur_task_label,
                'ego_status': cur_ego_status,
                'command': cur_command
            }
            samples, pred_trajs, prev_h, prev_c = self.sample_single_step(
                cur_image, cur_lidar2img, cur_lidar2global, prev_h, prev_c, prev_x_start,
                cur_z, model_kwargs
            )
            prev_x_start = samples
            sample_list.append(samples)
            pred_trajs_list.append(pred_trajs)

        return torch.cat(sample_list, dim=0), torch.cat(pred_trajs_list, dim=0)

    def sample_single_step(self, cur_image, cur_lidar2img, cur_lidar2global, prev_h, prev_c, prev_x_start,
                           cur_z, model_kwargs):
        """
            This method should be implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def forward(self, latent_z, plan_traj, plan_traj_mask, image, lidar2img, lidar2global, model_kwargs):
        """
            This method should be implemented in the subclass.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")
