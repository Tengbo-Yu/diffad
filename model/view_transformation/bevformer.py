import torch
from torch import nn
from model.base_module import BaseModule
from model.builder import VIEWTRANSFORMATION
from model.view_transformation.cosformer import CosformerAttention
from ops.bevformer.modules import MSDeformAttn
from util.transformer_util import _get_clones
from util.transformer_util import _get_activation_fn


@VIEWTRANSFORMATION.register()
class BevFormer(BaseModule):
    def __init__(self, view_transformation_config: dict):
        super().__init__()

        self.d_model = view_transformation_config['hidden_dim']
        nhead = view_transformation_config['nhead']
        num_decoder_layers = view_transformation_config['num_decoder_layers']
        dim_feedforward = view_transformation_config['dim_feedforward']
        dropout = view_transformation_config['dropout']
        activation = "relu"
        dec_n_points = view_transformation_config['dec_n_points']

        self.global_config = view_transformation_config['Global']
        self.camera_names = list(self.global_config['input_size'].keys())
        self.num_cameras = len(self.camera_names)
        self.num_feats = len(view_transformation_config['fpn_level'])

        pc_range = self.global_config['point_cloud_range']
        self.pc_range = pc_range
        grid_reso = view_transformation_config['grid_resolution']
        H = int((pc_range[4] - pc_range[1]) / grid_reso[0])
        W = int((pc_range[3] - pc_range[0]) / grid_reso[1])
        Z = int((pc_range[5] - pc_range[2]) / grid_reso[2])
        grid = self.generate_grid(H, W, Z)
        self.register_buffer("grid", grid)
        self.bev_h, self.bev_w, self.bev_z = self.grid.shape[:3]
        self.num_queries = self.bev_h * self.bev_w

        # init input projection layer
        input_proj_dict = {}
        for cam_idx in range(self.num_cameras):
            input_proj_dict[str(cam_idx)] = nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model, kernel_size=1),
                nn.BatchNorm2d(self.d_model)
            )
        self.input_proj = nn.ModuleDict(input_proj_dict)

        # init deformable transformer
        decoder_layer = DeformableTransformerLayer(d_model=self.d_model,
                                                   d_ffn=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                   n_levels=self.num_cameras * self.num_feats * self.bev_z,
                                                   n_heads=nhead,
                                                   n_points=dec_n_points)
        self.transformer = DeformableTransformer(decoder_layer, num_decoder_layers)

        # init embeddings
        self.query_embed = nn.Embedding(self.num_queries, self.d_model * 2)
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feats, self.d_model))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cameras, self.d_model))

        # init weight
        self._init_weight()

    @staticmethod
    def generate_grid(H, W, Z, num_points_in_pillar=4):
        """
        Generates a 3D grid with coordinates centered and normalized within [0,1].

        Parameters:
        -----------
        H : int
            The height of the grid, representing the number of points along the y-axis.
        W : int
            The width of the grid, representing the number of points along the x-axis.
        Z : int
            The depth of the grid, representing the number of points along the z-axis.
        num_points_in_pillar : int, optional (default=4)
            The number of discrete points to sample within each pillar along the z-axis.

        Returns:
        --------
        grid : torch.Tensor
            A tensor of shape (H, W, num_points_in_pillar, 3) containing the normalized
            coordinates (x, y, z) of each point in the grid. The x, y, and z values are
            scaled to the range [0, 1].
        """
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar).view(1, 1, -1).expand(H, W, num_points_in_pillar) / Z
        xs = torch.linspace(0.5, W - 0.5, W).view(1, W, 1).expand(H, W, num_points_in_pillar) / W
        ys = torch.linspace(0.5, H - 0.5, H).view(H, 1, 1).expand(H, W, num_points_in_pillar) / H
        grid = torch.stack((xs, ys, zs), -1)  # Shape: (H, W, num_points_in_pillar, 3)
        return grid

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

    def process_input(self, feats):
        src_flatten = []
        spatial_shapes = []
        for feat_idx in range(self.num_feats):
            cur_fpn_lvl_feat = feats[feat_idx]
            num_cameras = len(cur_fpn_lvl_feat)
            for cam_idx in range(num_cameras):
                src = cur_fpn_lvl_feat[cam_idx]
                bs, _, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)

                src = self.input_proj[str(cam_idx)](src)
                src = src + self.cams_embeds[cam_idx][None, :, None, None]
                src = src + self.level_embeds[feat_idx][None, :, None, None]
                src = src.flatten(2).transpose(1, 2)  # B, C, H, W --> B, C, H*W --> B, H*W, C
                src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, 1)  # B, N*H*W, C
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        return src_flatten, spatial_shapes, level_start_index

    @torch.no_grad()
    def point_sampling(self, reference_points, pc_range, lidar2img, device):
        lidar2img_list = []
        for camera_name in self.camera_names:
            lidar2img_list.append(lidar2img[camera_name].to(torch.float32).to(device))

        lidar2img = torch.stack(lidar2img_list, dim=1)  # bs,num_cam,4,4
        bs, num_cam = lidar2img.shape[:2]

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        H, W, Z, _ = reference_points.shape
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.view(
            1, H, W, Z, 1, 4).repeat(
            bs, 1, 1, 1, num_cam, 1)  # bs,h,w,z,num_cam,4

        reference_points = reference_points.flatten(1, 2).unsqueeze(-1)  # bs,query,z,num_cam,4,1
        _, num_query = reference_points.size()[:2]

        lidar2img = lidar2img.view(bs, 1, 1, num_cam, 4, 4).repeat(1, num_query, Z, 1, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)

        eps = 1e-5
        valid_cam_points = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        img_h, img_w = self.global_config['input_size'][self.camera_names[0]]  # shapes of all views are the same
        reference_points_cam[..., 0] /= img_w
        reference_points_cam[..., 1] /= img_h

        valid_cam_points = (valid_cam_points & (reference_points_cam[..., 1:2] > 0.0)
                            & (reference_points_cam[..., 1:2] < 1.0)
                            & (reference_points_cam[..., 0:1] < 1.0)
                            & (reference_points_cam[..., 0:1] > 0.0))

        valid_cam_points = valid_cam_points.repeat(1, 1, 1, 1, 2)  # bs, query, z, num_cam, 2

        ref_points = reference_points_cam.view(
            bs, num_query, Z, 1, num_cam, 2).repeat(1, 1, 1, self.num_feats, 1, 1)
        valid_ref_points = valid_cam_points.view(
            bs, num_query, Z, 1, num_cam, 2).repeat(1, 1, 1, self.num_feats, 1, 1)

        ref_points = ref_points.reshape(bs, num_query, Z * self.num_feats * num_cam, 2)
        valid_ref_points = valid_ref_points.reshape(bs, num_query, Z * self.num_feats * num_cam, 2)

        valid_ref_points = valid_ref_points.to(bool)
        valid_weight = valid_ref_points[:, :, :, 0].sum(dim=2, keepdim=True)  # bs, num_query, 1
        if valid_weight.min() == 0:
            valid_weight = torch.clamp(valid_weight, min=1)

        return ref_points, valid_ref_points, valid_weight

    def forward(self, x, lidar2img):
        feats = []
        all_feat_size = 0
        for feat_idx in range(self.num_feats):
            cur_fpn_lvl_feats = []
            for cam_id in x:
                cur_fpn_lvl_feats.append(x[cam_id][feat_idx])
            feats.append(cur_fpn_lvl_feats)
            for feat_per_cam in cur_fpn_lvl_feats:
                bs, c, h, w = feat_per_cam.shape
                all_feat_size += h * w

        src_flatten, spatial_shapes, level_start_index = self.process_input(feats)

        query_embeds = self.query_embed.weight
        query_embed, tgt = torch.split(query_embeds, self.d_model, dim=1)
        bs = src_flatten.shape[0]
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        ref_points, valid_ref_points, valid_weight = self.point_sampling(self.grid, self.pc_range, lidar2img,
                                                                         src_flatten.device)
        ref_points[~valid_ref_points] = -99

        spatial_shapes = spatial_shapes.repeat(self.bev_z, 1)
        src_flatten = src_flatten.repeat(1, self.bev_z, 1)
        z_start_index = []
        for i in range(self.bev_z):
            next_z_start_index = (level_start_index + i * all_feat_size).clone().detach()
            z_start_index.append(next_z_start_index)
        level_start_index = torch.cat(z_start_index)

        hs = self.transformer(tgt, ref_points, src_flatten, spatial_shapes, level_start_index, query_embed)
        hs = hs / valid_weight

        bev_feat = hs.permute(0, 2, 1)  # b, c, hw
        bev_feat = bev_feat.reshape(bs, self.d_model, self.bev_h, self.bev_w)

        """
        convert to lidar coords system for spatially aligning with the bev map
            y(front)
            ↑
            |
            |
            0 -----> x(right)
        """
        bev_feat = torch.flip(bev_feat, dims=[2])  # vertial filp
        return bev_feat


class DeformableTransformerLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = CosformerAttention(d_model, n_heads, dropout_rate=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)).transpose(0, 1)  # cosformer
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points,
                               src, src_spatial_shapes, level_start_index)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformer(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, query_pos=None):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, reference_points, src, src_spatial_shapes, src_level_start_index)

        return output
