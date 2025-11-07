import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from model.temporal.motion_alignment import MotionAlignment


class ActionProp(nn.Module):
    def __init__(self, pc_range, grid_reso, hidden_size, patch_size, input_size,
                 in_channels, dropout=0.):

        super().__init__()
        self.motion_align = MotionAlignment(pc_range, grid_reso)
        self.in_channels = in_channels
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.droput = dropout
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

    def forward_single_frame(self, prev_x, lidar2global):
        bs, _, h, w = prev_x.shape
        if lidar2global is not None:
            align_prev_x = self.motion_align(prev_x, lidar2global[:, 0, ...], lidar2global[:, 1, ...])
        else:
            align_prev_x = torch.zeros([bs, self.in_channels, h, w]).to(prev_x.device)

        align_prev_x = self.x_embedder(align_prev_x)  # [bs, L, D]
        # dropout some tokens
        N = align_prev_x.shape[1]
        m = int(self.droput * N)
        random_indices = torch.randperm(N)[:m]
        align_prev_x[:, random_indices] = 0.
        return align_prev_x

    def forward_all_frames(self, prev_x, lidar2global):
        time = prev_x.shape[1]
        outputs = []
        for t in range(time):
            if t > 0:
                cur_lidar2global = lidar2global[:, t - 1:t + 1, ...]
            else:
                cur_lidar2global = None
            align_prev_x = self.forward_single_frame(prev_x[:, t, ...], cur_lidar2global)
            outputs.append(align_prev_x)
        outputs = torch.stack(outputs, dim=1)  # b,t,c,h,w
        return outputs

    def forward(self, prev_x, lidar2global):
        feats_dim = len(prev_x.size())
        if feats_dim == 4:  # bs, c, h, w
            return self.forward_single_frame(prev_x, lidar2global)
        elif feats_dim == 5:  # bs, time, c, h, w
            return self.forward_all_frames(prev_x, lidar2global)
