import torch
from model.builder import TEMPORAL
from model.base_module import BaseModule
from model.temporal.conv_lstm_cell import ConvLSTMCell
from model.temporal.motion_alignment import MotionAlignment


@TEMPORAL.register()
class ConvLSTM(BaseModule):
    def __init__(self, temporal_config: dict):

        super().__init__()
        self.hidden_c = temporal_config['hidden_c']
        self.motion_align = MotionAlignment(temporal_config['pc_range'], temporal_config['grid_reso'])
        self.cell = ConvLSTMCell(in_c=temporal_config['in_c'],
                                 hidden_c=temporal_config['hidden_c'],
                                 kernel=temporal_config['kernel'],
                                 stride=temporal_config['stride'])

    def forward_single_frame(self, feat, prev_hs, prev_cs, cur_ego_pose):
        bs, _, h, w = feat.shape
        if prev_hs is None and prev_cs is None:
            prev_hs = torch.zeros([bs, self.hidden_c, h, w]).to(feat.device)
            prev_cs = torch.zeros([bs, self.hidden_c, h, w]).to(feat.device)

        if cur_ego_pose is not None:
            align_prev_hs = self.motion_align(prev_hs, cur_ego_pose[:, 0, ...], cur_ego_pose[:, 1, ...])
            align_prev_cs = self.motion_align(prev_cs, cur_ego_pose[:, 0, ...], cur_ego_pose[:, 1, ...])
        else:
            align_prev_hs = prev_hs
            align_prev_cs = prev_cs

        hs, cs = self.cell(feat, align_prev_hs, align_prev_cs)

        return hs, cs

    def forward_all_frames(self, feats, prev_hs, prev_cs, ego_pose):
        time = feats.shape[1]
        outputs = []
        for t in range(time):
            if t > 0:
                cur_ego_pose = ego_pose[:, t - 1:t + 1, ...]
            else:
                cur_ego_pose = None
            prev_hs, prev_cs = self.forward_single_frame(feats[:, t, ...], prev_hs, prev_cs, cur_ego_pose)
            outputs.append(prev_hs)
        outputs = torch.stack(outputs, dim=1)  # b,t,c,h,w
        return outputs

    def forward(self, feats, prev_hs, prev_cs, ego_pose):
        feats_dim = len(feats.size())
        if feats_dim == 4:  # bs, c, h, w
            return self.forward_single_frame(feats, prev_hs, prev_cs, ego_pose)
        elif feats_dim == 5:  # bs, time, c, h, w
            return self.forward_all_frames(feats, prev_hs, prev_cs, ego_pose)

    def onnx_export(self, feat, align_prev_hs, align_prev_cs):
        hs, cs = self.cell(feat, align_prev_hs, align_prev_cs)
        return hs, cs
