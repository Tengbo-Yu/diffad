from torch import nn
import torch.nn.functional as F
from model.base_module import BaseModule
from model.builder import NECK
from model.neck.fpn_layers import ConvBn
from model.builder import BACKBONE_OUT_CHANNELS_DICT


class BiFPNBlock(nn.Module):
    def __init__(self, num_channel, num_input_levels, output_levels_index, with_bn):
        super(BiFPNBlock, self).__init__()
        self.top_down = nn.ModuleList(
            [ConvBn(num_channel, num_channel, 1, 1, 0, with_bn=with_bn) for _ in range(num_input_levels - 1)])
        self.num_input_levels = num_input_levels
        self.output_levels_index = output_levels_index
        self.max_output_level_index = max(output_levels_index)
        assert self.max_output_level_index < num_input_levels
        self.bottom_up = nn.ModuleList(
            [ConvBn(num_channel, num_channel, 1, 1, 0, with_bn=with_bn) for _ in range(self.max_output_level_index)])

    def forward(self, inputs):
        x = inputs[-1]
        top_down_features = [x]
        for i in range(self.num_input_levels - 2, -1, -1):
            size = inputs[i].shape[-2:]
            x = inputs[i] + F.interpolate(x, size, mode='bilinear', align_corners=False)
            x = self.top_down[i](x)
            top_down_features.insert(0, x)

        x = top_down_features[0]
        bottom_up_features = [x]
        for i in range(1, self.max_output_level_index + 1):
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x = x + top_down_features[i] + inputs[i]
            x = self.bottom_up[i - 1](x)
            bottom_up_features.append(x)

        return [bottom_up_features[i] for i in self.output_levels_index]


@NECK.register()
class BiFPN(BaseModule):
    def __init__(self, neck_config: dict):
        super().__init__()
        output_channel = neck_config['output_channel']
        backbone_type = neck_config['backbone_type']
        input_channels = BACKBONE_OUT_CHANNELS_DICT[backbone_type]
        output_levels_index = neck_config['output_levels_index']
        num_input_levels = len(input_channels)
        self.with_p6 = True
        with_bn = True

        self.input_projections = nn.ModuleList(
            [ConvBn(input_channel, output_channel, 1, 1, 0, with_bn=with_bn) for input_channel in input_channels])
        if self.with_p6:
            input_channel = input_channels[-1]
            self.p6_projection = ConvBn(input_channel, output_channel, 3, 2, 1, with_bn=with_bn)
            num_input_levels += 1

        self.bifpn = BiFPNBlock(output_channel, num_input_levels, output_levels_index, with_bn)
        self.output_projections = nn.ModuleList([
            ConvBn(output_channel, output_channel, 3, 1, 1, with_bn=with_bn) for _ in range(len(output_levels_index))
        ])

        self.init_weight()

    def forward(self, x):
        proj_x = [self.input_projections[i](x[i]) for i in range(len(x))]
        if self.with_p6:
            proj_x.append(self.p6_projection(x[-1]))

        bifpn_output = self.bifpn(proj_x)
        output = [self.output_projections[i](bifpn_output[i]) for i in range(len(bifpn_output))]
        return output
