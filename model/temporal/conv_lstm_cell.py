import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hidden_c, kernel, stride):
        super().__init__()

        self.hidden_c = hidden_c
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        padding = (kernel - 1) // 2

        self.conv = nn.Conv2d(in_channels=in_c + hidden_c,
                              out_channels=4 * hidden_c,
                              kernel_size=kernel,
                              stride=stride,
                              padding=padding)

        self._init_weight()

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, prev_h, prev_c):
        combined_x = torch.cat((x, prev_h), 1)
        fg, ig, gg, og = torch.split(self.conv(combined_x), self.hidden_c, dim=1)

        fg = self.sigmoid(fg)
        ig = self.sigmoid(ig)
        gg = self.tanh(gg)
        og = self.sigmoid(og)

        cs = fg * prev_c + ig * gg
        hs = og * self.tanh(cs)
        return hs, cs
