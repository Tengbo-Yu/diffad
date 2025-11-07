from torch import nn


class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, with_bn):
        super().__init__()
        if with_bn:
            self.net = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
                                     nn.BatchNorm2d(out_c))
        else:
            self.net = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=True)

    def forward(self, x):
        return self.net(x)
