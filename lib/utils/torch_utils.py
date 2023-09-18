import torch
from torch import nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class NormConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dilation,
            norm='inst', bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.norm = norm
        if norm == 'inst':
            self.norm_layer = nn.InstanceNorm2d(out_planes)
        elif norm == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm_layer(x)
        return x
