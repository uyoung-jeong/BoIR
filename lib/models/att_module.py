import torch
from torch import nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
import math
from typing import List

# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)

# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
# deeplabhead's in_channels: 2048, atrous_rates: [12, 24, 36]
class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], channels: int = 128, out_channels: int = 256, 
                residual: bool = False, downsample = None) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, channels, rate))

        modules.append(ASPPPooling(in_channels, channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #nn.Dropout(0.5),
        )
        self.residual = residual
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        res = self.project(res)
        if self.residual:
            res = res + self.downsample(x)
        return res

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations
