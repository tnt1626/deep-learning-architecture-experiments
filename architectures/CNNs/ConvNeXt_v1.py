import torch
import torch.nn as nn
from timm.models.layers import DropPath

class StemBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()

        # (N, in_channels, H, W) -> (N, out_channels, H/4, W/4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=4)

        # (N, out_channels, H/4, W/4) -> (N, out_channels, H/4, W/4)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


# TODO: ConvNeXt block
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels: int, drop_patch: float = 0., layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.pointwise_conv_exp = nn.Conv2d(in_channels, 4*in_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_conv_red = nn.Conv2d(4*in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(
            torch.ones(in_channels) * layer_scale_init_value, 
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_patch = DropPath(drop_patch) if drop_patch > 0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.pointwise_conv_exp(x)
        x = self.activation(x)
        x = self.pointwise_conv_red(x)
        if self.gamma:
            x = x * self.gamma
        x = self.drop_patch(x)
        output = input + x

        return output


