import torch
import torch.nn as nn
from typing import List
from timm.models.layers import DropPath

class StemBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, padding=0)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, drop_patch: float = 0., layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv_exp = nn.Conv2d(dim, 4*dim, kernel_size=1)
        self.activation = nn.GELU()
        self.pointwise_conv_red = nn.Conv2d(4*dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(
            torch.ones(dim) * layer_scale_init_value, 
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


class DownsamplingLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv2d(dim, 2*dim, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.downsampling(x)
        return x

    
class GlobalAveragePoolingHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x

class ConvNeXt_V1(nn.Module):
    def __init__(
        self, in_channels: int = 3, num_classes: int = 10, 
        depths: List = [3, 3, 9, 3], dims: List = [96, 192, 384, 768],
        drop_patch: float = 0., layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        
        stem_block = StemBlock(in_channels, dims[0])
        self.downsampling_layers = nn.ModuleList()
        self.downsampling_layers.append(stem_block)
        for i in range(3):
            downsampling_layer = DownsamplingLayer(dims[i])
            self.downsampling_layers.append(downsampling_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_patch, sum(depths))]
        current = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], drop_patch=dp_rates[current + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            current += depths[i]
            self.stages.append(stage)

        self.gap_head = GlobalAveragePoolingHead(in_channels=dims[-1], num_classes=num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsampling_layers[i](x)
            x = self.stages[i](x)
        return self.gap_head(x)

    