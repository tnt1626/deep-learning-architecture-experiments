import torch.nn as nn


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



