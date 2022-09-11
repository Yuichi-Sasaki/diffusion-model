""" Parts of the U-Net model """
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, time_channels=128):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, mid_channels))
        self.double_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.double_conv_2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        hx = self.double_conv_1(x)
        ht = self.mlp(t)
        ht = torch.unsqueeze(ht,axis=-1)
        ht = torch.unsqueeze(ht,axis=-1)
        h = hx + ht
        h = self.double_conv_2(h)
        return h


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, time_channels=time_channels)

    def forward(self, x, t):
        h = self.max_pool(x)
        h = self.double_conv(h,t)
        return h


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, time_channels=128):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, time_channels=time_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, time_channels=time_channels)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear = False

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, time_channels=128)
        self.down2 = Down(128, 256, time_channels=128)
        self.down3 = Down(256, 512, time_channels=128)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, time_channels=128)
        self.up1 = Up(1024, 512 // factor, bilinear, time_channels=128)
        self.up2 = Up(512, 256 // factor, bilinear, time_channels=128)
        self.up3 = Up(256, 128 // factor, bilinear, time_channels=128)
        self.up4 = Up(128, 64, bilinear, time_channels=128)
        self.outc = OutConv(64, n_channels)

        self.t_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(dim=28)
            nn.Linear(28, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def forward(self, x, t):
        ht = self.t_embed(t)

        x1 = self.inc(x)
        x2 = self.down1(x1,ht)
        x3 = self.down2(x2,ht)
        x4 = self.down3(x3,ht)
        x5 = self.down4(x4,ht)
        x = self.up1(x5, x4,ht)
        x = self.up2(x, x3,ht)
        x = self.up3(x, x2,ht)
        x = self.up4(x, x1,ht)
        logits = self.outc(x)
        return logits
