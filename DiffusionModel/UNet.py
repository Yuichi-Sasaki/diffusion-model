import math

import torch
import torch.nn.functional as F
from torch import nn
from .modules import *

class UNet(nn.Module):
    def __init__(self, in_channels, dims=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.dims = dims

        bilinear = True
        self.inc = DoubleConv(in_channels, dims[0])
        self.down1 = Down(dims[0], dims[1])
        self.down2 = Down(dims[1], dims[2])
        factor = 2 if bilinear else 1
        self.down3 = Down(dims[2], dims[3] // factor)
        self.up1 = Up(dims[3], dims[2] // factor, bilinear)
        self.up2 = Up(dims[2], dims[1] // factor, bilinear)
        self.up3 = Up(dims[1], dims[0], bilinear)
        self.outc = OutConv(dims[0], in_channels)
        self.sa1 = SAWrapper(dims[2])
        self.sa2 = SAWrapper(dims[2])
        self.sa3 = SAWrapper(dims[1])

    def pos_encoding(self, t, channels, img_shape):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        imgH, imgW = img_shape
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, imgH, imgW)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x2 += self.pos_encoding(t, self.dims[1], img_shape=x2.shape[-2:])

        x3 = self.down2(x2)
        x3 += self.pos_encoding(t, self.dims[2], img_shape=x3.shape[-2:])
        x3 = self.sa1(x3)

        x4 = self.down3(x3)
        x4 += self.pos_encoding(t, self.dims[2], img_shape=x4.shape[-2:])
        x4 = self.sa2(x4)

        x = self.up1(x4, x3)
        x+= self.pos_encoding(t, self.dims[1], img_shape=x.shape[-2:])
        x = self.sa3(x)

        x = self.up2(x, x2)
        x+= self.pos_encoding(t, self.dims[0], img_shape=x.shape[-2:])

        x = self.up3(x, x1)
        x+= self.pos_encoding(t, self.dims[0], img_shape=x.shape[-2:])

        output = self.outc(x)

        return output

