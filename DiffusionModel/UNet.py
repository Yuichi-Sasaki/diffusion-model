import math

import torch
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Transformerのpositional embeddingと同じ仕組みで、時間をembedするベクトルを計算する。
    詳細はこちらなどを参照。https://zenn.dev/attentionplease/articles/1a01887b783494
    dim: もともとは、単語ベクトルの次元数。今回であれば、例えば画像のピクセル数などで代用
    """

    def __init__(self, dim=28):
        super().__init__()
        assert dim % 2 == 0, "dim cannot be an odd number."
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * (-embeddings))
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim=None,
        multiplier=2,
        normalize=True,
    ):
        super().__init__()

        # Time embedding module
        if time_embed_dim:
            self.time_embed_module = nn.Sequential(
                nn.GELU(), nn.Linear(time_embed_dim, in_channels)
            )
        else:
            self.time_embed_module = None

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if normalize else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * multiplier, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * multiplier),
            nn.Conv2d(
                out_channels * multiplier, out_channels, kernel_size=3, padding=1
            ),
        )

        if in_channels == out_channels:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, time_embed=None):
        h = self.conv1(x)

        if self.time_embed_module is not None:
            c = self.time_embed_module(time_embed)
            c = c.unsqueeze(axis=-1).unsqueeze(axis=-1)
            h = h + c

        h = self.block(h)

        h = h + self.conv2(x)

        return h


class DownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim=None,
        sampling=None,
        attention=False,
        multiplier=2,
    ):
        super().__init__()
        self.block1 = ConvNextBlock(
            in_channels,
            out_channels,
            time_embed_dim=time_embed_dim,
            multiplier=multiplier,
        )
        self.block2 = ConvNextBlock(
            out_channels,
            out_channels,
            time_embed_dim=time_embed_dim,
            multiplier=multiplier,
        )
        if attention:
            raise NotImplementedError()
        else:
            self.attention = nn.Identity()
        self.norm = nn.GroupNorm(1, out_channels)

        if sampling == "up":
            self.sampling = UpSample(out_channels)
        elif sampling == "down":
            self.sampling = DownSample(out_channels)
        else:
            self.sampling = nn.Identity()

    def forward(self, x, time_embed=None):
        h = x
        h = self.block1(h, time_embed=time_embed)
        h = self.block2(h, time_embed=time_embed)
        h = self.attention(h)
        h = self.norm(h)
        h = self.sampling(h)
        return h


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dims=[32, 64, 128, 256, 512],
        enable_time_embedding=True,
        use_attention=False,
        representative_time_dim=32,
    ):
        super().__init__()

        # time embeddings
        if enable_time_embedding:
            time_embed_dim = representative_time_dim * 4
            self.time_embed_module = nn.Sequential(
                SinusoidalPositionEmbeddings(dim=representative_time_dim),
                nn.Linear(representative_time_dim, time_embed_dim),
                nn.GELU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None
            self.time_embed_module = None

        # layers
        self.conv1 = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)
        self.block_down_1 = Block(
            dims[0],
            dims[1],
            time_embed_dim,
            sampling="down",
            attention=use_attention,
        )
        self.block_down_2 = Block(
            dims[1],
            dims[2],
            time_embed_dim,
            sampling="down",
            attention=use_attention,
        )
        self.block_down_3 = Block(
            dims[2],
            dims[3],
            time_embed_dim,
            sampling="down",
            attention=use_attention,
        )
        self.block_down_4 = Block(
            dims[3],
            dims[4],
            time_embed_dim,
            sampling="down",
            attention=use_attention,
        )

        self.block_mid_1 = Block(
            dims[4],
            dims[4],
            time_embed_dim,
            attention=use_attention,
        )

        self.block_mid_2 = Block(
            dims[4],
            dims[4],
            time_embed_dim,
            attention=use_attention,
        )

        self.block_up_4 = Block(
            dims[4] * 2,
            dims[3],
            time_embed_dim,
            sampling="up",
            attention=use_attention,
        )

        self.block_up_3 = Block(
            dims[3] * 2,
            dims[2],
            time_embed_dim,
            sampling="up",
            attention=use_attention,
        )

        self.block_up_2 = Block(
            dims[2] * 2,
            dims[1],
            time_embed_dim,
            sampling="up",
            attention=use_attention,
        )

        self.block_up_1 = Block(
            dims[1] * 2,
            dims[0],
            time_embed_dim,
            sampling="up",
            attention=use_attention,
        )

        self.conv2 = nn.Conv2d(dims[0] * 2, in_channels, kernel_size=1)

    def forward(self, x, time):
        if self.time_embed_module is not None:
            t = self.time_embed_module(time)
        else:
            t = None

        h0 = self.conv1(x)

        print(h0.shape)
        h1 = self.block_down_1(h0, t)
        print(h1.shape)
        h2 = self.block_down_2(h1, t)
        print(h2.shape)
        h3 = self.block_down_3(h2, t)
        print(h3.shape)
        h4 = self.block_down_4(h3, t)
        print(h4.shape)

        h = h4
        h = self.block_mid_1(h, t)
        h = self.block_mid_2(h, t)

        h = self.block_up_4(torch.cat((h, h4), dim=1), t)
        print(h.shape, h3.shape)
        h = self.block_up_3(torch.cat((h, h3), dim=1), t)
        h = self.block_up_2(torch.cat((h, h2), dim=1), t)
        h = self.block_up_1(torch.cat((h, h1), dim=1), t)

        h = self.conv2(h)

        return h
