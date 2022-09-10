import os
from inspect import isfunction

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Resize,
    ToPILImage,
    ToTensor,
)

from models.unet2 import UNet


class DiffusionModel(object):
    def __init__(self, model, timestep, working_dir="working/test"):
        self.model = model
        self.timestep = timestep
        self.working_dir = working_dir
        self.prepare_alphas()
        super().__init__()
        return

    @staticmethod
    def get_data_transform():
        # 入力画像は、(-1,1)に規格化する (ノイズのレンジとの整合性をとるため)
        transform = Compose([ToTensor(), Lambda(lambda t: (t / 255.0 * 2) - 1)])
        return transform

    @staticmethod
    def convert_Tensor_to_PIL(img):
        reverse_transform = Compose(
            [
                Lambda(lambda t: (t + 1) / 2),
                Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                Lambda(lambda t: t * 255.0),
                Lambda(lambda t: t.numpy().astype(np.uint8)),
                ToPILImage(),
            ]
        )
        return reverse_transform(img)

    def prepare_alphas(self):
        # 完全なノイズになるまで、ノイズの大きさはtに対してリニアに増えていくスケジューリングを採用
        beta_start = 0.0001
        beta_end = 0.02  # 0.02で良いの？
        betas = np.linspace(beta_start, beta_end, self.timestep)

        # KL-divergenceの計算の中に登場する各種係数を計算する
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        # sqrt_recip_alphas = np.sqrt(1.0 / alphas)

        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        # sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod

        return

    def get_alphas(self, t):
        # t時点の各種係数の値を返す
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = np.sqrt(1.0 - sqrt_alphas_cumprod_t)
        return sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t

    def train(self, dataset, epochs=5, batch_size=16):
        # Init models
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = self.model
        model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)

        # Prepare DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        # Training loop
        for iEpoch in range(epochs):
            with tqdm.tqdm(dataloader, unit="batch", total=len(dataloader)) as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {iEpoch}")
                    model.train()
                    optimizer.zero_grad()

                    if isinstance(data, list):
                        data = data[0]
                    data = data.to("cpu").detach().numpy().copy()
                    batch_size = data.shape[0]
                    # data = data.to(device)

                    ##############################
                    # 学習データの準備
                    ##############################

                    x_batch = []
                    y_batch = []
                    for itemIdx in range(batch_size):

                        ###############
                        # 画像とノイズの準備
                        ###############

                        ## 読み込まれた画像をx0と置く
                        x0 = data[itemIdx]

                        ## 入力画像と同じサイズのノイズ(正規分布)も用意する。これがモデルで推定する量になるので、gtとサフィックスをつける。
                        noise_gt = np.random.randn(*(x0.shape))

                        ###############
                        # Forward Process (q) を実行して、学習データを準備する。すべて決定論的に求まる
                        ###############

                        ## 学習させるタイムスタンプをランダムに選ぶ
                        t = np.random.randint(0, self.timestep)

                        ## その時点でのalphaなどを計算
                        coeff_x0, coeff_noise = self.get_alphas(t)

                        ## 上記alphaなどの値を使って、学習画像(xt)を準備する
                        xt = x0 * coeff_x0 + noise_gt * coeff_noise

                        ###############
                        # 学習用のバッチに仕立てる
                        ###############
                        x_batch.append(xt)
                        y_batch.append(noise_gt)

                    # ListからTensorに変換して、学習可能な変数にする
                    x_batch = torch.Tensor(np.array(x_batch))
                    y_batch = torch.Tensor(np.array(y_batch))

                    ##############################
                    # 学習の実行。
                    # ノイズが加えられた画像(img_with_noise)が入力となり、そのノイズ(noise_gt)が推定すべき量になる
                    ##############################

                    # モデルにノイズが加えられた画像を入れる。モデルは、ノイズがどのようだったかを推定する
                    noise_predicted = model(x_batch)

                    # 推定されたノイズと、GroundTruthのノイズが近いことを要求するようlossを計算する
                    loss = F.smooth_l1_loss(y_batch, noise_predicted)

                    # lossを最小化するようパラメータを最適化する
                    loss.backward()
                    optimizer.step()

                    # 表示
                    tepoch.set_postfix(loss=loss.item())
            # モデルの保存
            model_path = f"{self.working_dir}/modelss/model_{iEpoch}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.to("cpu").state_dict(), model_path)


if __name__ == "__main__":
    # UNetを作成
    model = UNet(n_channels=1)

    # DiffusionModelを作成
    diff = DiffusionModel(model, timestep=1000, working_dir="working/mnist")

    # データを読み込む
    dataset = torchvision.datasets.MNIST(
        root="./datasets",
        train=False,
        transform=diff.get_data_transform(),
        download=True,
    )

    # 学習を実行
    diff.train(dataset, epochs=5, batch_size=2)
