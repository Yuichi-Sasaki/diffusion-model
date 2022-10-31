import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
from models.unet_embed import UNet
from PIL import Image
from torch.optim import Adam
from torch_ema import ExponentialMovingAverage
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomHorizontalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)


class DiffusionModel(object):
    def __init__(self, model, timesteps, gpu="-1", working_dir="working/test"):
        self.model = model
        self.timesteps = timesteps
        self.working_dir = working_dir

        self.gpu_ids = [int(x) for x in gpu.split(",")] if gpu != "-1" else None
        self.n_gpus = len(self.gpu_ids) if gpu != "-1" else 0
        self.device = "cuda:{}".format(self.gpu_ids[0]) if self.n_gpus > 0 else "cpu"
        self.do_clip_noise = False
        self.clip_grad = 1.0

        self.output_fig_size = (20, 20)
        self.prepare_params()
        super().__init__()
        return

    @staticmethod
    def get_data_transform():
        # 入力画像は、(-1,1)に規格化する (ノイズのレンジとの整合性をとるため)
        transform = Compose(
            [RandomHorizontalFlip(), ToTensor(), Lambda(lambda t: (t * 2) - 1)]
        )
        return transform

    @staticmethod
    def convert_Tensor_to_PIL(img):
        reverse_transform = Compose(
            [
                ToTensor(),
                Lambda(lambda t: (t + 1) / 2),
                Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                Lambda(lambda t: t * 255.0),
                Lambda(lambda t: t.numpy().astype(np.uint8)),
                ToPILImage(),
            ]
        )
        return reverse_transform(img)

    def prepare_params(self):
        # beta
        beta_start = 0.0001
        beta_end = 0.02  # 論文での設定値
        self.beta = beta = np.linspace(beta_start, beta_end, self.timesteps)

        # alpha
        self.alpha = alpha = 1.0 - beta

        # alpha_bar (cumproduct)
        self.alpha_bar = alpha_bar = np.cumprod(alpha, axis=0)
        alpha_bar_prev = alpha_bar.copy()
        alpha_bar_prev[1:] = alpha_bar[:-1]
        alpha_bar_prev[0] = 1.0
        self.alpha_bar_prev = alpha_bar_prev

        return

    def get_coeffs_for_training(self, t):
        # Algorithm1: Training

        coeff_x0 = np.sqrt(self.alpha_bar[t])  # x0に掛かる係数
        coeff_noise = np.sqrt(1.0 - self.alpha_bar[t])  # epsilonに掛かる係数

        return coeff_x0, coeff_noise

    def get_coeffs_for_sampling(self, t):
        # Algorithm2: Sampling

        coeff_normalize = 1.0 / np.sqrt(self.alpha[t])  # 全体に掛かる規格化係数
        coeff_noise = (1.0 - self.alpha[t]) / np.sqrt(
            1.0 - self.alpha_bar[t]
        )  # epsilon_thetaに掛かる係数
        coeff_additional_noise = np.sqrt(
            (1.0 - self.alpha_bar_prev[t]) / (1.0 - self.alpha_bar[t]) * self.beta[t]
        )  # zに掛かる係数

        return (coeff_normalize, coeff_noise, coeff_additional_noise)

    def train(
        self,
        dataset,
        epochs=10,
        batch_size=128,
        lr=2e-4,
        num_workers=2,
        ema_decay=None,
        save_freq=1,
        generate_freq=1,
        plot_timesteps=[0],
    ):
        assert max(plot_timesteps) < self.timesteps
        # Init models
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu
        self.model.to(self.device)
        if self.n_gpus > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        optimizer = Adam(self.model.parameters(), lr=lr)
        if ema_decay is not None:
            ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)

        # Prepare DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=num_workers,
        )

        # Training loop
        for iEpoch in range(epochs):
            loss_avg = []
            with tqdm.tqdm(dataloader, unit="batch", total=len(dataloader)) as tepoch:
                for data in tepoch:
                    tepoch.set_description(f"Epoch {iEpoch}")
                    self.model.train(True)
                    optimizer.zero_grad()

                    if isinstance(data, list):
                        data = data[0]
                    data = data.to("cpu").detach().numpy().copy()
                    batch_size = data.shape[0]
                    img_shape = data.shape[1:]

                    ##############################
                    # 学習データの準備
                    ##############################

                    x_batch = []
                    t_batch = []
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
                        t = np.random.randint(0, self.timesteps)

                        ## その時点での各種係数を計算
                        coeff_x0, coeff_noise = self.get_coeffs_for_training(t)

                        ## 上記alphaなどの値を使って、学習画像(xt)を準備する
                        xt = x0 * coeff_x0 + noise_gt * coeff_noise

                        ###############
                        # 学習用のバッチに仕立てる
                        ###############
                        x_batch.append(xt)
                        t_batch.append([t])
                        y_batch.append(noise_gt)

                    # ListからTensorに変換して、学習可能な変数にする
                    x_batch = torch.Tensor(np.array(x_batch)).to(self.device)
                    t_batch = torch.Tensor(np.array(t_batch)).to(self.device)
                    y_batch = torch.Tensor(np.array(y_batch)).to(self.device)

                    ##############################
                    # 学習の実行。
                    # ノイズが加えられた画像(img_with_noise)が入力となり、そのノイズ(noise_gt)が推定すべき量になる
                    ##############################

                    # モデルにノイズが加えられた画像を入れる。モデルは、ノイズがどのようだったかを推定する
                    noise_predicted = self.model(x_batch, t_batch)

                    # 推定されたノイズと、GroundTruthのノイズが近いことを要求するようlossを計算する
                    loss = F.mse_loss(noise_predicted, y_batch)

                    # lossを最小化するようパラメータを最適化する
                    loss.backward()
                    if self.clip_grad is not None:
                        torch.nn.utils.clip_grad_value_(
                            self.model.parameters(), self.clip_grad
                        )
                    optimizer.step()
                    if ema_decay is not None:
                        ema.update()
                        ema.copy_to()

                    # 表示
                    loss_avg.append(loss.item())
                    tepoch.set_postfix(loss=sum(loss_avg) / len(loss_avg))

            self.model.train(False)

            # モデルの保存
            if iEpoch % save_freq == 0:
                model_path = f"{self.working_dir}/models/model_{iEpoch:04d}.pt"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                if self.n_gpus > 1:
                    torch.save(self.model.module.to("cpu").state_dict(), model_path)
                else:
                    torch.save(self.model.to("cpu").state_dict(), model_path)

            # 画像の保存
            if iEpoch % generate_freq == 0:
                image_path = f"{self.working_dir}/images/image_{iEpoch:04d}.png"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                all_imgs = []
                for i in range(5):
                    imgs = self.generate(img_shape=img_shape, seed=i)
                    imgs_pick = [imgs[j] for j in plot_timesteps]
                    all_imgs.append(imgs_pick)
                self.plot(all_imgs, output=image_path)

    @torch.no_grad()
    def generate(self, img_shape, seed=0):
        np.random.seed(seed)
        self.model.train(False)
        self.model.to(self.device)
        # start from pure noise (for each example in the batch)
        imgs = []

        # Start from pure noise
        img = np.random.randn(*(img_shape))

        # loop to generate
        for t in range(0, self.timesteps)[::-1]:

            (
                coeff_normalize,
                coeff_noise,
                coeff_additional_noise,
            ) = self.get_coeffs_for_sampling(t)

            img_tensor = np.array([img])  # batchの軸を加える
            img_tensor = torch.Tensor(img_tensor).to(self.device)  # Tensorにして、GPUへ送る
            noise_estimate_tensor = self.model(
                img_tensor, torch.Tensor([[t]]).to(self.device)
            )
            noise_estimate = (
                noise_estimate_tensor[0].cpu().numpy()
            )  # batch軸を除いて、numpy形式にする

            img = coeff_normalize * (img - coeff_noise * noise_estimate)

            if t > 0:
                additional_noise = np.random.randn(*(img.shape))
                img += additional_noise * coeff_additional_noise

            if self.do_clip_noise:
                img = np.clip(img, -1.0, +1.0)

            imgs.append(img)

        return imgs

    def plot(self, imgs, output=None):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])

        fig, axs = plt.subplots(
            figsize=self.output_fig_size, nrows=num_rows, ncols=num_cols, squeeze=False
        )
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                img_np = img.copy().transpose(1, 2, 0)
                img_np = (img_np + 1.0) / 2.0 * 255.0
                img_np = img_np.astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(axis=-1)
                img_PIL = Image.fromarray(img_np)

                # img_PIL = self.convert_Tensor_to_PIL(img)
                ax.imshow(np.asarray(img_PIL))
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.tight_layout()

        if output:
            fig.savefig(output)

        plt.close()
