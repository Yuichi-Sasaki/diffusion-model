import argparse

import torchvision

from DiffusionModel.DiffusionModel import DiffusionModel
from DiffusionModel.UNet import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", type=str, default="working/CelebA")
parser.add_argument("--gpu", type=str, default="-1")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

# UNetを作成
model = UNet(in_channels=3, enable_time_embedding=True)

# DiffusionModelを作成
diff = DiffusionModel(
    model,
    timesteps=1000,
    gpu=args.gpu,
    working_dir=args.working_dir,
)

# データを読み込む
# データを読み込む
dataset = torchvision.datasets.CelebA(
    root="./datasets",
    split="Train",
    transform=diff.get_data_transform(),
    download=True,
)

# 学習を実行
diff.train(
    dataset,
    epochs=2000,
    batch_size=args.batch_size,
    lr=2e-4*diff.n_gpus,
    plot_timesteps=[500, 750, 900, 990, 999],
    save_freq=5,
    generate_freq=1,
    num_workers=self.n_gpus*4,
)
