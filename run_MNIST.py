import argparse

import torchvision

from DiffusionModel.DiffusionModel import DiffusionModel
from DiffusionModel.UNet import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", type=str, default="working/MNIST")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

# UNetを作成
model = UNet(in_channels=1, enable_time_embedding=True)

# DiffusionModelを作成
diff = DiffusionModel(
    model,
    timesteps=200,
    gpu=args.gpu,
    working_dir=args.working_dir,
)

# データを読み込む
dataset = torchvision.datasets.MNIST(
    root="./datasets",
    train=True,
    transform=diff.get_data_transform(),
    download=True,
)

# 学習を実行
diff.train(
    dataset,
    epochs=20,
    batch_size=args.batch_size,
    lr=1e-3,
    plot_timesteps=[100, 150, 180, 190, 199],
    save_freq=5,
    generate_freq=1,
    loss_type="l2",
)
