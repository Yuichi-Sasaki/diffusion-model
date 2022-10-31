import argparse

import torchvision
import torch

from DiffusionModel.DiffusionModel import DiffusionModel
from DiffusionModel.UNet import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", type=str, default="working/mnist")
parser.add_argument("--gpu", type=str, default="-1")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--ema_decay", type=float, default=None)
parser.add_argument("--load", type=str, default=None)
args = parser.parse_args()

# UNetを作成
model = UNet(in_channels=1)
if args.load is not None:
    print(model.load_state_dict(torch.load(args.load)))

# DiffusionModelを作成
diff = DiffusionModel(
    model,
    timesteps=1000,
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
    lr=args.lr*diff.n_gpus,
    plot_timesteps=[500, 750, 900, 990, 999],
    save_freq=1,
    generate_freq=1,
    num_workers=diff.n_gpus*4,
    ema_decay=args.ema_decay
)
