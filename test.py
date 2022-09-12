import torchvision

from DiffusionModel.DiffusionModel import DiffusionModel
from DiffusionModel.UNet import UNet

# UNetを作成
model = UNet(in_channels=1, enable_time_embedding=True)

# DiffusionModelを作成
diff = DiffusionModel(
    model,
    timesteps=200,
    gpu=2,
    working_dir="/shared/y_sasaki/works/diffusion_model/working/test",
)

# データを読み込む
dataset = torchvision.datasets.FashionMNIST(
    root="./datasets",
    train=True,
    transform=diff.get_data_transform(),
    download=True,
)

# 学習を実行
diff.train(
    dataset,
    epochs=200,
    batch_size=128,
    lr=1e-3,
    plot_timesteps=[100, 150, 180, 190, 199],
)
