import torchvision

from DiffusionModel.DiffusionModel import DiffusionModel
from DiffusionModel.UNet import UNet

# UNetを作成
model = UNet(in_channels=3, enable_time_embedding=True)

# DiffusionModelを作成
diff = DiffusionModel(
    model,
    timesteps=200,
    gpu=4,
    working_dir="/shared/y_sasaki/works/diffusion_model/working/sample_CelebA",
)

# データを読み込む
dataset = torchvision.datasets.CelebA(
    root="./datasets",
    train=True,
    transform=diff.get_data_transform(),
    download=True,
)

# 学習を実行
diff.train(
    dataset,
    epochs=2000,
    batch_size=128,
    lr=2e-4,
    plot_timesteps=[100, 150, 180, 190, 199],
    save_freq=100,
    generate_freq=20,
)
