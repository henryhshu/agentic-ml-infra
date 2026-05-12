"""Distributed CIFAR-10 training with Ray Train (TorchTrainer)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ray
import ray.train
import ray.train.torch
import torch
import torch.nn as nn
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.cifar10 import CIFAR10Parquet, download_split  # noqa: E402
from models.resnet import build_resnet18  # noqa: E402

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def train_loop_per_worker(config: dict) -> None:
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    cache_dir = config.get("cache_dir")

    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_path = download_split("train", cache_dir)
    test_path = download_split("test", cache_dir)

    train_loader = DataLoader(
        CIFAR10Parquet(train_path, transform=train_tfm),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        CIFAR10Parquet(test_path, transform=eval_tfm),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    test_loader = ray.train.torch.prepare_data_loader(test_loader)

    model = ray.train.torch.prepare_model(build_resnet18(num_classes=10))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            labels = torch.as_tensor(labels)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                labels = torch.as_tensor(labels)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        ray.train.report({"epoch": epoch + 1, "test_acc": correct / total})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--address", type=str, default=None, help="Ray cluster address (e.g. ray://head:10001)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.address:
        ray.init(address=args.address)
    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu),
    )
    result = trainer.fit()
    print(result.metrics)


if __name__ == "__main__":
    main()
