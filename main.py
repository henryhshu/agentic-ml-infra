"""Entry point: single-node ResNet18 training on CIFAR-10 from HF parquet."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.cifar10 import CIFAR10Parquet, download_split
from models.resnet import build_resnet18

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms():
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train, eval_


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-dir", type=Path, default=Path("data/cifar10"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tfm, eval_tfm = build_transforms()
    train_path = download_split("train", args.cache_dir)
    test_path = download_split("test", args.cache_dir)

    train_loader = DataLoader(
        CIFAR10Parquet(train_path, transform=train_tfm),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        CIFAR10Parquet(test_path, transform=eval_tfm),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = torch.as_tensor(labels, device=device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"epoch {epoch + 1:02d}/{args.epochs}  loss={train_loss:.4f}  test_acc={correct / total:.4f}")


if __name__ == "__main__":
    main()
