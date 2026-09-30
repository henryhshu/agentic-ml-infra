"""Entry point: single-node ResNet18 training on CIFAR-10 from HF parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

print("[boot] importing torch...", flush=True)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
print("[boot] importing torchvision...", flush=True)
from torchvision import transforms
print("[boot] importing project modules...", flush=True)
from datasets.cifar10 import CIFAR10Parquet, download_split
from models.resnet import build_resnet18
from models.tiny_cnn import build_tiny_cnn
print("[boot] imports done.", flush=True)

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
    p.add_argument("--model", choices=["resnet18", "tiny"], default="resnet18")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps",
                   help="explicit device (default: mps for Mac GPU)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; keep 0 on macOS to avoid fork/MPS hangs")
    p.add_argument("--subset", type=int, default=None,
                   help="if set, use only N samples from train + test (fast smoke test)")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument("--log-every", type=int, default=50, help="batches between progress lines")
    return p.parse_args()


BUILDERS = {
    "resnet18": build_resnet18,
    "tiny": build_tiny_cnn,
}


def main():
    args = parse_args()
    device = torch.device(args.device)
    pin_memory = device.type == "cuda"
    print(f"[boot] using device: {device}", flush=True)

    train_tfm, eval_tfm = build_transforms()
    train_path = download_split("train", args.cache_dir)
    test_path = download_split("test", args.cache_dir)

    train_set = CIFAR10Parquet(train_path, transform=train_tfm)
    test_set = CIFAR10Parquet(test_path, transform=eval_tfm)
    if args.subset is not None:
        train_set = Subset(train_set, range(min(args.subset, len(train_set))))
        test_set = Subset(test_set, range(min(args.subset, len(test_set))))
        print(f"subset enabled: train={len(train_set)} test={len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
    )

    model = BUILDERS[args.model](num_classes=10).to(device)
    print(f"model: {args.model}  params={sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    num_batches = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, device=device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            if step % args.log_every == 0 or step == num_batches:
                print(f"  epoch {epoch + 1:02d}  step {step:>4d}/{num_batches}  loss={loss.item():.4f}")
        scheduler.step()
        train_loss = running / len(train_set)

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
