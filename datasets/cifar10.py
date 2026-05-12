"""CIFAR-10 loader backed by Hugging Face parquet shards."""
from __future__ import annotations

import io
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

HF_REPO = "uoft-cs/cifar10"
HF_FILES = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
}


def download_split(split: str, cache_dir: str | Path | None = None) -> Path:
    if split not in HF_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {list(HF_FILES)}")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILES[split],
        repo_type="dataset",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(path)


class CIFAR10Parquet(Dataset):
    """Reads a CIFAR-10 parquet shard with `img` (struct with bytes) and `label` columns."""

    def __init__(self, parquet_path: str | Path, transform=None):
        table = pq.read_table(str(parquet_path), columns=["img", "label"])
        self._image_bytes = [row["bytes"] for row in table.column("img").to_pylist()]
        self._labels = table.column("label").to_pylist()
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        img = Image.open(io.BytesIO(self._image_bytes[idx])).convert("RGB")
        return self.transform(img), self._labels[idx]
