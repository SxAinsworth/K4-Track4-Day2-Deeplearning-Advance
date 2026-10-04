"""DeepWeeds split validation, transforms, dataset and deterministic loaders."""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
EXPECTED_TOTAL = 17_509
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
               "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Load the author's unchanged train/val/test files."""
    if not isinstance(fold, int) or not 0 <= fold <= 4:
        raise ValueError("fold phải là số nguyên trong khoảng 0..4")
    frames = []
    for split in ("train", "val", "test"):
        path = Path(labels_dir) / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        # Các subset chính chủ chỉ có Filename,Label; Species nằm trong labels.csv.
        missing = {"Filename", "Label"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} thiếu cột: {sorted(missing)}")
        frame = frame.copy()
        frame["Label"] = pd.to_numeric(frame["Label"], errors="raise").astype(int)
        if frame["Filename"].isna().any() or not frame["Label"].between(0, 8).all():
            raise ValueError(f"Dữ liệu không hợp lệ trong {path}")
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Fail on duplicates, leakage or missing images; return report-ready counts."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    for name, frame in frames.items():
        if {"Filename", "Label"} - set(frame.columns):
            raise ValueError(f"{name} thiếu Filename/Label")
        if frame["Filename"].duplicated().any():
            raise ValueError(f"{name} có Filename trùng")
    names = {name: set(frame["Filename"].astype(str)) for name, frame in frames.items()}
    overlap = {"train_val": len(names["train"] & names["val"]),
               "train_test": len(names["train"] & names["test"]),
               "val_test": len(names["val"] & names["test"])}
    if any(overlap.values()):
        raise ValueError(f"Rò rỉ dữ liệu: {overlap}")
    union = set().union(*names.values())
    if len(union) != EXPECTED_TOTAL:
        raise ValueError(f"Hợp có {len(union)}, kỳ vọng {EXPECTED_TOTAL}")
    root = Path(images_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    missing = sorted(filename for filename in union if not (root / filename).is_file())
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh; ví dụ {missing[:5]}")
    report = {
        "n": {name: len(frame) for name, frame in frames.items()},
        "ratio": {name: len(frame) / EXPECTED_TOTAL for name, frame in frames.items()},
        "per_class": {name: frame["Label"].value_counts().reindex(range(9), fill_value=0)
                      .sort_index().astype(int).to_dict() for name, frame in frames.items()},
        "overlap": overlap, "union": len(union), "missing_files": 0,
    }
    print(pd.DataFrame(report["per_class"]).rename_axis("Label"))
    print("Số ảnh:", report["n"], "| giao:", overlap, "| hợp:", len(union))
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if img_size <= 0:
        raise ValueError("img_size phải dương")
    tail = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.Resize(max(256, img_size)), transforms.CenterCrop(img_size), *tail])
    choices = {"basic": [], "color": [transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)],
               "trivial": [transforms.TrivialAugmentWide()],
               "randaug": [transforms.RandAugment(num_ops=2, magnitude=9)]}
    if aug not in choices:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return transforms.Compose([transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                               transforms.RandomHorizontalFlip(), *choices[aug], *tail])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        if {"Filename", "Label"} - set(df.columns):
            raise ValueError("DataFrame thiếu Filename/Label")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir, self.transform = Path(images_dir), transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = self.images_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image) if self.transform else image.copy()
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    ds, actual_sampler = DeepWeedsDataset(df, images_dir, transform), None
    if sampler not in (None, "balanced"):
        raise ValueError("sampler chỉ nhận None hoặc balanced")
    if sampler == "balanced":
        if not train:
            raise ValueError("Balanced sampler chỉ dùng cho train")
        counts = df["Label"].value_counts()
        weights = df["Label"].map(lambda label: 1.0 / counts.loc[label]).to_numpy()
        actual_sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and actual_sampler is None,
                      sampler=actual_sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(ds) >= batch_size,
                      worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)
