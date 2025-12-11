import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from pathlib import Path
from typing import List, Tuple

# ====== CONFIG ======
# Root created by your splitter
DATASET_PATH = Path("dataset_split")
# Set to a specific crop like 'RICE' or 'PULSES'. Set to None to load all crops.
CROP = "RICE"

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ====== IMAGE TRANSFORMS ======
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def _find_split_dirs(crop_path: Path, split_name: str) -> List[Path]:
    """Find folders named 'train'/'val'/'test' anywhere under crop_path."""
    if not crop_path.exists():
        return []
    # use rglob to find nested split folders
    return [p for p in crop_path.rglob(split_name) if p.is_dir()]


def _collect_classes_and_samples(crop_path: Path, split_name: str) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Return (classes, samples) where samples is list of (path, class_idx).

    We assume each `.../<class>/<split>/*` layout produced by your splitter.
    The class name is taken as the parent folder of each split folder.
    """
    split_dirs = _find_split_dirs(crop_path, split_name)
    classes = []
    samples: List[Tuple[str, int]] = []

    # discover class names
    for d in split_dirs:
        cls = d.parent.name
        if cls not in classes:
            classes.append(cls)
    classes.sort()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # collect samples
    for d in split_dirs:
        cls = d.parent.name
        idx = class_to_idx[cls]
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                samples.append((str(p), idx))

    return classes, samples


class SimpleListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, loader=default_loader):
        self.samples = samples
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


def build_loaders(crop: str = "RICE"):
    if crop:
        crop_path = DATASET_PATH / crop
    else:
        crop_path = DATASET_PATH

    train_classes, train_samples = _collect_classes_and_samples(crop_path, "train")
    _, val_samples = _collect_classes_and_samples(crop_path, "val")
    _, test_samples = _collect_classes_and_samples(crop_path, "test")

    if not train_samples:
        print(f"Warning: no training images found under {crop_path}")

    train_dataset = SimpleListDataset(train_samples, transform=train_transforms)
    val_dataset = SimpleListDataset(val_samples, transform=val_transforms)
    test_dataset = SimpleListDataset(test_samples, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_classes, train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Build loaders for the configured crop
    classes, train_loader, val_loader, test_loader = build_loaders(CROP)

    print("Classes:", classes)
    print("Train:", len(train_loader.dataset), "images")
    print("Val:", len(val_loader.dataset), "images")
    print("Test:", len(test_loader.dataset), "images")
