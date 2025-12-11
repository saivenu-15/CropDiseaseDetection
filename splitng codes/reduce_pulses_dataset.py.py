import os
import random
from pathlib import Path

# Path to your combined PULSES dataset
DATASET_DIR = Path("dataset_split/PULSES")

# Number of images to keep
KEEP_TRAIN = 30
KEEP_VAL = 6
KEEP_TEST = 6

# Valid image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def reduce_folder(folder: Path, keep_count: int):
    """Keep only 'keep_count' images inside a folder."""
    if not folder.exists():
        return

    images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    if len(images) <= keep_count:
        print(f"✔ {folder} already has {len(images)} images (≤ {keep_count}), skipping...")
        return

    random.shuffle(images)

    keep = images[:keep_count]
    delete = images[keep_count:]

    for img in delete:
        img.unlink()  # delete

    print(f"🔻 {folder}: Kept {len(keep)}, Deleted {len(delete)}")

def process_class(class_folder: Path):
    """Process train/val/test inside each class folder."""
    print(f"\n📁 Processing class: {class_folder.name}")

    reduce_folder(class_folder / "train", KEEP_TRAIN)
    reduce_folder(class_folder / "val", KEEP_VAL)
    reduce_folder(class_folder / "test", KEEP_TEST)

def main():
    if not DATASET_DIR.exists():
        print("❌ PULSES dataset path not found!")
        return

    for class_folder in DATASET_DIR.iterdir():
        if class_folder.is_dir():
            # Each class must contain train/val/test
            process_class(class_folder)

    print("\n🎉 Dataset reduction completed successfully!")

if __name__ == "__main__":
    main()
