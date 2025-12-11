import os
import shutil
import random
from pathlib import Path

# ====== CHANGE THIS TO YOUR DATASET PATH ======
DATASET_DIR = Path("dataset")

# These are your actual folders
CROP_FOLDERS = {
    "RICE": [
        "Bacterial Leaf Blight",
        "Brown Spot",
        "Healthy Rice Leaf",
        "Leaf Blast",
        "Leaf scald",
        "Sheath Blight"
    ],
    "PULSES": [
        "BPLD",
        "Pea Plant dataset"
    ]
}

# Output folder
OUTPUT_DIR = Path("dataset_split")

# Split ratios (can be adjusted)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Supported image extensions (lowercase)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Reproducible shuffling
random.seed(42)


def _normalize_ratios(train, val, test):
    total = float(train + val + test)
    if total == 0:
        raise ValueError("Train/Val/Test ratios sum to 0")
    return train / total, val / total, test / total


def _gather_images_in_dir(d: Path):
    """Return list of image files directly under directory `d` (no recursion)."""
    return [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def split_and_copy_images(class_path: Path, output_path: Path, train_r, val_r, test_r):
    images = _gather_images_in_dir(class_path)
    if not images:
        print(f"⚠ No image files found in {class_path}; skipping")
        return

    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_r)
    val_end = train_end + int(total * val_r)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    # Create class directories
    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "val").mkdir(parents=True, exist_ok=True)
    (output_path / "test").mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in train_imgs:
        shutil.copy(img, output_path / "train" / img.name)
    for img in val_imgs:
        shutil.copy(img, output_path / "val" / img.name)
    for img in test_imgs:
        shutil.copy(img, output_path / "test" / img.name)

    print(f"✔ {class_path.relative_to(DATASET_DIR)}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")


def main():
    # Normalize ratios so they always sum to 1
    train_r, val_r, test_r = _normalize_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    if not DATASET_DIR.exists():
        print(f"Error: dataset folder '{DATASET_DIR}' not found.")
        return

    # For each crop configured, find either image folders directly under the class
    # or nested class folders (e.g., PULSES/BPLD/<disease>/images)
    for crop, class_list in CROP_FOLDERS.items():
        for class_name in class_list:
            base = DATASET_DIR / crop / class_name
            if not base.exists():
                print(f"⚠ Folder not found: {base}")
                continue

            # If base contains images directly, treat it as a class folder
            direct_images = _gather_images_in_dir(base)
            if direct_images:
                output_path = OUTPUT_DIR / crop / class_name
                print(f"\nProcessing {crop}/{class_name} (direct images) ...")
                split_and_copy_images(base, output_path, train_r, val_r, test_r)
                continue

            # Otherwise, walk immediate subdirectories and process those that contain images
            subdirs = [p for p in base.iterdir() if p.is_dir()]
            if not subdirs:
                print(f"⚠ No subfolders in {base}; nothing to process.")
                continue

            for sub in subdirs:
                imgs = _gather_images_in_dir(sub)
                if not imgs:
                    # skip empty or non-image subfolders
                    continue
                # Mirror the nested structure under OUTPUT_DIR: crop/class_name/sub.name
                output_path = OUTPUT_DIR / crop / class_name / sub.name
                print(f"\nProcessing {crop}/{class_name}/{sub.name} ...")
                split_and_copy_images(sub, output_path, train_r, val_r, test_r)

    print("\n🎉 Dataset Split Completed Successfully!")


if __name__ == "__main__":
    main()
