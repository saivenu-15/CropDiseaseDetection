import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

# ====================================================
# CONFIG
# ====================================================
# Use an available model file if present; adjust path if necessary
MODEL_PATH = "rice_model_improved.pth"
if not Path(MODEL_PATH).exists():
    # fallback to existing model file in workspace
    if Path("rice_model.pth.pth").exists():
        MODEL_PATH = "rice_model.pth.pth"

TEST_IMAGE = "aug_0_1254.jpg"        # put any test image here
TEST_DIR = Path("dataset_split/RICE")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", DEVICE)

# ====================================================
# DATASET CLASS (same as training)
# ====================================================
class RiceDataset(Dataset):
    def __init__(self, root_dir, split="test", transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []

        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for class_dir in class_dirs:
            split_path = class_dir / split
            if split_path.exists():
                for img_path in split_path.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                        self.samples.append((img_path, self.class_to_idx[class_dir.name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ====================================================
# TRANSFORMS
# ====================================================
tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ====================================================
# LOAD TEST DATASET
# ====================================================
test_ds = RiceDataset(TEST_DIR, split="test", transform=tf)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

classes = test_ds.classes
print("\nClasses:")
for c in classes:
    print(" -", c)

# ====================================================
# MODEL (same architecture as training)
# ====================================================
class ImprovedRiceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ====================================================
# LOAD MODEL
# ====================================================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = ImprovedRiceCNN(len(classes)).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

print("\nModel Loaded Successfully!\n")

# ====================================================
# 1️⃣ PREDICT SINGLE IMAGE
# ====================================================
print("Predicting for image:", TEST_IMAGE)

img = Image.open(TEST_IMAGE).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(img_tensor)
    _, pred = torch.max(output, 1)

predicted_class = classes[pred.item()]

print("\n===============================")
print("Predicted Class:", predicted_class)
print("===============================\n")

# ====================================================
# 2️⃣ COMPUTE TEST ACCURACY
# ====================================================
print("Calculating Test Accuracy...\n")

correct = 0
total = len(test_ds)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100

print("===========================================")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Correct Predictions: {correct}/{total}")
print("===========================================")
