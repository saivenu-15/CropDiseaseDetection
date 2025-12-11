import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path

# ====================================================
# CONFIG
# ====================================================
MODEL_PATH = "pulses_model_improved.pth"
TEST_IMAGE = "37y.jpg"       # your input image
TEST_DIR = Path("dataset_split/PULSES")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", DEVICE)

# ====================================================
# CUSTOM DATASET (same as training)
# ====================================================
class PulsesDataset(Dataset):
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
test_ds = PulsesDataset(TEST_DIR, split="test", transform=tf)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

classes = test_ds.classes
print("\n🌱 Pulses Classes Detected:")
for c in classes:
    print(" -", c)

# ====================================================
# MODEL ARCHITECTURE (same as pulses training)
# ====================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),

            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ====================================================
# LOAD MODEL
# ====================================================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = ImprovedCNN(len(classes)).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

print("\n🚀 Pulses Model Loaded Successfully!\n")

# ====================================================
# 1️⃣ SINGLE IMAGE PREDICTION
# ====================================================
print(f"📸 Predicting for: {TEST_IMAGE}")

img = Image.open(TEST_IMAGE).convert("RGB")
img_tensor = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

predicted_class = classes[predicted.item()]

print("\n===============================")
print("🔮 Predicted Class:", predicted_class)
print("===============================\n")


# ====================================================
# 2️⃣ TESTING ACCURACY
# ====================================================
print("🧪 Calculating Pulses Model Test Accuracy...\n")

correct = 0
total = len(test_ds)

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()

accuracy = (correct / total) * 100

print("===============================================")
print(f"✅ Pulses Model Test Accuracy: {accuracy:.2f}%")
print(f"📌 Correct Predictions: {correct}/{total}")
print("===============================================")
