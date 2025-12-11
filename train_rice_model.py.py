import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np

DATASET_DIR = Path("dataset_split/RICE")
BATCH_SIZE = 8
EPOCHS = 25
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", DEVICE)

# If dataset directory missing, create a small synthetic demo dataset so script can run
if not DATASET_DIR.exists():
    print(f"Dataset path {DATASET_DIR} not found — creating synthetic demo dataset...")
    classes_demo = ["healthy", "diseased"]
    for cls in classes_demo:
        for split in ("train", "val"):
            d = DATASET_DIR / cls / split
            d.mkdir(parents=True, exist_ok=True)
            n = 16 if split == "train" else 8
            for i in range(n):
                arr = (np.random.rand(128, 128, 3) * 255).astype('uint8')
                Image.fromarray(arr).save(d / f"{cls}_{i}.png")

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Custom dataset for nested structure
class RiceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
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
                for img_path in sorted(split_path.glob('*')):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((img_path, self.class_to_idx[class_dir.name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = RiceDataset(DATASET_DIR, split='train', transform=train_tf)
val_ds = RiceDataset(DATASET_DIR, split='val', transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

classes = train_ds.classes
num_classes = len(classes)

print("\n🌾 Rice Classes:")
for c in classes:
    print(" -", c)

# ----------------------------------------------------
# IMPROVED CNN
# ----------------------------------------------------
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

model = ImprovedRiceCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0
final_training_accuracy = 0     # Save for TXT file

print("\n🚀 Starting Improved RICE Training...\n")

for epoch in range(EPOCHS):
    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    model.train()
    correct = 0
    total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    final_training_accuracy = train_acc  # Save last epoch accuracy

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({"model": model.state_dict(), "classes": classes},
                   "rice_model_improved.pth")
        print("✅ Saved best rice model!")

# ----------------------------------------------------
# SAVE RESULTS TO TEXT FILE
# ----------------------------------------------------
with open("rice_training_results.txt", "w") as f:
    f.write("RICE MODEL TRAINING RESULTS\n")
    f.write("=================================\n")
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Final Training Accuracy: {final_training_accuracy:.2f}%\n")
    f.write("\nModel saved as: rice_model_improved.pth\n")

print("\n🎉 Training Complete!")
print("Best Validation Accuracy:", best_val_acc)
