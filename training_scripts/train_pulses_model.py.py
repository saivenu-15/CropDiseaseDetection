import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import os

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_DIR = Path("dataset_split/PULSES")
BATCH_SIZE = 8
EPOCHS = 25
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# =====================================================
# STRONG DATA AUGMENTATION FOR SMALL DATASET
# =====================================================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =====================================================
# LOAD DATASETS - Custom loader for nested structure
# =====================================================
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Get all class folders
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Collect samples from split folder
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

train_ds = CustomImageDataset(DATASET_DIR, split='train', transform=train_tf)
val_ds = CustomImageDataset(DATASET_DIR, split='val', transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

classes = train_ds.classes
num_classes = len(classes)

print("\n🌱 PULSES Classes:")
for c in classes:
    print(" -", c)
print(f"\nTotal Classes: {num_classes}")

# =====================================================
# IMPROVED FAST CNN MODEL
# =====================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =====================================================
# MODEL + LOSS + OPTIMIZER
# =====================================================
model = ImprovedCNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# =====================================================
# TRAINING LOOP
# =====================================================
print("\n🚀 Starting Improved Training...\n")

for epoch in range(EPOCHS):
    print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

    model.train()
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = (train_correct / train_total) * 100

    # =====================================================
    # VALIDATION
    # =====================================================
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

    val_acc = (val_correct / val_total) * 100

    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model": model.state_dict(),
            "classes": classes
        }, "pulses_model_improved.pth")
        print("✅ Saved best pulses model!")

print("\n🎉 Training Complete!")
print("Best Validation Accuracy:", best_val_acc)
