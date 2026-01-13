# model_predict.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
DEVICE = torch.device("cpu")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

RICE_MODEL_PATH = MODELS_DIR / "rice_model_improved.pth"
PULSES_MODEL_PATH = MODELS_DIR / "pulses_model_improved.pth"

# -------------------------------------------------
# TRANSFORM
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# =================================================
# RICE MODEL (EXACT TRAINING ARCH)
# =================================================
class RiceCNN(nn.Module):
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

        # ⚠ MUST MATCH TRAINING
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# =================================================
# PULSES MODEL (EXACT TRAINING ARCH)
# =================================================
class PulsesCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ⚠ MUST MATCH TRAINING
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
def load_model(model_path, model_class):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = model_class(len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])  # STRICT MATCH
    model.eval()
    return model, checkpoint["classes"]

rice_model, rice_classes = load_model(RICE_MODEL_PATH, RiceCNN)
pulses_model, pulses_classes = load_model(PULSES_MODEL_PATH, PulsesCNN)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
def predict(pil_img):
    img = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        rice_out = rice_model(img)
        pulses_out = pulses_model(img)

        rice_prob = F.softmax(rice_out, dim=1)[0]
        pulses_prob = F.softmax(pulses_out, dim=1)[0]

        r_conf, r_idx = torch.max(rice_prob, 0)
        p_conf, p_idx = torch.max(pulses_prob, 0)

    if r_conf >= p_conf:
        return ("Rice", rice_classes[r_idx], float(r_conf))
    else:
        return ("Pulses", pulses_classes[p_idx], float(p_conf))
