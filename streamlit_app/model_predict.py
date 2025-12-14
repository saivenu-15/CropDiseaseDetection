# model_predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models are stored at the workspace root; compute path from this file
BASE_DIR = Path(__file__).resolve().parent.parent
RICE_MODEL_PATH = BASE_DIR / "rice_model_improved.pth"
PULSES_MODEL_PATH = BASE_DIR / "pulses_model_improved.pth"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    state = checkpoint["model"]
    classes = checkpoint["classes"]

    import torch.nn as nn

    # infer classifier in_features from checkpoint state dict
    in_features = None
    for k, v in state.items():
        if k.endswith("classifier.1.weight"):
            in_features = v.shape[1]
            break
    if in_features is None:
        # fallback to common sizes
        in_features = 128

    class Net(nn.Module):
        def __init__(self, num_classes, feat_channels=128):
            super().__init__()
            if feat_channels == 256:
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
            else:
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(feat_channels, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    model = Net(len(classes), feat_channels=in_features)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, classes


if not RICE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Rice model not found at {RICE_MODEL_PATH}")
if not PULSES_MODEL_PATH.exists():
    raise FileNotFoundError(f"Pulses model not found at {PULSES_MODEL_PATH}")


rice_model, rice_classes = load_model(RICE_MODEL_PATH)
pulses_model, pulses_classes = load_model(PULSES_MODEL_PATH)


def predict(pil_img):
    img = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        r_logits = rice_model(img)
        p_logits = pulses_model(img)

        r_probs = F.softmax(r_logits, dim=1)[0]
        p_probs = F.softmax(p_logits, dim=1)[0]

        r_conf, r_idx = torch.max(r_probs, 0)
        p_conf, p_idx = torch.max(p_probs, 0)

    if r_conf >= p_conf:
        return "Rice", rice_classes[r_idx], float(r_conf)
    else:
        return "Pulses", pulses_classes[p_idx], float(p_conf)
