# model_predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from abc import ABC, abstractmethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
RICE_MODEL_PATH = BASE_DIR / "rice_model_improved.pth"
PULSES_MODEL_PATH = BASE_DIR / "pulses_model_improved.pth"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# =========================
# Abstraction (SOLID + OOPS)
# =========================
class BaseDiseaseModel(ABC):
    def __init__(self, model, classes):
        self.model = model
        self.classes = classes

    @abstractmethod
    def predict(self, image_tensor):
        pass


# =========================
# Concrete Implementations
# =========================
class RiceDiseaseModel(BaseDiseaseModel):
    def predict(self, image_tensor):
        output = self.model(image_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, idx = torch.max(probs, 0)
        return "Rice", self.classes[idx], float(conf)


class PulsesDiseaseModel(BaseDiseaseModel):
    def predict(self, image_tensor):
        output = self.model(image_tensor)
        probs = F.softmax(output, dim=1)[0]
        conf, idx = torch.max(probs, 0)
        return "Pulses", self.classes[idx], float(conf)


# =========================
# Model Loader (SRP)
# =========================
def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model = checkpoint["model"]
    classes = checkpoint["classes"]
    model.to(DEVICE)
    model.eval()
    return model, classes


rice_model, rice_classes = load_model(RICE_MODEL_PATH)
pulses_model, pulses_classes = load_model(PULSES_MODEL_PATH)

rice_predictor = RiceDiseaseModel(rice_model, rice_classes)
pulses_predictor = PulsesDiseaseModel(pulses_model, pulses_classes)


# =========================
# Unified Prediction API
# =========================
def predict(pil_img):
    img = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        r_crop, r_label, r_conf = rice_predictor.predict(img)
        p_crop, p_label, p_conf = pulses_predictor.predict(img)

    if r_conf >= p_conf:
        return r_crop, r_label, r_conf
    else:
        return p_crop, p_label, p_conf
