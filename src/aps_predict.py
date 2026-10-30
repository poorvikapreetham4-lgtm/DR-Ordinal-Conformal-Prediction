
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torchvision import transforms, models
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "efficientnet_b0.pth"
THRESHOLD_PATH = ROOT / "outputs" / "aps_threshold.npy"

device = torch.device("cpu")

# Load APS threshold
q_hat = float(np.load(THRESHOLD_PATH))

# Load trained model
model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 5)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load one test image
test_folder = ROOT / "datasets" / "aptos" / "test_images" / "test_images"
image_path = next(test_folder.glob("*.png"))

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Predict probabilities
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0].numpy()

# APS prediction set
sorted_indices = np.argsort(-probabilities)
prediction_set = []
cumulative_probability = 0.0

for grade in sorted_indices:
    prediction_set.append(int(grade))
    cumulative_probability += probabilities[grade]

    if cumulative_probability >= q_hat:
        break

grade_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

print("\nImage:", image_path.name)
print("\nPredicted probabilities:")

for grade, probability in enumerate(probabilities):
    print(f"{grade_names[grade]}: {probability:.4f}")

print("\nAPS Prediction Set:")

for grade in prediction_set:
    print(f"Grade {grade}: {grade_names[grade]}")

print("\nNumber of possible grades:", len(prediction_set))
