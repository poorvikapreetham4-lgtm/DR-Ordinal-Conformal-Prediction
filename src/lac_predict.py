
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "efficientnet_b0.pth"
THRESHOLD_PATH = ROOT / "outputs" / "lac_threshold.npy"

device = torch.device("cpu")

# Load threshold
q_hat = np.load(THRESHOLD_PATH)

# Load model
model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 5)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Select one test image
test_folder = ROOT / "datasets" / "aptos" / "test_images" / "test_images"
image_path = next(test_folder.glob("*.png"))

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Predict probabilities
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0].numpy()

# Generate LAC prediction set
prediction_set = [
    grade for grade, probability in enumerate(probabilities)
    if probability >= 1 - q_hat
]

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

print("\nLAC Prediction Set:")

if prediction_set:
    for grade in prediction_set:
        print(f"Grade {grade}: {grade_names[grade]}")
else:
    print("Empty prediction set")

print("\nNumber of possible grades:", len(prediction_set))
