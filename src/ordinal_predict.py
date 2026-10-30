
import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "efficientnet_b0.pth"
THRESHOLD_PATH = ROOT / "outputs" / "ordinal_threshold.npy"

device = torch.device("cpu")

# Reproducible randomization for this demonstration
rng = np.random.default_rng(42)
u = rng.uniform(0, 1)

# Load calibrated threshold
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
test_folder = (
    ROOT / "datasets" / "aptos" /
    "test_images" / "test_images"
)

image_path = next(test_folder.glob("*.png"))

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)

# Predict probabilities
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)[0].numpy()

# Ordinal prediction using adjacent expansion
modal_grade = int(np.argmax(probabilities))

left = modal_grade
right = modal_grade
accumulated_mass = 0.0

# Check whether the modal grade is included
if (1 - u) * probabilities[modal_grade] <= q_hat:

    accumulated_mass = probabilities[modal_grade]

    # Expand the interval using the likelier neighbour
    while left > 0 or right < 4:

        neighbours = []

        if left > 0:
            neighbours.append(left - 1)

        if right < 4:
            neighbours.append(right + 1)

        next_grade = max(
            neighbours,
            key=lambda grade: probabilities[grade]
        )

        # Check whether the next grade can be included
        if (
            accumulated_mass
            + (1 - u) * probabilities[next_grade]
            <= q_hat
        ):
            accumulated_mass += probabilities[next_grade]

            left = min(left, next_grade)
            right = max(right, next_grade)

        else:
            break

else:
    # Force a non-empty set by retaining the modal grade
    left = modal_grade
    right = modal_grade

grade_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

# Display results
print("\nImage:", image_path.name)

print("\nPredicted probabilities:")
for grade, probability in enumerate(probabilities):
    print(f"{grade_names[grade]}: {probability:.4f}")

print("\nOrdinal Prediction Interval:")

for grade in range(left, right + 1):
    print(f"Grade {grade}: {grade_names[grade]}")

print("\nLower grade:", left)
print("Upper grade:", right)
print("Number of possible grades:", right - left + 1)
print("Randomization value:", u)
