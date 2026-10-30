
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "aptos"
MODEL_PATH = ROOT / "outputs" / "efficientnet_b0.pth"
OUTPUT_PATH = ROOT / "outputs"

device = torch.device("cpu")

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Settings
alpha = 0.10

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Dataset class
class DRDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = self.image_folder / (row["id_code"] + ".png")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, int(row["diagnosis"])


# Ordinal nonconformity score
def ordinal_score(probabilities, true_label, u):
    """
    Calculate the ordinal nonconformity score
    using the paper's greedy adjacent expansion.
    """

    modal_grade = int(np.argmax(probabilities))

    # Start interval at the modal grade
    left = modal_grade
    right = modal_grade

    # Probability mass accumulated before the true grade
    accumulated_mass = 0.0

    # If the true grade is the modal grade
    if true_label == modal_grade:
        return (1 - u) * probabilities[modal_grade]

    accumulated_mass = probabilities[modal_grade]

    # Expand until the true grade is included
    while not (left <= true_label <= right):

        neighbours = []

        if left > 0:
            neighbours.append(left - 1)

        if right < len(probabilities) - 1:
            neighbours.append(right + 1)

        # Select the neighbour with higher probability
        next_grade = max(
            neighbours,
            key=lambda grade: probabilities[grade]
        )

        # If the selected neighbour is the true grade
        if next_grade == true_label:
            return (
                accumulated_mass
                + (1 - u) * probabilities[true_label]
            )

        # Otherwise accumulate its probability
        accumulated_mass += probabilities[next_grade]

        # Expand the interval
        left = min(left, next_grade)
        right = max(right, next_grade)

    return accumulated_mass


# Load calibration dataset
calibration_dataset = DRDataset(
    DATASET / "valid.csv",
    DATASET / "val_images" / "val_images",
    transform
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Load trained EfficientNet-B0
model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 5)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()

# Collect randomized ordinal calibration scores
ordinal_scores = []

with torch.no_grad():
    for images, labels in calibration_loader:

        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1).numpy()

        labels = labels.numpy()

        for probs, true_label in zip(probabilities, labels):

            # Independent uniform randomization
            u = np.random.uniform(0, 1)

            score = ordinal_score(
                probs,
                int(true_label),
                u
            )

            ordinal_scores.append(score)

ordinal_scores = np.array(ordinal_scores)

# Calculate split-conformal threshold
n = len(ordinal_scores)

rank = int(np.ceil((n + 1) * (1 - alpha)))

if rank > n:
    q_hat = float("inf")
else:
    q_hat = np.sort(ordinal_scores)[rank - 1]

# Display results
print("Ordinal calibration completed!")
print("Calibration images:", n)
print("Target error rate:", alpha)
print("Target coverage:", 1 - alpha)
print("Ordinal threshold:", q_hat)

# Save threshold
OUTPUT_PATH.mkdir(exist_ok=True)

np.save(
    OUTPUT_PATH / "ordinal_threshold.npy",
    q_hat
)

print("Ordinal threshold saved successfully!")
