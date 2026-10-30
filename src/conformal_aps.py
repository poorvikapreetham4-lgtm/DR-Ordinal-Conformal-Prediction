
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

device = torch.device("cpu")

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

# Load trained model
model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 5)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model.to(device)
model.eval()

# Collect calibration probabilities and labels
all_probabilities = []
all_labels = []

with torch.no_grad():
    for images, labels in calibration_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        all_probabilities.extend(probabilities.numpy())
        all_labels.extend(labels.numpy())

probabilities = np.array(all_probabilities)
labels = np.array(all_labels)

# Calculate APS nonconformity scores
aps_scores = []

for i in range(len(labels)):
    probs = probabilities[i]
    true_label = labels[i]

    # Sort grades by probability, highest first
    sorted_indices = np.argsort(-probs)

    # Find the position of the true grade
    true_position = np.where(sorted_indices == true_label)[0][0]

    # Cumulative probability up to the true grade
    score = np.sum(probs[sorted_indices[:true_position + 1]])

    aps_scores.append(score)

aps_scores = np.array(aps_scores)

# Calculate calibration threshold
alpha = 0.10
n = len(aps_scores)

rank = int(np.ceil((n + 1) * (1 - alpha)))

if rank > n:
    q_hat = float("inf")
else:
    q_hat = np.sort(aps_scores)[rank - 1]

print("APS calibration completed!")
print("Calibration images:", n)
print("Target error rate:", alpha)
print("APS threshold:", q_hat)

# Save threshold
OUTPUT_PATH = ROOT / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

np.save(OUTPUT_PATH / "aps_threshold.npy", q_hat)

print("APS threshold saved successfully!")
