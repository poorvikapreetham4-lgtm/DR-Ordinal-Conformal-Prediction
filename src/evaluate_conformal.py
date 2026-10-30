
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# -----------------------------
# 1. Paths and settings
# -----------------------------

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "aptos"
OUTPUTS = ROOT / "outputs"

MODEL_PATH = OUTPUTS / "efficientnet_b0.pth"

LAC_PATH = OUTPUTS / "lac_threshold.npy"
APS_PATH = OUTPUTS / "aps_threshold.npy"
RAPS_PATH = OUTPUTS / "raps_calibration.npz"
ORDINAL_PATH = OUTPUTS / "ordinal_threshold.npy"

device = torch.device("cpu")

ALPHA = 0.10

grade_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# -----------------------------
# 2. Load test dataset
# -----------------------------

class DRDataset(Dataset):

    def __init__(self):
        self.data = pd.read_csv(DATASET / "test.csv")

        self.image_folder = (
            DATASET / "test_images" / "test_images"
        )

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (
            self.image_folder / (row["id_code"] + ".png")
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = int(row["diagnosis"])

        return image, label


dataset = DRDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False
)

print("Test images:", len(dataset))


# -----------------------------
# 3. Load trained model
# -----------------------------

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


# -----------------------------
# 4. Generate probabilities
# -----------------------------

all_probabilities = []
all_labels = []

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs, dim=1
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_labels.extend(labels.numpy())


all_probabilities = np.array(all_probabilities)
all_labels = np.array(all_labels)

predictions = np.argmax(all_probabilities, axis=1)

accuracy = np.mean(predictions == all_labels)

print(
    "\nModel Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# -----------------------------
# 5. Load calibrated thresholds
# -----------------------------

lac_q = float(np.load(LAC_PATH))

aps_q = float(np.load(APS_PATH))

raps_data = np.load(RAPS_PATH)

# Corrected key: threshold, not q_hat
raps_q = float(raps_data["threshold"])

lambda_reg = float(raps_data["lambda_reg"])

k_reg = int(raps_data["k_reg"])

ordinal_q = float(np.load(ORDINAL_PATH))

print("\nCalibration thresholds loaded.")

print("LAC threshold:", lac_q)
print("APS threshold:", aps_q)
print("RAPS threshold:", raps_q)
print("Ordinal threshold:", ordinal_q)


# -----------------------------
# 6. Prediction set functions
# -----------------------------

def lac_set(probabilities):

    return [
        grade for grade, p in enumerate(probabilities)
        if 1 - p <= lac_q
    ]


def aps_set(probabilities):

    order = np.argsort(-probabilities)

    cumulative = 0.0
    selected = []

    for grade in order:

        cumulative += probabilities[grade]

        if cumulative <= aps_q:
            selected.append(int(grade))
        else:
            break

    if not selected:
        selected = [int(order[0])]

    return selected


def raps_set(probabilities):

    order = np.argsort(-probabilities)

    cumulative = 0.0
    selected = []

    for rank, grade in enumerate(order, start=1):

        cumulative += probabilities[grade]

        score = (
            cumulative
            + lambda_reg * max(rank - k_reg, 0)
        )

        if score <= raps_q:
            selected.append(int(grade))
        else:
            break

    if not selected:
        selected = [int(order[0])]

    return selected


def ordinal_set(probabilities, u):

    modal = int(np.argmax(probabilities))

    left = modal
    right = modal

    accumulated_mass = 0.0

    if (1 - u) * probabilities[modal] <= ordinal_q:

        accumulated_mass = probabilities[modal]

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

            if (
                accumulated_mass
                + (1 - u) * probabilities[next_grade]
                <= ordinal_q
            ):

                accumulated_mass += probabilities[next_grade]

                left = min(left, next_grade)
                right = max(right, next_grade)

            else:
                break

    return list(range(left, right + 1))


# -----------------------------
# 7. Evaluate all methods
# -----------------------------

methods = {
    "LAC": [],
    "APS": [],
    "RAPS": [],
    "Ordinal": []
}

rng = np.random.default_rng(42)

for probabilities in all_probabilities:

    methods["LAC"].append(lac_set(probabilities))

    methods["APS"].append(aps_set(probabilities))

    methods["RAPS"].append(raps_set(probabilities))

    u = rng.uniform(0, 1)

    methods["Ordinal"].append(
        ordinal_set(probabilities, u)
    )


# -----------------------------
# 8. Calculate overall metrics
# -----------------------------

print("\n========== OVERALL RESULTS ==========")

results = []

for method, prediction_sets in methods.items():

    covered = [
        all_labels[i] in prediction_sets[i]
        for i in range(len(all_labels))
    ]

    coverage = np.mean(covered) * 100

    average_size = np.mean([
        len(prediction_set)
        for prediction_set in prediction_sets
    ])

    continuity = np.mean([
        max(s) - min(s) + 1 == len(s)
        for s in prediction_sets
    ]) * 100

    results.append({
        "Method": method,
        "Coverage (%)": round(coverage, 2),
        "Average Set Size": round(average_size, 2),
        "Continuity (%)": round(continuity, 2)
    })

    print(f"\n{method}")
    print("Coverage:", round(coverage, 2), "%")
    print("Average set size:", round(average_size, 2))
    print("Continuity:", round(continuity, 2), "%")


# -----------------------------
# 9. Class-wise coverage
# -----------------------------

print("\n========== CLASS-WISE COVERAGE ==========")

class_results = {}

for method, prediction_sets in methods.items():

    class_coverage = []

    for grade in range(5):

        indices = np.where(all_labels == grade)[0]

        if len(indices) == 0:
            coverage = np.nan

        else:
            coverage = np.mean([
                grade in prediction_sets[i]
                for i in indices
            ]) * 100

        class_coverage.append(round(coverage, 2))

    class_results[method] = class_coverage

    print(f"\n{method}")

    for grade in range(5):

        print(
            f"{grade_names[grade]}: "
            f"{class_coverage[grade]}%"
        )


# -----------------------------
# 10. Save results
# -----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUTS / "conformal_comparison.csv",
    index=False
)

class_df = pd.DataFrame(
    class_results,
    index=grade_names
)

class_df.index.name = "DR Grade"

class_df.to_csv(
    OUTPUTS / "class_wise_coverage.csv"
)


print("\nEvaluation completed successfully!")

print("Saved:")
print("outputs/conformal_comparison.csv")
print("outputs/class_wise_coverage.csv")
