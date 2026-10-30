
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "aptos"
MODEL_PATH = ROOT / "outputs" / "efficientnet_b0.pth"
OUTPUT_PATH = ROOT / "outputs"

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

test_dataset = DRDataset(
    DATASET / "test.csv",
    DATASET / "test_images" / "test_images",
    transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 5)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)

model.to(device)
model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Accuracy
accuracy = sum(
    p == y for p, y in zip(all_predictions, all_labels)
) / len(all_labels) * 100

print(f"\nTest Accuracy: {accuracy:.2f}%")

# Classification report
grade_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

print("\nCLASSIFICATION REPORT")
print(classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4],
    target_names=grade_names,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=grade_names,
    yticklabels=grade_names
)

plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.title("EfficientNet-B0 Confusion Matrix")
plt.tight_layout()

OUTPUT_PATH.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH / "confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved successfully!")
