
import os
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# 1. Device
device = torch.device("cpu")

# 2. Dataset class
class DRDataset(Dataset):

    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data.iloc[index]["id_code"]
        label = int(self.data.iloc[index]["diagnosis"])

        image_path = os.path.join(
            self.image_folder,
            image_name + ".png"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# 3. Image preprocessing: 224 x 224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Training dataset
train_dataset = DRDataset(
    "datasets/aptos/train_1.csv",
    "datasets/aptos/train_images/train_images",
    transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

# 5. Validation dataset
val_dataset = DRDataset(
    "datasets/aptos/valid.csv",
    "datasets/aptos/val_images/val_images",
    transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# 6. Load EfficientNet-B0
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Freeze feature extraction layers
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace classifier for five DR grades
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 5)

model = model.to(device)

# 7. Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)

# 8. Training
epochs = 1

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 20 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"completed"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Training Loss: {average_loss:.4f}"
    )

# 9. Validation
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print(f"Validation Accuracy: {accuracy:.2f}%")

# 10. Save model
os.makedirs("outputs", exist_ok=True)

torch.save(
    model.state_dict(),
    "outputs/efficientnet_b0.pth"
)

print("Model saved successfully!")
