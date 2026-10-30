
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = DRDataset(
    "datasets/aptos/train_1.csv",
    "datasets/aptos/train_images/train_images",
    transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

# Load one batch
images, labels = next(iter(train_loader))

print("Total training images:", len(train_dataset))
print("Batch image shape:", images.shape)
print("Batch labels:", labels)
