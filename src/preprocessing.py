
import pandas as pd
from PIL import Image
from torchvision import transforms
import os

# Dataset paths
csv_path = "datasets/aptos/train_1.csv"
image_folder = "datasets/aptos/train_images/train_images"

# Read dataset
df = pd.read_csv(csv_path)

# Select first image
image_name = df.iloc[0]["id_code"]
image_path = os.path.join(image_folder, image_name + ".png")

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open(image_path).convert("RGB")

# Apply preprocessing
processed_image = preprocess(image)

# Display information
print("Original image size:", image.size)
print("Processed image shape:", processed_image.shape)
print("Image data type:", processed_image.dtype)
print("DR Grade:", df.iloc[0]["diagnosis"])
