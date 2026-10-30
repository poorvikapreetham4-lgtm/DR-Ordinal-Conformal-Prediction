
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os

# Project dataset paths
csv_path = "datasets/aptos/train_1.csv"
image_folder = "datasets/aptos/train_images/train_images"

# Read CSV
df = pd.read_csv(csv_path)

# Select first image
image_name = df.iloc[0]["id_code"]
grade = df.iloc[0]["diagnosis"]

# Load image
image_path = os.path.join(image_folder, image_name + ".png")
image = Image.open(image_path)

# Display image
plt.imshow(image)
plt.title(f"Diabetic Retinopathy Grade: {grade}")
plt.axis("off")
plt.show()

print("Image name:", image_name)
print("DR Grade:", grade)
