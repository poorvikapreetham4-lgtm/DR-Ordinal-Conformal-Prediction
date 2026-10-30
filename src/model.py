
import torch
import torch.nn as nn
from torchvision import models

# Use CPU for this setup
device = torch.device("cpu")

# Load pretrained EfficientNet-B0
weights = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Get the number of input features
input_features = model.classifier[1].in_features

# Replace the final layer with 5 output classes
model.classifier[1] = nn.Linear(input_features, 5)

# Move model to device
model = model.to(device)

# Test the model with one batch
dummy_input = torch.randn(1, 3, 384, 384).to(device)

with torch.no_grad():
    output = model(dummy_input)

print("Model: EfficientNet-B0")
print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Number of classes:", output.shape[1])
print("Device:", device)
