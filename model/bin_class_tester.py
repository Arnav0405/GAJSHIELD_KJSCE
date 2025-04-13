import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from xgboost import XGBClassifier
import joblib

# Step 1: Define the CNN Model Architecture
class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: 1 channel (grayscale)
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        self.fc = nn.Linear(256, 256)  # Output 256-dim feature vector

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Step 2: Preprocess the Input Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize all images to 128x128
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    img = Image.open(image_path)  # Ensure grayscale
    img_tensor = transform(img).unsqueeze(0)   # Add batch dimension
    print(f"Image shape after preprocessing: {img_tensor.shape}")  # Debugging line
    return img_tensor

# Step 3: Load Models and Predict Probabilities
def load_models(cnn_model_path, xgb_model_path, device="cuda"):
    # Load CNN model
    cnn_model = FeatureExtractorCNN().to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()

    # Load XGBoost model
    xgb_model = joblib.load(xgb_model_path)

    return cnn_model, xgb_model

def predict_probability(image_path, cnn_model_path="bin_cnn_model_real.pth", xgb_model_path="bin_xgb_model_real.json"):
    # Load the CNN model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = FeatureExtractorCNN().to(device)
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval()

    # Load the XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_model_path)

    # Preprocess the input image
    img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # Extract features using the CNN model

    with torch.no_grad():
        features = cnn_model(img_tensor).cpu().numpy()

    # Predict probabilities using the XGBoost model
    probabilities = xgb_model.predict_proba(features)[0]
    return probabilities

# Main Script
if __name__ == "__main__":
    # Path to the test image
    test_image_path = "./test_bytecode_image.png"

    # Predict probabilities
    probabilities = predict_probability(test_image_path)

    # Output the results
    print(f"Probability of being benign: {probabilities[0]:.4f}")
    print(f"Probability of being malware: {probabilities[1]:.4f}")
    if probabilities[0] > probabilities[1]:
        print("The image is classified as benign.")
    else:
        print("The image is classified as malware.")