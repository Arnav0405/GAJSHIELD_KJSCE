import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from xgboost import XGBClassifier
import joblib

class BinaryClassifier:
    def __init__(self, cnn_model_path="bin_cnn_model_real.pth", xgb_model_path="bin_xgb_model_real.json", device="cuda"):
        """
        Initialize the MalwareClassifier with paths to the CNN and XGBoost models.
        """
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.cnn_model_path = cnn_model_path
        self.xgb_model_path = xgb_model_path
        self.cnn_model = None
        self.xgb_model = None
        self.load_models()

    def load_models(self):
        """
        Load the pre-trained CNN and XGBoost models.
        """
        # Load CNN model
        self.cnn_model = FeatureExtractorCNN().to(self.device)
        self.cnn_model.load_state_dict(torch.load(self.cnn_model_path, map_location=self.device))
        self.cnn_model.eval()

        # Load XGBoost model
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(self.xgb_model_path)

    @staticmethod
    def preprocess_image(image_path):
        """
        Preprocess the input image: resize, convert to grayscale, normalize, and convert to tensor.
        """
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize all images to 128x128
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
        img = Image.open(image_path)  # Ensure grayscale
        img_tensor = transform(img).unsqueeze(0)   # Add batch dimension
        return img_tensor

    def extract_features(self, img_tensor):
        """
        Extract features from the preprocessed image using the CNN model.
        """
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            features = self.cnn_model(img_tensor).cpu().numpy()
        return features

    def predict_probabilities(self, image_path):
        """
        Predict probabilities for the input image being benign or malware.
        Returns a tuple of probabilities (benign_probability, malware_probability).
        """
        # Preprocess the input image
        img_tensor = self.preprocess_image(image_path)

        # Extract features using the CNN model
        features = self.extract_features(img_tensor)

        # Predict probabilities using the XGBoost model
        probabilities = self.xgb_model.predict_proba(features)[0]
        return probabilities[0], probabilities[1]

    def classify_image(self, image_path):
        """
        Classify the input image as benign (True) or malware (False).
        Returns a boolean value: True if benign, False if malware.
        """
        benign_prob, malware_prob = self.predict_probabilities(image_path)
        return benign_prob > malware_prob

    def get_benign_probability(self, image_path):
        """
        Get the probability of the input image being benign.
        Returns a float value representing the benign probability.
        """
        benign_prob, _ = self.predict_probabilities(image_path)
        return benign_prob

    def get_malware_probability(self, image_path):
        """
        Get the probability of the input image being malware.
        Returns a float value representing the malware probability.
        """
        _, malware_prob = self.predict_probabilities(image_path)
        return malware_prob


# Define the CNN Model Architecture
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