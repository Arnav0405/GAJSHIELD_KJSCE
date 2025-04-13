import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
from xgboost import XGBClassifier

class MalwareClassifier:
    def __init__(self, cnn_model_path="feature_extractor_test.pth", xgb_model_path="feature_extractor_test.json", dataset_path='/home/arnavw/Code/ml/GAJSHIELD_KJSCE/model/data/image_data/malware_dataset/train', device="cuda"):
        """
        Initialize the MalwareClassifier with paths to the CNN and XGBoost models.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.cnn_model_path = cnn_model_path
        self.xgb_model_path = xgb_model_path
        self.dataset_path = dataset_path
        self.cnn_model = None
        self.xgb_model = None
        self.class_names = None
        self.load_models()

    def load_models(self):
        """
        Load the pre-trained CNN and XGBoost models, and extract class names from the dataset.
        """
        # Load the CNN model
        self.cnn_model = ClassifierCNN().to(self.device)
        self.cnn_model.load_state_dict(torch.load(self.cnn_model_path, map_location=self.device))
        self.cnn_model.eval()  # Set to evaluation mode

        # Load the XGBoost model
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(self.xgb_model_path)

        # Extract class names from the dataset
        dataset = datasets.ImageFolder(root=self.dataset_path)
        self.class_names = dataset.classes  # List of class names

    @staticmethod
    def preprocess_image(image_path, target_size=(64, 64)):
        """
        Preprocess the input image: resize, convert to grayscale, normalize, and convert to tensor.
        """
        transform = transforms.Compose([
            transforms.Resize(target_size),  # Resize to fixed size
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])
        image = Image.open(image_path)
        return transform(image).unsqueeze(0)  # Add batch dimension

    def extract_features(self, image_tensor):
        """
        Extract features from the preprocessed image using the CNN model.
        """
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            features = self.cnn_model(image_tensor).cpu().numpy()
        return features

    def predict_class_and_confidence(self, image_path):
        """
        Predict the class and confidence for the input image.
        Returns the predicted class name and confidence as a float.
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image_path)

        # Extract features using the CNN
        features = self.extract_features(image_tensor)

        # Predict probabilities using XGBoost
        probabilities = self.xgb_model.predict_proba(features)[0]  # Shape: (num_classes,)

        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(probabilities)
        predicted_class_name = self.class_names[predicted_class_index]
        confidence = probabilities[predicted_class_index] * 100  # Convert to percentage

        return predicted_class_name, confidence

    def get_predicted_class(self, image_path):
        """
        Get the predicted class name for the input image.
        Returns the predicted class name as a string.
        """
        predicted_class_name, _ = self.predict_class_and_confidence(image_path)
        return predicted_class_name

    def get_confidence(self, image_path):
        """
        Get the confidence for the predicted class of the input image.
        Returns the confidence as a float value.
        """
        _, confidence = self.predict_class_and_confidence(image_path)
        return confidence


# Define the CNN architecture
class ClassifierCNN(nn.Module):
    def __init__(self):
        super(ClassifierCNN, self).__init__()
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (Batch, 64, 32, 32)

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (Batch, 64, 16, 16)

            # Block 3
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (Batch, 32, 8, 8)

            # Flatten
            nn.Flatten(),

            # Fully Connected Layer
            nn.Linear(32 * 8 * 8, 256),  # Adjust based on input size
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Final feature vector size
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)