import numpy as np
import cv2
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from facenet_pytorch import InceptionResnetV1
import csv


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Path to the directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = os.listdir(img_dir)  # List all files in img_dir

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        if self.transform:
            image = self.transform(image)

        # Return both the image and its filename
        return self.img_names[idx], image


class CachedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]


class FaceNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceNetClassifier, self).__init__()
        # Load the pre-trained FaceNet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        # Freeze the parameters of FaceNet to avoid updating them during training
        for param in self.facenet.parameters():
            param.requires_grad = False

        # Replace the final layer of FaceNet with a new classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, images):
        # Pass the images through FaceNet to get embeddings
        embeddings = self.facenet(images)
        # Pass the embeddings through the new classifier
        outputs = self.classifier(embeddings)
        return outputs


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the cropped face
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),  # Normalize the tensor
])

# Load the category mappings
category_df = pd.read_csv('category.csv')
# Mapping from numeric index to category name
category_mapping = category_df['Category'].to_dict()


# Instantiate the dataset
dataset = CustomImageDataset(img_dir='temp4', transform=transform)
cached_dataset = CachedDataset(dataset)
# Create a DataLoader
test_loader = DataLoader(cached_dataset, batch_size=4, shuffle=False)

num_classes = 100
model = FaceNetClassifier(num_classes=num_classes)
checkpoint_path = "facenet_epoch_10.pth"  # this files contains the trained model weights.
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state'])
model.eval()

predictions = []
with torch.no_grad():
    for img_nums, images in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        # Convert numerical labels to category names
        category_labels = [category_mapping[label.item()] for label in preds.cpu()]
        predictions.extend(zip(img_nums, category_labels))


# Sort predictions by image number (if they're not already sorted)
predictions.sort(key=lambda x: x[0])

# Write predictions to CSV
with open('predictions_with_categories.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ImageNumber', 'CategoryLabel'])
    for img_num, category_label in predictions:
        writer.writerow([img_num, category_label])


