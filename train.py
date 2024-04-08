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


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, category_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            category_file (string): Path to the csv file with category mappings.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = pd.read_csv(csv_file)
        categories = pd.read_csv(category_file)
        # Map category names to index values (which act as numerical labels here)
        self.label_to_num = {row['Category']: index for index, row in categories.iterrows()}
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['File Name'])
        image = Image.open(img_path)
        # Convert label name to numerical value using the mapping
        label_name = self.img_labels.iloc[idx]['Category']
        label_num = self.label_to_num[label_name] if label_name in self.label_to_num else -1  # Use -1 or some error value if not found
        if self.transform:
            image = self.transform(image)
        return image, label_num

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


def initialize_model(num_classes):
    # Load a pre-trained ResNet model
    model = models.resnet50(pretrained=True)
    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False
    # Replace the final layer with a new one that has the desired number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


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

# Specify the path to images and the CSV file
img_dir = 'temp2'
csv_file = 'train_small.csv'
category_file = 'category.csv'

# Create the dataset
dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, category_file=category_file, transform=transform)
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)  # 80% of the data for training
val_size = dataset_size - train_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

cached_train_dataset = CachedDataset(train_dataset)
cached_val_dataset = CachedDataset(val_dataset)

# create DataLoaders for each of these datasets
train_loader = DataLoader(cached_train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(cached_val_dataset, batch_size=64, shuffle=False, pin_memory=True)

# model = initialize_model(num_classes=100)
num_classes = 100
model = FaceNetClassifier(num_classes=num_classes)

# Move the model to the appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.3, verbose=True)

# load the trained model checkpoint after 10 epochs of training.
checkpoint_path = "facenet_epoch_10.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scheduler.load_state_dict(checkpoint['scheduler_state'])

num_epochs = 7  # Number of epochs to train for
max_acc = 0.0   # max validation accuracy variable to check if model weights are worth saving.

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    # Training loop
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    running_corrects = 0
    val_loss = 0.0
    with torch.no_grad():  # No gradients needed
        for images, labels in val_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    print(f'Validation Accuracy: {epoch_acc:.4f}')
    print(f'Epoch {epoch + 1}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')
    # Step the scheduler with the validation loss
    scheduler.step(val_loss)

    # Save the model, optimizer, and scheduler
    if epoch_acc > max_acc:
        print('saving')
        max_acc = epoch_acc
        state = {
            'epoch': 10,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }
        torch.save(state, f"facenet_epoch_10.pth")

