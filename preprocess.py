import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from check_files import remove_unreadable_images

# Image preprocessing function
def preprocess_image(image_path, img_height=64, img_width=64):
    """
    Load and preprocess an image for OCR.
    Converts the image to grayscale, resizes it, and normalizes pixel values.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_width, img_height))
    image = image / 255.0  # Normalize pixel values to [0,1]
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

# Dataset class for loading images
class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the image folders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        print(f"\nDirectory: {root_dir}")
        remove_unreadable_images(root_dir)

        # Initialize counters
        total_images = 0
        total_labels = 0

        # Iterate through the folders in the root directory
        for label_idx, label_name in enumerate(os.listdir(root_dir)):
            label_folder = os.path.join(root_dir, label_name)

            # Only process if it is a directory
            if os.path.isdir(label_folder):
                total_labels += 1  # Count the label
                for image_name in os.listdir(label_folder):
                    image_path = os.path.join(label_folder, image_name)
                    if os.path.isfile(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(label_idx)  # Assign label index to images
                        total_images += 1  # Count the image

        # Print totals
        print(f"Total number of images: {total_images}")
        print(f"Total number of labels: {total_labels}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess_image(self.image_paths[idx])
        label = self.labels[idx]
        
        # Apply any transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label
