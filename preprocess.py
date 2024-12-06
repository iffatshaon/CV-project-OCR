# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader

# # Image preprocessing function
# def preprocess_image(image_path, img_height=32, img_width=128):
#     """
#     Load and preprocess an image for OCR.
#     Converts the image to grayscale, resizes it, and normalizes pixel values.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (img_width, img_height))
#     image = image / 255.0  # Normalize pixel values to [0,1]
#     return torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

# # Dataset class for loading images
# class OCRDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = preprocess_image(self.image_paths[idx])
#         label = self.labels[idx]
#         return image, label


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


# root_dir = 'Dataset/Bangla/Dataset/Train'  # The root directory containing label folders
# dataset = OCRDataset(root_dir)

# # Example DataLoader setup
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Directory to save the images
# save_dir = "saved_images"
# os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# # Save a few images with their labels
# for i in range(5):  # Change the range to save more or fewer images
#     image, label = dataset[i]  # Get the i-th image and label
#     image_path = os.path.join(save_dir, f"sample_{i}_label_{label}.png")
    
#     # Save the image using matplotlib
#     plt.imsave(image_path, image.squeeze(), cmap='gray')  # Remove the channel dimension and save in grayscale
#     print(f"Saved: {image_path}")