import os
from torch.utils.data import DataLoader, random_split
from preprocess import OCRDataset  # Import your OCRDataset class


def prepare_data_loaders(train_dir, test_dir, validation_split=0.05, batch_size=32):
    # Determine the number of classes
    num_classes = len([name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))])

    # Load full training dataset
    full_dataset = OCRDataset(train_dir)

    # Split into training and validation datasets
    train_size = int((1 - validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Load testing dataset
    test_dataset = OCRDataset(test_dir)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes

# Bangla Dataset
bangla_train_dir = "Dataset/Bangla/Dataset/Train"
bangla_test_dir = "Dataset/Bangla/Dataset/Test"
bangla_load = prepare_data_loaders(
    bangla_train_dir, bangla_test_dir, validation_split=0.05, batch_size=32
)

# English Dataset
english_train_dir = "Dataset/English/data/training_data"
english_test_dir = "Dataset/English/data/testing_data"
english_load = prepare_data_loaders(
    english_train_dir, english_test_dir, validation_split=0.05, batch_size=32
)