# combined.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocess import preprocess_image  # Import from preprocess.py
from model import OCRModel  # Import from separate.py
from dataload import bangla_load, english_load

# Load Bangla and English data loaders
bangla_train_loader, bangla_val_loader, bangla_test_loader, bangla_num_classes = bangla_load
english_train_loader, english_val_loader, english_test_loader, english_num_classes = english_load

# Load pre-trained models with weights_only=True
bangla_model = OCRModel(bangla_num_classes)
english_model = OCRModel(english_num_classes)
bangla_model.load_state_dict(torch.load("saved_models/bangla_model.pth", weights_only=True)['model_state_dict'])
english_model.load_state_dict(torch.load("saved_models/english_model.pth", weights_only=True)['model_state_dict'])

# Freeze layers in both models up to the LSTM layers
for param in list(bangla_model.parameters())[:-4]:
    param.requires_grad = False
for param in list(english_model.parameters())[:-4]:
    param.requires_grad = False

# Combined model
class CombinedModel(nn.Module):
    def __init__(self, bangla_model, english_model, combined_classes):
        super(CombinedModel, self).__init__()
        self.bangla_feature_extractor = nn.Sequential(*list(bangla_model.children())[:-2])  # Up to LSTM
        self.bangla_lstm = list(bangla_model.children())[-2]  # LSTM layer
        self.english_feature_extractor = nn.Sequential(*list(english_model.children())[:-2])  # Up to LSTM
        self.english_lstm = list(english_model.children())[-2]  # LSTM layer
        
        # Joint layers
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, combined_classes)

    def forward(self, bangla_x, english_x):
        # Bangla feature extraction
        bangla_features = self.bangla_feature_extractor(bangla_x)  # 4D output
        bangla_features = bangla_features.permute(0, 2, 3, 1).reshape(bangla_features.size(0), -1, 1024)
        bangla_features, _ = self.bangla_lstm(bangla_features)  # LSTM processing
        
        # English feature extraction
        english_features = self.english_feature_extractor(english_x)  # 4D output
        english_features = english_features.permute(0, 2, 3, 1).reshape(english_features.size(0), -1, 1024)
        english_features, _ = self.english_lstm(english_features)  # LSTM processing

        # Concatenate features
        combined_features = torch.cat((bangla_features[:, -1, :], english_features[:, -1, :]), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

class CombinedDataset(Dataset):
    def __init__(self, bangla_loader, english_loader):
        self.bangla_data = list(bangla_loader.dataset)
        self.english_data = list(english_loader.dataset)

        # Match lengths
        min_length = min(len(self.bangla_data), len(self.english_data))
        self.bangla_data = self.bangla_data[:min_length]
        self.english_data = self.english_data[:min_length]

        # Unique combined labels
        self.combined_labels = set(
            hash((f'b{self.bangla_data[idx][1]}', f'e{self.english_data[idx][1]}'))
            for idx in range(len(self.bangla_data))
        )

    def __len__(self):
        return len(self.bangla_data)

    def __getitem__(self, idx):
        bangla_image, bangla_label = self.bangla_data[idx]
        english_image, english_label = self.english_data[idx]

        # Unique label
        combined_label = hash((f'b{bangla_label}', f'e{english_label}')) % len(self.combined_labels)
        return bangla_image, english_image, combined_label

    @property
    def num_classes(self):
        return len(self.combined_labels)

# Create combined dataset and dataloader
combined_dataset = CombinedDataset(bangla_train_loader, english_train_loader)
combined_dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# Initialize combined model
combined_classes = combined_dataset.num_classes
combined_model = CombinedModel(bangla_model, english_model, combined_classes)

# Define training settings for the combined model
combined_optimizer = optim.Adam(filter(lambda p: p.requires_grad, combined_model.parameters()), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Function to train the combined model
def train_combined_model(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for bangla_images, english_images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(bangla_images, english_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}')

# Train combined model
train_combined_model(combined_model, combined_dataloader, criterion, combined_optimizer)

# Save the combined model
torch.save(combined_model.state_dict(), "bilingual_ocr_model.pth")
