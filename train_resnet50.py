import os
import re
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.nn import MSELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

# Custom dataset class
class AgeDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = os.listdir(directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_name).convert('RGB')  # Convert all images to RGB
        # Extract age from filename
        match = re.search(r'A(\d+)', self.images[idx])
        if match:
            label = int(match.group(1))
        else:
            raise ValueError(f"Label not found in filename: {self.images[idx]}")

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

# Load dataset
dataset = AgeDataset('/cs/labs/yweiss/nimrod.kremer/datasets/fignet/train', transform=transform)

# Split data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load pre-trained ResNet50 and modify for regression
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Change the last layer for regression

# Setup optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = MSELoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 250
loss_per_epoch = []
best_loss = float('inf')
best_model_wts = None

for epoch in tqdm(range(num_epochs)):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Modify this part
        if outputs.size(0) > 1:
            outputs = outputs.squeeze()
        else:
            outputs = outputs.view(-1)

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # And also modify this part
            if outputs.size(0) > 1:
                outputs = outputs.squeeze()
            else:
                outputs = outputs.view(-1)

            total_loss += criterion(outputs, labels.float()).item()

    avg_loss = total_loss / len(val_loader)
    loss_per_epoch.append(avg_loss)
    print(f"Epoch {epoch+1}, Avg. Loss: {avg_loss}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_wts = model.state_dict().copy()

# Save the best model
torch.save(best_model_wts, 'resnet_best_model.pth')

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), loss_per_epoch, marker='o', color='b')
plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.savefig('resnet_loss_per_epoch.pdf')
plt.show()

