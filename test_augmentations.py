import os
import re
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from PIL import Image
import antialiased_cnns
import numpy as np

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
        image = Image.open(img_name).convert('RGB')

        match = re.search(r'A(\d+)', self.images[idx])
        if match:
            label = int(match.group(1))
        else:
            raise ValueError(f"Label not found in filename: {self.images[idx]}")

        return image, label

# Argument parser
parser = argparse.ArgumentParser(description='Test models on FG-NET validation set.')
parser.add_argument('--zhang', action='store_true', help='Use Zhang parameter for model')
parser.add_argument('--val_dir', type=str, required=True, help='Path to validation dataset directory')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
args = parser.parse_args()

# Transformations
resize_transform = transforms.Resize((256, 256))
crop_transform = transforms.RandomCrop((224, 224))
tensor_transform = transforms.ToTensor()

# Load validation dataset
val_dataset = AgeDataset(args.val_dir, transform=resize_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model setup
if args.zhang:
    model = antialiased_cnns.resnet50(pretrained=False, filter_size=4)
else:
    model = resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 1)

# Load model checkpoint
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate the model
model.eval()
mse = torch.nn.MSELoss()
total_mse = 0.0
num_images = 0

with torch.no_grad():
    for image, label in val_loader:
        crops_mse = []
        for _ in range(4):  # Perform 4 random crops
            cropped_image = crop_transform(image[0])
            input_tensor = tensor_transform(cropped_image).unsqueeze(0)
            output = model(input_tensor)
            crops_mse.append(mse(output, torch.tensor([[label]], dtype=torch.float32)))

        total_mse += np.mean(crops_mse)
        num_images += 1

average_mse = total_mse / num_images
print(f"Average MSE on validation set: {average_mse}")

