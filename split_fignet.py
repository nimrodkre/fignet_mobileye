import os
import shutil
import random

# Set the seed for reproducibility
random.seed(42)

# Paths
source_dir = "/homes/nimrodkr/datasets/fgnet"
train_dir = "/homes/nimrodkr/datasets/fgnet/train"
val_dir = "/homes/nimrodkr/datasets/fgnet/val"

# Create train and val directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all image file names
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Randomly select 802 images for training
train_files = random.sample(all_files, 802)

# Move the selected images to the train directory
for file in train_files:
    shutil.move(os.path.join(source_dir, file), os.path.join(train_dir, file))

# Move the remaining images to the val directory
for file in set(all_files) - set(train_files):
    shutil.move(os.path.join(source_dir, file), os.path.join(val_dir, file))

print("Images have been successfully moved.")
