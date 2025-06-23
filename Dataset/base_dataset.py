# Dataset/base_dataset.py

import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DeepfakeDataset(Dataset):
    """Custom Dataset for loading deepfake images from a list of file paths."""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            if not os.path.exists(img_path):
                 # print(f"Warning: File not found at {img_path}. Skipping.")
                 return None # Return None to indicate sample failure

            # Ensure 'RGB' to handle grayscale images potentially in the dataset
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return (image, label)
        except Exception as e:
            # print(f"Error loading or transforming image {img_path}: {e}. Skipping.")
            return None # Return None to indicate sample failure

# Custom collate_fn to handle None values from the dataset
def collate_fn_robust(batch):
    """Filters out None samples from the batch and stacks valid ones."""
    batch = [item for item in batch if item is not None]

    if not batch:
        # print("collate_fn: Batch is empty after filtering.")
        return None, None # Return None to signal an empty batch

    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    try:
       # Stack images (assuming they are all tensors of the same shape)
       images_batch = torch.stack(images)
    except Exception as e:
       print(f"\nError stacking images in collate_fn: {e}. Skipping batch.", file=sys.stderr)
       return None, None # Return None if stacking fails

    try:
        # Stack labels
        labels_batch = torch.tensor(labels)
    except Exception as e:
       print(f"\nError stacking labels in collate_fn: {e}. Skipping batch.", file=sys.stderr)
       return None, None # Return None if stacking fails

    return images_batch, labels_batch

def get_base_transforms(image_size, imagenet_mean, imagenet_std):
    """Returns standard train and validation/test transformations."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        # Add other augmentations as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_transform, val_test_transform
