# Dataset/Celeb_DF_v2.py

import kagglehub
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .base_dataset import DeepfakeDataset, collate_fn_robust, get_base_transforms # Import from base
# from .utils import extract_frames # Assuming you might add this utility later

def download_celeb_df_v2(download_path='./data/Celeb-DF-v2'):
    """
    Downloads the Celeb-DF v2 dataset from Kaggle using kagglehub.

    Args:
        download_path (str): The local path where the dataset should be downloaded.

    Returns:
        str: Path to the downloaded dataset files or None if failed.
    """
    try:
        print("Starting download of Celeb-DF v2 dataset...")
        # Note: kagglehub automatically handles target path implicitly or via environment variables
        # The dataset_download function returns the *actual* path where the dataset lives locally
        # It's often within ~/.kaggle/kagglehub/datasets/...
        path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
        print("Download completed successfully.")
        print(f"Dataset downloaded to: {path}")
        print("\n--- IMPORTANT ---")
        print("Celeb-DF v2 is provided as videos.")
        print("You MUST extract frames from these videos and organize them into 'real' and 'fake' subdirectories.")
        print(f"Create a directory (e.g., '{download_path}/frames/')")
        print(f"Place extracted real frames into '{download_path}/frames/real/'")
        print(f"Place extracted fake frames into '{download_path}/frames/fake/'")
        print("Then, provide '{download_path}/frames/' as the dataset path to main.py")
        print("-----------------\n")

        # Return the base path where kagglehub placed the dataset (contains the video files)
        return path

    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}", file=sys.stderr)
        print("Please ensure you have Kaggle API credentials set up correctly.", file=sys.stderr)
        return None

def load_and_split_data(dataset_root_path, split_ratios, batch_size, image_size, seed,
                        real_dir='real', fake_dirs=['fake'],
                        imagenet_mean=[0.485, 0.456, 0.406], imagenet_std=[0.229, 0.224, 0.225]):
    """
    Collects file paths, splits data, and creates DataLoaders for Celeb-DF v2
    (Assumes frames are already extracted into real/fake subfolders).

    Args:
        dataset_root_path (str): Path to the directory containing 'real' and 'fake' subfolders with frames.
        split_ratios (list): List of [train_ratio, val_ratio, test_ratio].
        batch_size (int): Batch size for DataLoaders.
        image_size (int): Size to resize images to (e.g., 299).
        seed (int): Random seed for reproducibility.
        real_dir (str): Name of the real data subfolder.
        fake_dirs (list): List of fake data subfolder names.
        imagenet_mean (list): Image normalization mean.
        imagenet_std (list): Image normalization std.


    Returns:
        tuple: (train_loader, val_loader, test_loader, files_train, files_val, files_test, labels_train, labels_val, labels_test)
               Returns None loaders/empty lists if data collection fails or splits are empty.
    """
    print(f"\n--- Loading Data from {dataset_root_path} ---")

    if not os.path.isdir(dataset_root_path):
         print(f"Error: Dataset root directory not found at {dataset_root_path}", file=sys.stderr)
         return None, None, None, [], [], [], [], [], []


    # Collect files based on expected subfolders
    real_files = []
    real_path = os.path.join(dataset_root_path, real_dir)
    if os.path.isdir(real_path):
        real_files.extend(glob.glob(os.path.join(real_path, '*.jpg')))
        real_files.extend(glob.glob(os.path.join(real_path, '*.png'))) # Add other extensions if needed
    else:
        print(f"Warning: Real data directory not found at {real_path}. Skipping.", file=sys.stderr)


    fake_files = []
    for fake_dir_name in fake_dirs:
        fake_path = os.path.join(dataset_root_path, fake_dir_name)
        if os.path.isdir(fake_path):
            fake_files.extend(glob.glob(os.path.join(fake_path, '*.jpg')))
            fake_files.extend(glob.glob(os.path.join(fake_path, '*.png'))) # Add other extensions
        else:
            print(f"Warning: Fake data directory not found at {fake_path}. Skipping.", file=sys.stderr)

    real_labels = [0] * len(real_files)
    fake_labels = [1] * len(fake_files)

    all_files = real_files + fake_files
    all_labels = real_labels + fake_labels

    print(f"\n--- Data Collection Summary (from {dataset_root_path}) ---")
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images.")
    print(f"Total images collected: {len(all_files)}")
    if not all_files:
        print("Warning: No images found in specified directories.")
        return None, None, None, [], [], [], [], [], []
    print("-------------------------------------------------")


    # Split data
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    min_samples_per_class = counts.min() if len(unique_labels) > 1 else 0
    if min_samples_per_class < 2:
         print(f"Warning: Minimum class samples ({min_samples_per_class}) is too low for reliable stratified split. Using non-stratified split.")
         stratify_all = None
    else:
         stratify_all = all_labels

    test_ratio = split_ratios[2]
    if test_ratio <= 0 or test_ratio >= 1 or len(all_files) < 2:
         print(f"Warning: Invalid test split ratio {test_ratio} or insufficient total samples ({len(all_files)}). Setting test_size to 0.")
         test_size = 0
         files_train_val, files_test, labels_train_val, labels_test = all_files, [], all_labels, []
    else:
         files_train_val, files_test, labels_train_val, labels_test = train_test_split(
             all_files, all_labels, test_size=test_ratio, random_state=seed, stratify=stratify_all
         )

    train_val_sum = split_ratios[0] + split_ratios[1]
    if len(files_train_val) == 0 or train_val_sum <= 0:
         print("Warning: No samples for train/validation split or invalid train/val ratios. Setting train_size and val_size to 0.")
         files_train, files_val, labels_train, labels_val = [], [], [], []
    else:
        relative_val_size = split_ratios[1] / train_val_sum
        if stratify_all is not None and len(labels_train_val) > 0:
             unique_labels_tv, counts_tv = np.unique(labels_train_val, return_counts=True)
             min_samples_per_class_tv = counts_tv.min() if len(unique_labels_tv) > 1 else 0
             if min_samples_per_class_tv < 2:
                 print(f"Warning: Minimum class samples ({min_samples_per_class_tv}) in train/val set is too low for reliable stratified split. Using non-stratified split.")
                 stratify_tv = None
             else:
                 stratify_tv = labels_train_val
        else:
             stratify_tv = None

        if relative_val_size <= 0 or relative_val_size >= 1 or len(files_train_val) < 2:
            print(f"Warning: Invalid relative validation size {relative_val_size} or insufficient samples ({len(files_train_val)}) for train/val split. Putting all samples into train.")
            files_train, files_val, labels_train, labels_val = files_train_val, [], labels_train_val, []
        else:
            files_train, files_val, labels_train, labels_val = train_test_split(
                files_train_val, labels_train_val, test_size=relative_val_size,
                random_state=seed, stratify=stratify_tv
            )

    print(f"\n--- Data Split Summary ---")
    print(f"Train samples: {len(files_train)}, Val samples: {len(files_val)}, Test samples: {len(files_test)}")
    print("--------------------------")


    # Define transformations using the base helper
    train_transform, val_test_transform = get_base_transforms(image_size, imagenet_mean, imagenet_std)


    # Create datasets
    train_dataset = DeepfakeDataset(files_train, labels_train, transform=train_transform)
    val_dataset = DeepfakeDataset(files_val, labels_val, transform=val_test_transform)
    test_dataset = DeepfakeDataset(files_test, labels_test, transform=val_test_transform)

    # Create dataloaders (only if dataset is not empty)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_robust) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_robust) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_robust) if len(test_dataset) > 0 else None

    # Return loaders AND file paths/labels lists for potential individual image access (like Grad-CAM, t-SNE)
    return (train_loader, val_loader, test_loader,
            files_train, files_val, files_test,
            labels_train, labels_val, labels_test)

# Optional: Add a main block here if you want to test the downloader or loader standalone
def main():
    print("Running Celeb_DF_v2 dataset module standalone.")
    # Example: Download (might require Kaggle credentials)
    # download_celeb_df_v2()

    # Example: Load data (requires extracted frames at the specified path)
    # Replace with a path where you have extracted frames into real/fake subfolders
    # example_dataset_path = './data/Celeb-DF-v2-Frames'
    # if os.path.isdir(example_dataset_path):
    #     print(f"\nAttempting to load data from: {example_dataset_path}")
    #     loaders = load_and_split_data(example_dataset_path, [0.7, 0.15, 0.15], 32, 299, 42)
    #     if loaders[0] is not None:
    #          print("\nSuccessfully created DataLoaders.")
    #          print(f"Train Loader batches: {len(loaders[0])}")
    #          if loaders[1]: print(f"Val Loader batches: {len(loaders[1])}")
    #          if loaders[2]: print(f"Test Loader batches: {len(loaders[2])}")
    #     else:
    #          print("\nFailed to create DataLoaders.")
    # else:
    #      print(f"\nExample dataset path not found: {example_dataset_path}. Cannot test load_and_split_data.")


if __name__ == "__main__":
    main()
