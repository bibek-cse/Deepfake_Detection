# -------------------- Inception-ResNetV2 Deepfake Detector (Comprehensive Pipeline) ---------------------------

import os
import glob
import random
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score # Import metrics for AUC/PR
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import cv2
import traceback
import sys

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Configuration ---
class Config:
    # These paths should ideally be passed or configured externally,
    # but for this integration, we'll keep them here as defaults.
    # NOTE: When running from main.py, dataset_path will be passed,
    # but we still need to configure the subdirectories within that path.
    # We will assume the dataset_path passed to test_model *is* the DATA_ROOT.
    DATA_ROOT = None # Will be set by the dataset_path argument in test_model
    REAL_DIR = 'real' # Assuming 'real' subfolder name based on main.py's expectation
    FAKE_DIRS = ['fake'] # Assuming 'fake' subfolder name based on main.py's expectation

    IMAGE_SIZE = 299 # Inception-ResNetV2 typically uses 299x299 input
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 25
    TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15] # 70% train, 15% validation, 15% test
    RANDOM_SEED = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the target layer for Grad-CAM
    # For Inception-ResNetV2 (timm), 'conv2d_7b' is a common name for the last conv layer before pooling.
    # Inspecting timm's InceptionResnetV2 structure suggests 'conv2d_7b' or something within 'block8'.
    # Let's try 'conv2d_7b'. If this doesn't work, printing model structure is needed.
    TARGET_GRADCAM_LAYER = 'conv2d_7b' # Common name for the last conv before pooling in Inception-ResNetV2

    # ImageNet mean and std for normalization (standard for models pre-trained on ImageNet)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Path to save the best model weights
    BEST_MODEL_PATH = 'best_inception_resnet_v2_model.pth' # Changed model name

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Custom Dataset ---
class DeepfakeDataset(Dataset):
    """Custom Dataset for loading deepfake images."""
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
                 return None # Return None to indicate sample failure

            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return (image, label)
        except Exception as e:
            # print(f"Error loading or transforming image {img_path}: {e}. Skipping.") # Keep quiet unless debugging
            return None # Return None to indicate sample failure

# Custom collate_fn to handle None values from the dataset
def collate_fn_robust(batch):
    """Filters out None samples from the batch and stacks valid ones."""
    batch = [item for item in batch if item is not None]

    if not batch:
        return None, None # Return None to signal an empty batch

    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    try:
       images_batch = torch.stack(images)
    except Exception as e:
       print(f"\nError stacking images in collate_fn: {e}. Skipping batch.", file=sys.stderr)
       return None, None # Return None if stacking fails

    try:
        labels_batch = torch.tensor(labels)
    except Exception as e:
       print(f"\nError stacking labels in collate_fn: {e}. Skipping batch.", file=sys.stderr)
       return None, None # Return None if stacking fails

    return images_batch, labels_batch


# --- Data Loading and Splitting ---
def collect_files(data_root, real_dir, fake_dirs):
    """Collects all image file paths and their labels."""
    if not os.path.isdir(data_root):
         print(f"Error: Data root directory not found at {data_root}", file=sys.stderr)
         return [], []

    real_files = []
    real_path = os.path.join(data_root, real_dir)
    if os.path.isdir(real_path):
        real_files.extend(glob.glob(os.path.join(real_path, '*.jpg')))
        real_files.extend(glob.glob(os.path.join(real_path, '*.png'))) # Add other extensions if needed
    else:
        print(f"Warning: Real data directory not found at {real_path}", file=sys.stderr)


    fake_files = []
    for fake_dir in fake_dirs:
        fake_path = os.path.join(data_root, fake_dir)
        if os.path.isdir(fake_path):
            fake_files.extend(glob.glob(os.path.join(fake_path, '*.jpg')))
            fake_files.extend(glob.glob(os.path.join(fake_path, '*.png'))) # Add other extensions
        else:
            print(f"Warning: Fake data directory not found at {fake_path}", file=sys.stderr)


    real_labels = [0] * len(real_files) # 0 for real
    fake_labels = [1] * len(fake_files) # 1 for fake

    all_files = real_files + fake_files
    all_labels = real_labels + fake_labels

    print(f"\n--- Data Collection Summary ---")
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images.")
    print(f"Total images collected: {len(all_files)}")
    if not all_files:
        print("Warning: No images found in specified directories.")
    print("-----------------------------")


    return all_files, all_labels

def create_dataloaders(all_files, all_labels, split_ratios, batch_size, image_size, seed):
    """Splits data and creates PyTorch DataLoaders."""
    if not all_files:
        print("No images found. Cannot create dataloaders.")
        return None, None, None, [], [], [], [], [], [] # Return empty lists for files/labels

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


    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])

    # Create datasets
    train_dataset = DeepfakeDataset(files_train, labels_train, transform=train_transform)
    val_dataset = DeepfakeDataset(files_val, labels_val, transform=val_test_transform)
    test_dataset = DeepfakeDataset(files_test, labels_test, transform=val_test_transform)

    # Create dataloaders (only if dataset is not empty)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_robust) if len(train_dataset) > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_robust) if len(val_dataset) > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_robust) if len(test_dataset) > 0 else None

    return (train_loader, val_loader, test_loader,
            files_train, files_val, files_test,
            labels_train, labels_val, labels_test)

# --- Model Definition ---
def setup_model(num_classes=1):
    """Loads and configures the Inception-ResNetV2 model."""
    try:
        model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)
        print(f"\nLoaded Inception-ResNetV2 model from timm. {'Pre-trained weights loaded.' if timm.is_model_pretrained('inception_resnet_v2') else 'Using untrained model.'}")
    except Exception as e:
        print(f"Error loading model from timm: {e}", file=sys.stderr)
        print("Please check if timm is installed correctly and the model name is valid.", file=sys.stderr)
        raise e

    model = model.to(Config.DEVICE)

    # --- FIX FOR GRAD-CAM INPLACE ERROR ---
    print("\nChecking and replacing inplace ReLU modules for potential compatibility...")
    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and module.inplace:
             try:
                 parts = name.rsplit('.', 1)
                 if len(parts) > 1:
                     parent_name = parts[0]
                     child_name = parts[1]
                     parent_module = model.get_submodule(parent_name)
                     setattr(parent_module, child_name, nn.ReLU(inplace=False))
                     replaced_count += 1
             except (AttributeError, KeyError, RuntimeError) as e:
                  pass # Silently skip if replacement is not possible

    if replaced_count > 0:
        print(f"Replaced {replaced_count} inplace ReLU modules.")
    else:
        print("No inplace ReLU modules found or replaced (might not be necessary or they are not addressable by name).")


    # Verify the target layer for Grad-CAM
    target_layer_found = False
    try:
         test_layer = model.get_submodule(Config.TARGET_GRADCAM_LAYER)
         print(f"Target Grad-CAM layer '{Config.TARGET_GRADCAM_LAYER}' found.")
         if isinstance(test_layer, nn.Conv2d) or (hasattr(test_layer, 'out_channels') and test_layer.out_channels > 0) or (hasattr(test_layer, 'c') and test_layer.c > 0):
              print("Target layer seems suitable for Grad-CAM.")
              target_layer_found = True
         else:
              print(f"Warning: Target Grad-CAM layer '{Config.TARGET_GRADCAM_LAYER}' found, but its type ({type(test_layer).__name__}) might not be ideal for standard Conv-based Grad-CAM.")
              target_layer_found = True # Still set found=True if the module exists
    except AttributeError:
         print(f"Warning: Target Grad-CAM layer '{Config.TARGET_GRADCAM_LAYER}' not found in the model.")
         print("Please inspect the model structure (e.g., print(model)) to find a suitable convolutional layer name.")
         # print(model) # Uncomment to print model structure for debugging layer names
    except Exception as e:
         print(f"An unexpected error occurred while verifying target layer: {e}", file=sys.stderr)

    if not target_layer_found:
        print("Grad-CAM might not work correctly without a valid target layer.")

    print("-------------------------------------------------------")

    return model

# --- Grad-CAM Implementation ---
class GradCAM:
    """Implements Grad-CAM for a PyTorch model."""
    def __init__(self, model, target_layer_name):
        self.model = model.eval() # Ensure model is in evaluation mode
        self.target_layer = None
        self.activation = None
        self.gradient = None
        self.hook_handles = [] # List to store hook handles

        try:
            self.target_layer = self.model.get_submodule(target_layer_name)
        except AttributeError:
            raise ValueError(f"Target layer '{target_layer_name}' not found in the model.")
        except Exception as e:
             raise RuntimeError(f"Error finding target layer '{target_layer_name}': {e}") from e


    def _save_activation(self, module, input, output):
        """Hook to save the output (activation) of the target layer."""
        self.activation = output.clone().detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save the gradient w.r.t. the output of the target layer."""
        self.gradient = grad_output[0].clone().detach()


    def __call__(self, x, target_category=None):
        """
        Compute Grad-CAM for a single input image.
        Args:
            x (torch.Tensor): Input image tensor (shape: [1, C, H, W]). Must be on the correct device.
            target_category (int or None): The target class index for backpropagation.
                                            For binary classification with 1 output neuron,
                                            backpropagate the single logit (None).
        Returns:
            torch.Tensor: Grad-CAM map (shape: [1, H_cam, W_cam]) on the same device as input x.
                          Returns None if processing fails.
        """
        if x.ndim != 4 or x.shape[0] != 1:
            print(f"Grad-CAM: Input tensor must have shape [1, C, H, W], but got {x.shape}. Skipping.", file=sys.stderr)
            return None

        self.remove_hooks()

        try:
            self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
            self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        except Exception as e:
             print(f"Grad-CAM: Error registering hooks on target layer '{Config.TARGET_GRADCAM_LAYER}': {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None


        self.model.zero_grad()
        self.activation = None
        self.gradient = None

        try:
             if next(self.model.parameters()).device != x.device:
                  self.model.to(x.device)
             output = self.model(x)

             if isinstance(output, tuple):
                 print("Warning: Model returned a tuple output during Grad-CAM forward pass. Using only the first element.", file=sys.stderr)
                 output = output[0]

        except Exception as e:
             print(f"Grad-CAM: Error during model forward pass: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None

        try:
            if output.numel() == 1:
                 target_score = output.squeeze()
            elif output.ndim == 2 and output.shape[1] > 1 and target_category is not None:
                 target_score = output[:, target_category].squeeze()
            elif output.ndim == 2 and output.shape[1] > 1 and target_category is None:
                 predicted_class = output.argmax(dim=1).item()
                 target_score = output[:, predicted_class].squeeze()
                 print(f"Warning: Multi-class output detected but target_category is None. Using predicted class {predicted_class} for Grad-CAM.", file=sys.stderr)
            else:
                print(f"Grad-CAM: Unexpected model output shape {output.shape}. Skipping Grad-CAM.", file=sys.stderr)
                self.remove_hooks()
                return None

        except Exception as e:
             print(f"Grad-CAM: Error selecting target score for backprop: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None


        if self.activation is None:
             print(f"Grad-CAM: Activation was not captured. Make sure target layer '{Config.TARGET_GRADCAM_LAYER}' is correctly named and part of the model's forward pass.", file=sys.stderr)
             self.remove_hooks()
             return None

        try:
             target_score.backward(gradient=torch.ones_like(target_score), retain_graph=True)
        except RuntimeError as e:
             print(f"\nGrad-CAM: RuntimeError during backward pass: {e}", file=sys.stderr)
             print("This might be due to inplace operations or issues with graph retention.", file=sys.stderr)
             self.remove_hooks()
             return None
        except Exception as e:
             print(f"Grad-CAM: Error during backward pass: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None


        if self.gradient is None:
             print("Grad-CAM: Gradient was not captured by the hook.", file=sys.stderr)
             self.remove_hooks()
             return None

        try:
             if self.gradient.shape[0] != 1 or self.activation.shape[0] != 1:
                  print(f"Grad-CAM: Expected batch size 1 for gradient ({self.gradient.shape[0]}) or activation ({self.activation.shape[0]}). Skipping.", file=sys.stderr)
                  self.remove_hooks()
                  return None

             pooled_gradients = torch.mean(self.gradient, dim=[2, 3], keepdim=True)

             if pooled_gradients.shape[0] != self.activation.shape[0] or pooled_gradients.shape[1] != self.activation.shape[1]:
                  print(f"Grad-CAM: Shape mismatch between pooled gradients {pooled_gradients.shape} and activation {self.activation.shape}. Cannot compute CAM. Skipping.", file=sys.stderr)
                  self.remove_hooks()
                  return None

             weighted_activation = self.activation * pooled_gradients
             cam = torch.sum(weighted_activation, dim=1, keepdim=True)
             cam = F.relu(cam)

        except Exception as e:
             print(f"Grad-CAM: Error computing CAM map: {e}. Skipping Grad-CAM.", file=sys.stderr)
             self.remove_hooks()
             return None

        self.remove_hooks()
        return cam

    def remove_hooks(self):
        """Removes registered hooks to clean up."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


# --- Visualization Function ---
def visualize_cam(original_img_pil, cam_map, title="Grad-CAM"):
    """
    Visualizes the Grad-CAM map overlaid on the original image.
    Args:
        original_img_pil (PIL.Image): The original image.
        cam_map (np.ndarray): The Grad-CAM map (HxW numpy array).
        title (str): Title for the plot.
    """
    if cam_map is None or cam_map.size == 0:
         print("Warning: CAM map is empty or None. Cannot visualize.", file=sys.stderr)
         return

    try:
        cam_map = np.nan_to_num(cam_map, nan=0.0, posinf=1e5, neginf=-1e5)
        cam_map = cam_map.astype(np.float32)
        cam_map_resized = cv2.resize(cam_map, (original_img_pil.width, original_img_pil.height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Error resizing CAM map: {e}. Skipping visualization.", file=sys.stderr)
        return

    cam_min, cam_max = cam_map_resized.min(), cam_map_resized.max()
    if cam_max - cam_min < 1e-8:
         cam_map_normalized = np.zeros_like(cam_map_resized)
    else:
        cam_map_normalized = (cam_map_resized - cam_min) / (cam_max - cam_min)
        cam_map_normalized = np.clip(cam_map_normalized, 0, 1)

    original_img_np = np.array(original_img_pil)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_normalized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for matplotlib

    alpha = 0.5
    original_img_float = original_img_np.astype(np.float32) / 255.0
    heatmap_float = heatmap.astype(np.float32) / 255.0

    if original_img_float.ndim == 2:
        original_img_float = np.stack([original_img_float]*3, axis=-1)
    elif original_img_float.ndim == 3 and original_img_float.shape[2] == 1:
         original_img_float = np.repeat(original_img_float, 3, axis=2)

    overlaid_img_float = cv2.addWeighted(original_img_float, 1 - alpha, heatmap_float, alpha, 0)
    overlaid_img_np = np.uint8(255 * overlaid_img_float)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_img_np)
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# --- Training Function ---
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False, file=sys.stdout)

    for inputs, labels in train_loop:
        if inputs is None or labels is None:
            train_loop.write("Skipping empty batch in training.")
            continue

        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        try:
             outputs = model(inputs)
        except Exception as e:
             train_loop.write(f"\nError during model forward pass in training: {e}. Skipping batch.")
             continue

        try:
            loss = criterion(outputs, labels)
        except Exception as e:
            train_loop.write(f"\nError calculating loss in training: {e}. Skipping batch.")
            continue

        try:
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
             train_loop.write(f"\nRuntimeError during backward pass or optimizer step: {e}. Skipping batch.")
             train_loop.write("This might be due to graph issues (e.g., inplace operations).")
             continue
        except Exception as e:
             train_loop.write(f"\nError during backward pass or optimizer step: {e}. Skipping batch.")
             continue


        current_batch_size = inputs.size(0)
        running_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

        try:
             preds = (torch.sigmoid(outputs) > 0.5).long()
             correct_preds += (preds.squeeze() == labels.squeeze().long()).sum().item()
        except Exception as e:
             train_loop.write(f"\nError calculating predictions/accuracy in training: {e}.")

        if total_samples > 0:
            train_loop.set_postfix(loss=running_loss / total_samples, acc=correct_preds / total_samples)
        else:
             train_loop.set_postfix(loss="N/A", acc="N/A")


    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_preds / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc

# --- Evaluation Function ---
def evaluate(model, data_loader, criterion, device, phase="Validation"):
    """Evaluates the model on a given data loader."""
    model.eval()
    running_loss = 0.0
    all_preds_logits = []
    all_true_labels = []

    eval_loop = tqdm(data_loader, desc=f"Evaluating [{phase}]", leave=False, file=sys.stdout)

    with torch.no_grad():
        for inputs, labels in eval_loop:
            if inputs is None or labels is None:
                eval_loop.write(f"Skipping empty batch during {phase} evaluation.")
                continue

            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            try:
                outputs = model(inputs)
            except Exception as e:
                 eval_loop.write(f"\nError during model forward pass in evaluation: {e}. Skipping batch.")
                 continue

            try:
                 loss = criterion(outputs, labels)
            except Exception as e:
                 eval_loop.write(f"\nError calculating loss in evaluation: {e}. Skipping batch.")
                 continue

            current_batch_size = inputs.size(0)
            running_loss += loss.item() * current_batch_size

            all_preds_logits.extend(outputs.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

            if len(all_true_labels) > 0:
                eval_loop.set_postfix(loss=running_loss / len(all_true_labels))
            else:
                eval_loop.set_postfix(loss="N/A")


    if not all_true_labels:
        print(f"Warning: No valid samples processed during {phase} evaluation.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([[0,0],[0,0]]), np.array([]), np.array([]), np.array([])

    true_labels_flat = np.array(all_true_labels).flatten()
    preds_logits = np.array(all_preds_logits).flatten()

    if len(true_labels_flat) != len(preds_logits):
         print(f"Error: Mismatch between true labels ({len(true_labels_flat)}) and predictions ({len(preds_logits)}) length during {phase} evaluation. Truncating.", file=sys.stderr)
         min_len = min(len(true_labels_flat), len(preds_logits))
         true_labels_flat = true_labels_flat[:min_len]
         preds_logits = preds_logits[:min_len]
         if min_len == 0:
             print(f"No valid samples after length check for {phase}. Returning defaults.", file=sys.stderr)
             return 0.0, 0.0, 0.0, 0.0, 0.0, np.array([[0,0],[0,0]]), np.array([]), np.array([]), np.array([])


    epoch_loss = running_loss / len(true_labels_flat)

    binary_preds = (preds_logits > 0).astype(int)
    true_labels_int = true_labels_flat.astype(int)

    # Handle cases with only one class
    average_type = 'binary' if len(np.unique(true_labels_int)) == 2 else 'weighted'
    zero_div = 0

    accuracy = accuracy_score(true_labels_int, binary_preds)
    precision = precision_score(true_labels_int, binary_preds, average=average_type, zero_division=zero_div)
    recall = recall_score(true_labels_int, binary_preds, average=average_type, zero_division=zero_div)
    f1 = f1_score(true_labels_int, binary_preds, average=average_type, zero_division=zero_div)

    try:
        conf_matrix = confusion_matrix(true_labels_int, binary_preds)
    except Exception as e:
        print(f"Error computing confusion matrix: {e}. Returning default.", file=sys.stderr)
        conf_matrix = np.array([[0,0],[0,0]])


    return epoch_loss, accuracy, precision, recall, f1, conf_matrix, true_labels_int, binary_preds, preds_logits

# --- Feature Extraction Function (for t-SNE) ---
def extract_features_from_paths(model, file_paths, labels, device, image_size):
    """Extracts features for a list of image file paths."""
    model.eval()

    feature_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
    ])

    all_features = []
    extracted_labels = []
    extracted_paths = []

    if not file_paths:
         print("No file paths provided for feature extraction.")
         return np.array([]), np.array([]), []

    print(f"\n--- Extracting features for t-SNE from {len(file_paths)} images ---")
    feature_loop = tqdm(zip(file_paths, labels), total=len(file_paths), desc="Extracting Features", file=sys.stdout)

    with torch.no_grad():
        for file_path, label in feature_loop:
            try:
                if not os.path.exists(file_path):
                     continue
                img = Image.open(file_path).convert('RGB')
                img_tensor = feature_transform(img).unsqueeze(0).to(device)

                if hasattr(model, 'forward_features') and hasattr(model, 'global_pool'):
                     features = model.forward_features(img_tensor)
                     pooled_features = model.global_pool(features)
                else:
                    raise NotImplementedError("Feature extraction method not compatible with model structure.")

                all_features.append(pooled_features.cpu().numpy().flatten())
                extracted_labels.append(label)
                extracted_paths.append(file_path)

            except NotImplementedError as e:
                 print(f"\n{e}", file=sys.stderr)
                 print("Please check your timm version or model architecture.", file=sys.stderr)
                 return np.array([]), np.array([]), []

            except Exception as e:
                print(f"\nError extracting feature for {file_path}: {e}. Skipping.", file=sys.stderr)
                continue

    print(f"Successfully extracted features for {len(all_features)} images.")
    return np.array(all_features), np.array(extracted_labels), extracted_paths


# --- Main Entry Point (called by main.py) ---
def test_model(dataset_path):
    """
    Runs the full training, evaluation, and visualization pipeline for Inception-ResNetV2.
    This function is called by the project's main.py script.

    Args:
        dataset_path (str): Path to the dataset folder containing 'real' and 'fake' subfolders.

    Returns:
        dict: A dictionary containing test metrics (accuracy, precision, recall, F1 score, confusion matrix).
              Returns default zero metrics if the pipeline cannot run due to lack of data.
    """
    print(f"\n--- Starting Inception-ResNetV2 Pipeline for Dataset: {dataset_path} ---")

    Config.DATA_ROOT = dataset_path
    set_seed(Config.RANDOM_SEED)

    all_files, all_labels = collect_files(Config.DATA_ROOT, Config.REAL_DIR, Config.FAKE_DIRS)

    if not all_files:
         print("No image files found in the dataset. Exiting pipeline.")
         return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'conf_matrix': [[0,0],[0,0]]}

    (train_loader, val_loader, test_loader,
     files_train, files_val, files_test,
     labels_train, labels_val, labels_test) = create_dataloaders(
        all_files, all_labels, Config.TRAIN_VAL_TEST_SPLIT, Config.BATCH_SIZE, Config.IMAGE_SIZE, Config.RANDOM_SEED
    )

    has_training_data = train_loader is not None
    has_validation_data = val_loader is not None
    has_test_data = test_loader is not None

    model = None # Initialize model outside the if blocks

    if not has_training_data and not has_test_data:
         print("No training or test data available after processing. Cannot train or test. Exiting pipeline.")
         return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'conf_matrix': [[0,0],[0,0]]}


    # Always set up the model, whether training or just testing
    model = setup_model(num_classes=1)
    criterion = nn.BCEWithLogitsLoss() # Needed for both training and testing loss calc


    if has_training_data:
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) if has_validation_data else None
        best_val_metric = -float('inf')

        print("\n--- Starting Training Loop ---")
        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, epoch, Config.NUM_EPOCHS)

            if has_validation_data:
                 val_loss, val_acc, val_precision, val_recall, val_f1, _, _, _, _ = evaluate(model, val_loader, criterion, Config.DEVICE, phase="Validation")

                 print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
                 print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                 print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}, Val Prec: {val_precision:.4f}, Val Rec: {val_recall:.4f}, Val F1: {val_f1:.4f}")

                 current_val_metric = val_acc

                 if current_val_metric > best_val_metric:
                     best_val_metric = current_val_metric
                     try:
                          torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
                          print(f"  Validation metric improved ({best_val_metric:.4f}). Saving model to {Config.BEST_MODEL_PATH}.")
                     except Exception as e:
                          print(f"Error saving model: {e}", file=sys.stderr)

                 scheduler.step(val_loss)

            else:
                 print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
                 print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                 print("  Skipping validation and scheduler step due to no validation data.")

        print("\n--- Training Finished ---")
    else:
        print("\n--- Skipping Training Loop (No training data) ---")


    # 4. Load the best model (if saved) and evaluate on the test set
    test_metrics_dict = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'conf_matrix': [[0,0],[0,0]]} # Default

    if has_test_data:
        print("\n--- Evaluating on Test Set ---")
        # Try loading the best saved model if training happened
        if has_training_data and os.path.exists(Config.BEST_MODEL_PATH):
             try:
                  model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE))
                  print(f"Loaded best model state from {Config.BEST_MODEL_PATH} for testing.")
             except Exception as e:
                  print(f"Error loading best model from {Config.BEST_MODEL_PATH}: {e}. Evaluating the model from the last epoch or ImageNet pre-trained state.", file=sys.stderr)
        elif has_training_data: # Training happened but no best model file was saved/found
            print(f"No best model file found at {Config.BEST_MODEL_PATH}. Evaluating the model from the last epoch.")
        # If has_training_data is False, we evaluate the model loaded by setup_model (likely ImageNet pre-trained)


        (test_loss, test_accuracy, test_precision, test_recall, test_f1,
         test_conf_matrix, true_labels_test, binary_preds_test, test_preds_logits) = evaluate(
            model, test_loader, criterion, Config.DEVICE, phase="Test"
        )

        if len(true_labels_test) > 0:
            print("\n--- Test Results ---")
            print(f"Test Loss:     {test_loss:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Precision:{test_precision:.4f}")
            print(f"Test Recall:   {test_recall:.4f}")
            print(f"Test F1 Score: {test_f1:.4f}")
            print("\nConfusion Matrix:")
            print(test_conf_matrix)

            # Print FP/FN counts explicitly
            # Ensure matrix has expected shape
            if test_conf_matrix.shape == (2, 2):
                 tn, fp, fn, tp = test_conf_matrix.ravel()
                 print(f"\nFalse Positives (Actual Real, Predicted Fake): {fp}")
                 print(f"False Negatives (Actual Fake, Predicted Real): {fn}")
                 print(f"True Positives (Actual Fake, Predicted Fake): {tp}")
                 print(f"True Negatives (Actual Real, Predicted Real): {tn}")
            else:
                 print("\nConfusion matrix shape is not (2, 2). Cannot print FP/FN counts.")


            # Visualize Confusion Matrix
            if test_conf_matrix.shape == (2, 2):
                plt.figure(figsize=(8, 6))
                sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Predicted Real (0)', 'Predicted Fake (1)'],
                            yticklabels=['True Real (0)', 'True Fake (1)'])
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix (Test Set)')
                plt.show()
            else:
                print(f"\nCould not plot confusion matrix (expected shape (2, 2), got {test_conf_matrix.shape})")

            # --- ROC AUC and Precision-Recall Curve ---
            if len(np.unique(true_labels_test)) > 1 and len(test_preds_logits) > 1:
                 print("\n--- Generating ROC AUC and Precision-Recall Curves ---")
                 try:
                     test_probabilities = torch.sigmoid(torch.tensor(test_preds_logits)).numpy()

                     fpr, tpr, _ = roc_curve(true_labels_test, test_probabilities)
                     roc_auc = auc(fpr, tpr)
                     print(f"ROC AUC: {roc_auc:.4f}")

                     precision_vals, recall_vals, _ = precision_recall_curve(true_labels_test, test_probabilities)
                     pr_auc = average_precision_score(true_labels_test, test_probabilities)
                     print(f"PR AUC (Average Precision): {pr_auc:.4f}")

                     plt.figure(figsize=(12, 5))
                     plt.subplot(1, 2, 1)
                     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
                     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                     plt.xlim([0.0, 1.0])
                     plt.ylim([0.0, 1.05])
                     plt.xlabel('False Positive Rate')
                     plt.ylabel('True Positive Rate')
                     plt.title('Receiver Operating Characteristic (ROC) Curve')
                     plt.legend(loc="lower right")
                     plt.grid(True, linestyle='--', alpha=0.6)

                     plt.subplot(1, 2, 2)
                     plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
                     plt.xlim([0.0, 1.0])
                     plt.ylim([0.0, 1.05])
                     plt.xlabel('Recall')
                     plt.ylabel('Precision')
                     plt.title('Precision-Recall Curve')
                     plt.legend(loc="lower left")
                     plt.grid(True, linestyle='--', alpha=0.6)

                     plt.tight_layout()
                     plt.show()

                 except Exception as e:
                     print(f"Error generating AUC/PR curves: {e}", file=sys.stderr)

            elif len(true_labels_test) > 0:
                 print(f"Skipping AUC/PR curves: Need at least 2 different labels (got {len(np.unique(true_labels_test))}) or insufficient samples (got {len(test_preds_logits)}).")
            else:
                 print("No test data available for AUC/PR curves.")


            # --- Grad-CAM Visualization for a sample ---
            if len(files_test) > 0 and hasattr(model, 'get_submodule') and 'TARGET_GRADCAM_LAYER' in dir(Config):
                 try:
                      model.get_submodule(Config.TARGET_GRADCAM_LAYER)
                      can_run_gradcam = True
                 except AttributeError:
                      print(f"\nSkipping Grad-CAM visualization: Target layer '{Config.TARGET_GRADCAM_LAYER}' not found in the model.")
                      can_run_gradcam = False
                 except Exception as e:
                      print(f"\nSkipping Grad-CAM visualization: Error checking target layer '{Config.TARGET_GRADCAM_LAYER}': {e}", file=sys.stderr)
                      can_run_gradcam = False


                 if can_run_gradcam:
                     print("\n--- Generating Grad-CAM Visualization for a sample ---")
                     try:
                         valid_indices = [i for i, path in enumerate(files_test) if os.path.exists(path)]
                         if not valid_indices:
                             print("No valid image files found in the test set for Grad-CAM.")
                             can_run_gradcam = False
                         else:
                             sample_idx = random.choice(valid_indices)
                             sample_img_path = files_test[sample_idx]
                             sample_true_label = labels_test[sample_idx]

                             original_img_pil = Image.open(sample_img_path).convert('RGB')

                             test_transform_single = transforms.Compose([
                                 transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
                             ])
                             input_tensor = test_transform_single(original_img_pil).unsqueeze(0).to(Config.DEVICE)

                             gradcam = None
                             try:
                                 gradcam = GradCAM(model, target_layer_name=Config.TARGET_GRADCAM_LAYER)
                             except Exception as e:
                                  print(f"Could not initialize Grad-CAM: {e}", file=sys.stderr)

                             if gradcam:
                                 try:
                                      cam_map_tensor = gradcam(input_tensor, target_category=None)

                                      if cam_map_tensor is not None:
                                          with torch.no_grad():
                                              output_logit = model(input_tensor).squeeze().item()
                                          predicted_label = 1 if output_logit > 0 else 0
                                          prediction_prob = torch.sigmoid(torch.tensor(output_logit)).item()

                                          cam_map_np = cam_map_tensor.squeeze().cpu().numpy()

                                          cam_title = f"Grad-CAM (Pred: {'Fake' if predicted_label else 'Real'} ({prediction_prob:.4f}), True: {'Fake' if sample_true_label else 'Real'})"
                                          visualize_cam(original_img_pil, cam_map_np, title=cam_title)
                                      else:
                                           print("Grad-CAM computation failed for the sample.")

                                 except Exception as e:
                                      print(f"Error generating or visualizing Grad-CAM for sample {sample_img_path}: {e}", file=sys.stderr)
                                 finally:
                                      if gradcam:
                                          gradcam.remove_hooks()

                             else:
                                  print("Grad-CAM skipped due to initialization failure.")

                     except Exception as e:
                         print(f"Error selecting or processing Grad-CAM sample: {e}", file=sys.stderr)

            else:
                 print("Skipping Grad-CAM visualization: No test files available or target layer not configured/found.")


            # 5. Generate t-SNE Visualization (requires features)
            if len(files_test) > 1 and len(np.unique(labels_test)) > 1:
                 print("\n--- Generating t-SNE Visualization ---")

                 all_features_tsne, all_labels_tsne, _ = extract_features_from_paths(
                     model, files_test, labels_test, Config.DEVICE, Config.IMAGE_SIZE
                 )

                 if len(all_features_tsne) > 1 and len(np.unique(all_labels_tsne)) > 1:
                     print(f"Running t-SNE on {len(all_features_tsne)} samples with {len(np.unique(all_labels_tsne))} classes...")
                     try:
                         n_samples_for_tsne = len(all_features_tsne)
                         perplexity = min(30, n_samples_for_tsne - 1 if n_samples_for_tsne > 1 else 1)
                         perplexity = max(5, perplexity) if n_samples_for_tsne > 5 else (n_samples_for_tsne - 1 if n_samples_for_tsne > 1 else 1)

                         min_samples_for_tsne_lib = max(3 * 2, perplexity + 1) # n_components=2
                         if n_samples_for_tsne < min_samples_for_tsne_lib:
                             print(f"Skipping t-SNE: Too few samples ({n_samples_for_tsne}) for reliable T-SNE (requires at least {min_samples_for_tsne_lib}).")
                         else:
                             tsne = TSNE(n_components=2, random_state=Config.RANDOM_SEED, perplexity=perplexity, init='pca' if all_features_tsne.shape[1] > 50 else 'random')
                             tsne_results = tsne.fit_transform(all_features_tsne)

                             plt.figure(figsize=(10, 8))
                             scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels_tsne, cmap='coolwarm', alpha=0.7)
                             plt.colorbar(scatter, label='Label (0: Real, 1: Fake)')
                             plt.title('t-SNE Visualization of Test Set Features')
                             plt.xlabel('t-SNE Dimension 1')
                             plt.ylabel('t-SNE Dimension 2')
                             plt.grid(True, linestyle='--', alpha=0.5)
                             plt.show()
                     except Exception as e:
                         print(f"Error running t-SNE: {e}", file=sys.stderr)

                 elif len(all_features_tsne) > 0:
                      if len(np.unique(all_labels_tsne)) <= 1:
                           print(f"Skipping t-SNE visualization: Only one unique label ({len(np.unique(all_labels_tsne))}) found in extracted features.")
                      else:
                           print(f"Skipping t-SNE: Need at least 2 samples (got {len(all_features_tsne)}) with extracted features.")
                 else:
                     print("No features extracted for t-SNE.")

            elif len(files_test) > 0:
                 print(f"Skipping t-SNE visualization: Need at least 2 different labels in original test set (got {len(np.unique(labels_test))}).")
            else:
                print("Skipping t-SNE visualization due to no test data.")

            # Update the metrics dictionary to be returned
            test_metrics_dict = {
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1': float(test_f1),
                'conf_matrix': test_conf_matrix.tolist()
            }

        else: # Case where len(true_labels_test) == 0 (no valid samples in test loader)
             print("\nSkipping test evaluation metrics, plots, Grad-CAM, and t-SNE as no valid test samples were processed.")
             # test_metrics_dict remains the default zero metrics

    else: # Case when no test loader was created
        print("\nSkipping test evaluation, confusion matrix, AUC/PR curves, t-SNE, and Grad-CAM due to no test data.")
        # test_metrics_dict remains the default zero metrics


    print("\n--- Inception-ResNetV2 Pipeline Finished ---")

    # Return the test metrics dictionary as required by main.py
    return test_metrics_dict
