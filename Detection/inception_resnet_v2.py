# Detection/InceptionResNetV2.py

import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F # Not needed in this file anymore
from torch.utils.data import DataLoader # Only needed for type hinting
import timm
import sys
# Import utility functions for feature extraction/evaluation within the model module if needed
# from Utility.gradcam_helper import GradCAM # We'll handle GradCAM in main or a dedicated utility function called by main
# from Utility.plots import visualize_cam, plot_confusion_matrix # These are called by main

# --- Configuration ---
class Config:
    # Model-specific config
    IMAGE_SIZE = 299 # Inception-ResNetV2 typically uses 299x299 input
    # Data-related config moved to main.py/Dataset loaders: BATCH_SIZE, TRAIN_VAL_TEST_SPLIT, DATA_ROOT, REAL_DIR, FAKE_DIRS

    # Training config
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 25

    # System config
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grad-CAM config (Target layer name)
    TARGET_GRADCAM_LAYER = 'conv2d_7b' # Common name for the last conv before pooling

    # Normalization constants (used in transforms, defined in base_dataset or passed)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Path to save the best model weights relative to the script's execution directory
    BEST_MODEL_FILENAME = 'best_inception_resnet_v2_model.pth'


print(f"InceptionResNetV2 using device: {Config.DEVICE}")

# Set random seeds for reproducibility (called once in main)
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


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

    # --- Check Target Layer for Grad-CAM ---
    try:
         model.get_submodule(Config.TARGET_GRADCAM_LAYER)
         print(f"Target Grad-CAM layer '{Config.TARGET_GRADCAM_LAYER}' found.")
    except AttributeError:
         print(f"Warning: Target Grad-CAM layer '{Config.TARGET_GRADCAM_LAYER}' not found in the model.", file=sys.stderr)
         print("Grad-CAM might not work correctly for this model instance. Inspect model structure.", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred while verifying target layer: {e}", file=sys.stderr)
    print("-------------------------------------------------------")


    return model

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
    """
    Evaluates the model on a given data loader and returns raw results for metrics/plots.
    Does NOT plot anything.
    """
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
        # Return empty arrays
        return 0.0, np.array([]), np.array([])

    true_labels_flat = np.array(all_true_labels).flatten()
    preds_logits = np.array(all_preds_logits).flatten()

    if len(true_labels_flat) != len(preds_logits):
         print(f"Error: Mismatch between true labels ({len(true_labels_flat)}) and predictions ({len(preds_logits)}) length during {phase} evaluation. Truncating.", file=sys.stderr)
         min_len = min(len(true_labels_flat), len(preds_logits))
         true_labels_flat = true_labels_flat[:min_len]
         preds_logits = preds_logits[:min_len]
         if min_len == 0:
             print(f"No valid samples after length check for {phase}. Returning defaults.", file=sys.stderr)
             return 0.0, np.array([]), np.array([])

    epoch_loss = running_loss / len(true_labels_flat)

    # Return loss, true labels, and raw logits
    return epoch_loss, true_labels_flat, preds_logits


# --- Feature Extraction Function (for t-SNE) ---
def extract_features_from_paths(model, file_paths, device, image_size,
                                imagenet_mean=[0.485, 0.456, 0.406], imagenet_std=[0.229, 0.224, 0.225]):
    """
    Extracts features for a list of image file paths using the model's feature extractor.
    Does NOT use a DataLoader.
    """
    model.eval()

    feature_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    all_features = []
    extracted_paths = [] # Return paths to link features back to images

    if not file_paths:
         print("No file paths provided for feature extraction.")
         return np.array([]), []

    print(f"\n--- Extracting features from {len(file_paths)} images ---")
    # No need for labels here, we'll get labels from the corresponding list provided to main
    feature_loop = tqdm(file_paths, total=len(file_paths), desc="Extracting Features", file=sys.stdout)

    with torch.no_grad():
        for file_path in feature_loop:
            try:
                if not os.path.exists(file_path):
                     continue
                img = Image.open(file_path).convert('RGB')
                img_tensor = feature_transform(img).unsqueeze(0).to(device)

                if hasattr(model, 'forward_features') and hasattr(model, 'global_pool'):
                     features = model.forward_features(img_tensor)
                     pooled_features = model.global_pool(features)
                else:
                    raise NotImplementedError("Feature extraction method not compatible with model structure (missing forward_features or global_pool).")

                all_features.append(pooled_features.cpu().numpy().flatten())
                extracted_paths.append(file_path)

            except NotImplementedError as e:
                 print(f"\n{e}", file=sys.stderr)
                 print("Please check your timm version or model architecture.", file=sys.stderr)
                 return np.array([]), [] # Stop feature extraction if the method is fundamentally incompatible

            except Exception as e:
                print(f"\nError extracting feature for {file_path}: {e}. Skipping.", file=sys.stderr)
                continue

    print(f"Successfully extracted features for {len(all_features)} images.")
    return np.array(all_features), extracted_paths


# --- Main Test Function for the Model (Called by main.py) ---
def test_model(data_loaders, data_lists, model_config):
    """
    Runs the training and evaluation pipeline for Inception-ResNetV2.
    This function is called by main.py and uses data components prepared by a Dataset module.

    Args:
        data_loaders (tuple): (train_loader, val_loader, test_loader)
        data_lists (tuple): (files_train, files_val, files_test, labels_train, labels_val, labels_test)
        model_config (object): A configuration object (e.g., Config class from this file)

    Returns:
        dict: A dictionary containing results for analysis/plotting in main.py.
              Returns default zero metrics if the pipeline cannot run due to lack of data.
    """
    train_loader, val_loader, test_loader = data_loaders
    files_train, files_val, files_test, labels_train, labels_val, labels_test = data_lists

    has_training_data = train_loader is not None
    has_validation_data = val_loader is not None
    has_test_data = test_loader is not None

    # Setup model (using model-specific config like IMAGE_SIZE)
    model = setup_model(num_classes=1)
    criterion = nn.BCEWithLogitsLoss()

    # --- Training ---
    if has_training_data:
        optimizer = optim.AdamW(model.parameters(), lr=model_config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) if has_validation_data else None
        best_val_metric = -float('inf')
        best_model_path = model_config.BEST_MODEL_FILENAME # Use the filename from config

        print("\n--- Starting Training Loop ---")
        for epoch in range(model_config.NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, model_config.DEVICE, epoch, model_config.NUM_EPOCHS)

            if has_validation_data:
                 # evaluate returns loss, true_labels, preds_logits
                 val_loss, val_true_labels, val_preds_logits = evaluate(model, val_loader, criterion, model_config.DEVICE, phase="Validation")

                 # Calculate standard metrics for validation from raw outputs
                 if len(val_true_labels) > 0:
                     val_binary_preds = (val_preds_logits > 0).astype(int)
                     val_acc = accuracy_score(val_true_labels, val_binary_preds)
                     val_precision = precision_score(val_true_labels, val_binary_preds, zero_division=0)
                     val_recall = recall_score(val_true_labels, val_binary_preds, zero_division=0)
                     val_f1 = f1_score(val_true_labels, val_binary_preds, zero_division=0)
                     val_conf_matrix = confusion_matrix(val_true_labels, val_binary_preds).tolist()
                 else:
                     val_acc, val_precision, val_recall, val_f1, val_conf_matrix = 0.0, 0.0, 0.0, 0.0, [[0,0],[0,0]]


                 print(f"\nEpoch {epoch+1}/{model_config.NUM_EPOCHS} Summary:")
                 print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                 print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}, Val Prec: {val_precision:.4f}, Val Rec: {val_recall:.4f}, Val F1: {val_f1:.4f}")

                 # Decide which metric to use for saving the best model (e.g., validation accuracy)
                 current_val_metric = val_acc # Can change this to val_f1 etc.

                 if current_val_metric > best_val_metric:
                     best_val_metric = current_val_metric
                     try:
                          torch.save(model.state_dict(), best_model_path)
                          print(f"  Validation metric improved ({best_val_metric:.4f}). Saving model to {best_model_path}.")
                     except Exception as e:
                          print(f"Error saving model: {e}", file=sys.stderr)

                 scheduler.step(val_loss) # Often stepped with loss

            else:
                 print(f"\nEpoch {epoch+1}/{model_config.NUM_EPOCHS} Summary:")
                 print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                 print("  Skipping validation and scheduler step due to no validation data.")

        print("\n--- Training Finished ---")
    else:
        print("\n--- Skipping Training Loop (No training data) ---")


    # --- Evaluation on Test Set ---
    results = {} # Dictionary to store results to be returned to main.py

    if has_test_data:
        print("\n--- Evaluating on Test Set ---")
        # Try loading the best saved model if training happened and a model was saved
        best_model_path = model_config.BEST_MODEL_FILENAME
        if has_training_data and os.path.exists(best_model_path):
             try:
                  model.load_state_dict(torch.load(best_model_path, map_location=model_config.DEVICE))
                  print(f"Loaded best model state from {best_model_path} for testing.")
             except Exception as e:
                  print(f"Error loading best model from {best_model_path}: {e}. Evaluating the model from the last epoch or ImageNet pre-trained state.", file=sys.stderr)
        elif has_training_data:
            print(f"No best model file found at {best_model_path}. Evaluating the model from the last epoch.")
        # If has_training_data is False, we evaluate the model loaded by setup_model (likely ImageNet pre-trained)


        # evaluate returns loss, true_labels, preds_logits
        test_loss, test_true_labels, test_preds_logits = evaluate(model, test_loader, criterion, model_config.DEVICE, phase="Test")

        # Calculate standard metrics for test from raw outputs
        if len(test_true_labels) > 0:
            test_binary_preds = (test_preds_logits > 0).astype(int) # Binary prediction
            test_accuracy = accuracy_score(test_true_labels, test_binary_preds)
            test_precision = precision_score(test_true_labels, test_binary_preds, zero_division=0)
            test_recall = recall_score(test_true_labels, test_binary_preds, zero_division=0)
            test_f1 = f1_score(test_true_labels, test_binary_preds, zero_division=0)
            test_conf_matrix = confusion_matrix(test_true_labels, test_binary_preds)

            # Store metrics in results dictionary
            results['accuracy'] = float(test_accuracy)
            results['precision'] = float(test_precision)
            results['recall'] = float(test_recall)
            results['f1'] = float(test_f1)
            results['conf_matrix'] = test_conf_matrix.tolist() # Return as list for compatibility
            results['test_loss'] = float(test_loss) # Include test loss

            # Store raw results for plots in main.py
            results['true_labels'] = test_true_labels.tolist() # Convert to list
            results['binary_preds'] = test_binary_preds.tolist() # Convert to list
            results['preds_logits'] = test_preds_logits.tolist() # Convert to list (or keep as np array if Utility expects it)

            # Store test file paths and original labels for Grad-CAM/t-SNE sample selection
            # These should correspond to the *successful* samples from the test loader if possible.
            # A simpler approach is to return the original split lists and handle potential load errors in Utility.
            # Let's return the original lists from create_dataloaders for simplicity.
            results['test_file_paths'] = files_test
            results['test_original_labels'] = labels_test


        else: # Case where len(test_true_labels) == 0 (no valid samples processed)
             print("\nSkipping test evaluation metrics as no valid test samples were processed.")
             # Return default zero metrics in results dictionary
             results['accuracy'] = 0.0
             results['precision'] = 0.0
             results['recall'] = 0.0
             results['f1'] = 0.0
             results['conf_matrix'] = [[0,0],[0,0]]
             results['test_loss'] = 0.0
             results['true_labels'] = []
             results['binary_preds'] = []
             results['preds_logits'] = []
             results['test_file_paths'] = []
             results['test_original_labels'] = []


        # --- Feature Extraction for t-SNE ---
        if len(results['test_file_paths']) > 1 and len(np.unique(results['test_original_labels'])) > 1: # Need at least 2 samples & 2 classes
             print("\n--- Extracting features for t-SNE ---")
             # Pass original file paths and model config (for device, image_size, normalization)
             extracted_features, extracted_paths = extract_features_from_paths(
                 model, results['test_file_paths'], model_config.DEVICE, model_config.IMAGE_SIZE,
                 imagenet_mean=model_config.IMAGENET_MEAN, imagenet_std=model_config.IMAGENET_STD
             )

             if len(extracted_features) > 1:
                  # Need to align extracted features/paths with their original labels
                  # Assuming the order is preserved during extraction or matching by path
                  # Simple case: assume order is preserved.
                  extracted_labels = [results['test_original_labels'][files_test.index(p)] for p in extracted_paths]

                  results['tsne_features'] = extracted_features # Keep as numpy array
                  results['tsne_labels'] = extracted_labels # Keep as list
                  results['tsne_paths'] = extracted_paths # Keep as list
             else:
                  print("Not enough features extracted for t-SNE.")
                  results['tsne_features'] = np.array([])
                  results['tsne_labels'] = []
                  results['tsne_paths'] = []
        else:
             print("Skipping t-SNE feature extraction: Not enough test files or unique labels.")
             results['tsne_features'] = np.array([])
             results['tsne_labels'] = []
             results['tsne_paths'] = []


    else: # Case when no test loader was created
        print("\nSkipping test evaluation, feature extraction for t-SNE.")
        # results dictionary remains the default zero/empty state

    print("\n--- InceptionResNetV2 Model Pipeline Section Finished ---")

    # Return the results dictionary for main.py to handle plotting/final output
    return results

# Note: The __main__ block is removed as this file is imported as a module.
