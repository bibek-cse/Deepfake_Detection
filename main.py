# main.py

import argparse
import os
import traceback
import sys
import numpy as np # Needed for processing confusion matrix lists

# Import detection models (each should have a test_model function)
# Add imports for your other models here when implemented
from Detection import InceptionResNetV2
# from Detection import XceptionNet
# from Detection import InceptionNet
# from Detection import ResNet50
# from Detection import EfficientNet
# from Detection import MesoNet
# from Detection import ConvxNet

# Import dataset loaders (each should have a load_and_split_data function)
# Add imports for your other datasets here when implemented
from Dataset import Celeb_DF_v2
from Dataset import Deepfake_Dataset # Placeholder

# Import plotting utilities
from Utility import plots
from Utility import gradcam_helper

# --- Global Configuration for the entire pipeline ---
class GlobalConfig:
    # Data Splitting Ratios
    TRAIN_VAL_TEST_SPLIT = [0.7, 0.15, 0.15] # 70% train, 15% validation, 15% test

    # Batch Size for DataLoaders
    BATCH_SIZE = 32

    # Random Seed for reproducibility
    RANDOM_SEED = 42

    # Image size expected by the model (this should ideally come from model_config, but define a default)
    # The dataset loader might need this. InceptionResNetV2 uses 299. Other models might use 224 etc.
    # We'll pass this from the model's specific config if available.
    DEFAULT_IMAGE_SIZE = 299

    # ImageNet mean and std for normalization (standard)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Default subfolder names for real/fake data
    REAL_DIR_NAME = 'real'
    FAKE_DIR_NAMES = ['fake'] # List to support multiple fake sources if needed


# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Map user-friendly string names to the actual modules and their configs/test functions ---
# Each entry: 'ModelName': (model_module, model_config_class, dataset_module_used_by_model)
# The model_module must have a `test_model` function.
# The dataset_module must have a `load_and_split_data` function.
model_pipeline_map = {
    'InceptionResNetV2': (InceptionResNetV2, InceptionResNetV2.Config, Celeb_DF_v2), # Link to Celeb_DF_v2 loader (you might link to Deepfake_Dataset or a generic one)
    # Add other models here when implemented:
    # 'XceptionNet': (XceptionNet, XceptionNet.Config, Celeb_DF_v2), # Example
}

# --- Map dataset names to their loader functions ---
dataset_loader_map = {
    'Celeb_DF_v2': Celeb_DF_v2.load_and_split_data,
    'Deepfake_Dataset': Deepfake_Dataset.load_and_split_data, # Placeholder
    # Add other dataset loaders here
}


def display_menu():
     print("\n================================================")
     print("       Deepfake Detection Models Menu")
     print("================================================")
     # Use model_pipeline_map keys for display names
     for i, name in enumerate(model_pipeline_map.keys(), 1):
         print(f"{i}. {name}")
     print(f"{len(model_pipeline_map)+1}. Exit")
     print("================================================")


def run_pipeline(model_name_str, dataset_path, dataset_loader_str=None):
     """
     Orchestrates the data loading, model pipeline execution, and results visualization.
     """
     pipeline_components = model_pipeline_map.get(model_name_str)

     if pipeline_components is None:
         print(f"Error: Unknown model name '{model_name_str}'. Available models are: {list(model_pipeline_map.keys())}", file=sys.stderr)
         return # Exit the function

     model_module, model_config, default_dataset_module = pipeline_components

     # Determine which dataset loader to use
     dataset_module_to_use = default_dataset_module # Start with the default linked in model_pipeline_map
     if dataset_loader_str and dataset_loader_str in dataset_loader_map:
          dataset_module_to_use = sys.modules[dataset_loader_map[dataset_loader_str].__module__] # Get the module object
          print(f"Using specified dataset loader: {dataset_loader_str}")
     elif dataset_loader_str:
          print(f"Warning: Specified dataset loader '{dataset_loader_str}' not found. Using default loader linked to the model ({default_dataset_module.__name__}).", file=sys.stderr)


     if not hasattr(dataset_module_to_use, 'load_and_split_data'):
          print(f"Error: Dataset module '{dataset_module_to_use.__name__}' does not have a 'load_and_split_data' function.", file=sys.stderr)
          return

     # 1. Load and split data using the selected dataset module
     print(f"\n--- Loading and Splitting Data using {dataset_module_to_use.__name__} ---")
     try:
          # Pass global config parameters and model's image size requirement
          data_loaders = dataset_module_to_use.load_and_split_data(
              dataset_path,
              GlobalConfig.TRAIN_VAL_TEST_SPLIT,
              GlobalConfig.BATCH_SIZE,
              model_config.IMAGE_SIZE if hasattr(model_config, 'IMAGE_SIZE') else GlobalConfig.DEFAULT_IMAGE_SIZE, # Use model's size if defined, else default
              GlobalConfig.RANDOM_SEED,
              GlobalConfig.REAL_DIR_NAME,
              GlobalConfig.FAKE_DIR_NAMES
          )
          train_loader, val_loader, test_loader, files_train, files_val, files_test, labels_train, labels_val, labels_test = data_loaders

          # Check if any data was loaded/split
          if train_loader is None and val_loader is None and test_loader is None:
               print("No data available in any split after loading. Exiting pipeline.")
               return # Cannot proceed without data

          has_training_data = train_loader is not None
          has_validation_data = val_loader is not None
          has_test_data = test_loader is not None

     except FileNotFoundError as e:
          print(f"Error loading data: Directory not found - {e}", file=sys.stderr)
          print("Please check the dataset path and subfolder names ('real', 'fake').", file=sys.stderr)
          return
     except Exception as e:
          print(f"An unexpected error occurred during data loading and splitting: {e}", file=sys.stderr)
          traceback.print_exc()
          return


     # 2. Run the model's training and evaluation pipeline
     print(f"\n--- Running {model_name_str} Model Pipeline ---")
     try:
         # Pass data loaders and lists, and the model-specific config object
         data_components = (train_loader, val_loader, test_loader)
         list_components = (files_train, files_val, files_test, labels_train, labels_val, labels_test)

         # The model's test_model returns a dictionary of results
         results = model_module.test_model(data_components, list_components, model_config)

         # Check if results dictionary is valid (e.g., not the default zero dictionary from model file)
         if 'conf_matrix' not in results or len(results['conf_matrix']) != 2 or len(results['conf_matrix'][0]) != 2:
              print("\nModel pipeline finished, but no valid test results were returned.", file=sys.stderr)
              print("Check the model's output for errors.", file=sys.stderr)
              return # Exit if model pipeline failed to produce results

     except Exception as e:
         print(f"An unexpected error occurred during the {model_name_str} model pipeline: {e}", file=sys.stderr)
         traceback.print_exc()
         return


     # 3. Display final metrics and generate plots using Utility functions
     print(f"\n--- Final Results and Visualizations for {model_name_str} ---")
     # Print metrics received from the model's test_model function
     print(f"Test Loss:     {results.get('test_loss', 'N/A'):.4f}")
     print(f"Test Accuracy: {results.get('accuracy', 'N/A'):.4f}")
     print(f"Test Precision:{results.get('precision', 'N/A'):.4f}")
     print(f"Test Recall:   {results.get('recall', 'N/A'):.4f}")
     print(f"Test F1 Score: {results.get('f1', 'N/A'):.4f}")


     # Ensure data exists for plotting
     if results.get('true_labels') and results.get('binary_preds') and len(results['true_labels']) > 0:
         true_labels_np = np.array(results['true_labels'])
         binary_preds_np = np.array(results['binary_preds'])
         preds_logits_np = np.array(results['preds_logits']) # Needed for probabilities

         # Confusion Matrix Plot and FP/FN counts
         plots.plot_confusion_matrix(true_labels_np, binary_preds_np)

         # ROC AUC and Precision-Recall Curves
         # Need probabilities (sigmoid of logits) for curves
         if len(preds_logits_np) > 0:
              test_probabilities = torch.sigmoid(torch.tensor(preds_logits_np)).numpy()
              plots.plot_roc_pr_curves(true_labels_np, test_probabilities)
         else:
              print("Skipping ROC/PR plots: No logits available.")


         # Grad-CAM Visualization (Requires raw image path, model, and original labels)
         # Need to re-instantiate the model here or get the model object from the model_module
         # Re-instantiating is simpler but less efficient. Getting the model object is cleaner.
         # Let's modify the model's test_model to return the trained model object?
         # Or, the GradCAM helper can take the model module and config and load weights internally.
         # Option: Pass the model module and config, and let GradCAM helper instantiate/load.
         # This keeps the model object handling within its module or helper.

         # To run Grad-CAM, we need:
         # 1. The trained model instance (or ability to load it)
         # 2. The target layer name (from model_config)
         # 3. Original test images (paths) and labels (from data_lists)
         # 4. Transforms (can use the same val_test_transform or pass necessary config)
         # 5. Device

         if results.get('test_file_paths') and results.get('test_original_labels') and hasattr(model_config, 'TARGET_GRADCAM_LAYER'):
             print("\n--- Generating Grad-CAM Visualization for a sample ---")
             try:
                 # Re-load the model state dictionary into a fresh model instance for safety
                 # Or assume the model_module keeps the trained instance internally and provides access?
                 # Let's stick to loading the saved best model state for CAM/t-SNE for consistency.
                 # Load the best model state again for visualization tasks if saved.
                 # This is slightly redundant but ensures we use the 'best' weights.
                 # A better approach is to have the model's test_model return the final trained model object.
                 # Let's update the InceptionResNetV2 test_model to return the model object.

                 # Assuming model_module.test_model now returns (results_dict, trained_model_instance)
                 # If not, we'd need to load the model state here. Let's refine test_model signature.
                 # (Refactored test_model return signature in InceptionResNetV2.py)
                 # Now we need the model object from the results tuple or re-load.
                 # Let's assume test_model returns results_dict only, and we reload the best state here if training happened.
                 # If training didn't happen, we use the ImageNet pre-trained state from setup_model.
                 temp_model_for_vis = model_module.setup_model(num_classes=1) # Get a fresh instance
                 best_model_path = model_config.BEST_MODEL_FILENAME
                 if os.path.exists(best_model_path): # Check if training actually saved a model
                     try:
                         temp_model_for_vis.load_state_dict(torch.load(best_model_path, map_location=model_config.DEVICE))
                         print(f"(Grad-CAM/t-SNE) Loaded best model state from {best_model_path}.")
                     except Exception as e:
                         print(f"(Grad-CAM/t-SNE) Error loading best model state: {e}. Using current model state.", file=sys.stderr)
                         # temp_model_for_vis will have ImageNet pretrained weights if training skipped, or last epoch weights if training failed to save.


                 # Pick a random sample from the test set files
                 valid_indices = [i for i, path in enumerate(results['test_file_paths']) if os.path.exists(path)]
                 if valid_indices:
                     sample_idx = random.choice(valid_indices)
                     sample_img_path = results['test_file_paths'][sample_idx]
                     sample_true_label = results['test_original_labels'][sample_idx]

                     try:
                         # Load original image and apply test transforms (needs image size and normalization from config)
                         original_img_pil = Image.open(sample_img_path).convert('RGB')

                         # Use transforms consistent with test evaluation
                         _, test_transform = Dataset.base_dataset.get_base_transforms(
                             model_config.IMAGE_SIZE if hasattr(model_config, 'IMAGE_SIZE') else GlobalConfig.DEFAULT_IMAGE_SIZE,
                             model_config.IMAGENET_MEAN if hasattr(model_config, 'IMAGENET_MEAN') else GlobalConfig.IMAGENET_MEAN,
                             model_config.IMAGENET_STD if hasattr(model_config, 'IMAGENET_STD') else GlobalConfig.IMAGENET_STD
                         )
                         input_tensor = test_transform(original_img_pil).unsqueeze(0).to(model_config.DEVICE)

                         # Compute CAM
                         gradcam_instance = gradcam_helper.GradCAM(temp_model_for_vis, model_config.TARGET_GRADCAM_LAYER)
                         cam_map_tensor = gradcam_instance(input_tensor, target_category=None) # Binary classification (target_category=None for single logit)

                         if cam_map_tensor is not None:
                             # Get model's prediction for this specific sample
                             with torch.no_grad():
                                 output_logit = temp_model_for_vis(input_tensor).squeeze().item()
                             predicted_label = 1 if output_logit > 0 else 0
                             prediction_prob = torch.sigmoid(torch.tensor(output_logit)).item()

                             cam_map_np = cam_map_tensor.squeeze().cpu().numpy()

                             cam_title = f"Grad-CAM (Pred: {'Fake' if predicted_label else 'Real'} ({prediction_prob:.4f}), True: {'Fake' if sample_true_label else 'Real'})"
                             gradcam_helper.visualize_cam(original_img_pil, cam_map_np, title=cam_title)
                         else:
                             print("Grad-CAM computation failed for the sample.")

                     except Exception as e:
                         print(f"Error generating or visualizing Grad-CAM for sample {sample_img_path}: {e}", file=sys.stderr)
                         traceback.print_exc()
                     finally:
                         # Ensure hooks are removed
                         if 'gradcam_instance' in locals() and gradcam_instance:
                              gradcam_instance.remove_hooks()
                         # Clean up temporary model instance? Not strictly necessary in Python but good practice
                         # del temp_model_for_vis
                         # torch.cuda.empty_cache()


                 else:
                      print("No valid image files found in the test set for Grad-CAM.")
             except Exception as e:
                 print(f"Error setting up Grad-CAM: {e}", file=sys.stderr)
                 traceback.print_exc()

         else:
              print("Skipping Grad-CAM visualization: Not enough test files or target layer not defined in model config.")


         # t-SNE Visualization (Requires extracted features and original labels from test set)
         if results.get('tsne_features') is not None and len(results['tsne_features']) > 1 and len(results['tsne_labels']) > 1:
             print("\n--- Generating t-SNE Visualization ---")
             try:
                  plots.plot_tsne(np.array(results['tsne_features']), np.array(results['tsne_labels']))
             except Exception as e:
                  print(f"Error generating t-SNE plot: {e}", file=sys.stderr)
                  traceback.print_exc()
         else:
              print("Skipping t-SNE visualization: Not enough features or unique labels for plotting.")


     else: # Case where no test data or evaluation produced results
          print("\nSkipping test results display and plots as no valid test samples were processed.")

     print(f"\n--- Pipeline for {model_name_str} Finished ---")


def main():
    # --- Dependency Check ---
    try:
        # Import all modules used at a high level or needed for basic functionality checks
        import torch, timm, sklearn, matplotlib, seaborn, cv2, numpy, tqdm, PIL
        # Check specific utility imports
        from Utility import plots, gradcam_helper
        from Dataset.base_dataset import DeepfakeDataset, collate_fn_robust, get_base_transforms
    except ImportError as e:
        print("------------------------------------------------------------", file=sys.stderr)
        print("ERROR: Missing required libraries.")
        print(f"Please install dependencies using: pip install -r requirements.txt", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("------------------------------------------------------------", file=sys.stderr)
        sys.exit(1) # Exit if dependencies are not met

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Run deepfake detection models on a dataset.',
                                     formatter_class=argparse.RawTextHelpFormatter) # Helps format description/help

    parser.add_argument('--model', type=str, choices=model_pipeline_map.keys(),
                        help=f'Specify the model to run. Available: {", ".join(model_pipeline_map.keys())}')

    parser.add_argument('--dataset', type=str,
                        help='Path to the dataset directory containing "real" and "fake" subfolders.')

    parser.add_argument('--dataset-loader', type=str, choices=dataset_loader_map.keys(),
                         help=f'Optional: Specify which dataset loading script to use if different from model default.\nAvailable loaders: {", ".join(dataset_loader_map.keys())}')

    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit.')

    parser.add_argument('--list-loaders', action='store_true',
                        help='List available dataset loaders and exit.')


    args = parser.parse_args()

    # --- Handle List Commands ---
    if args.list_models:
        print("Available Models:")
        for name in model_pipeline_map.keys():
            print(f"- {name}")
        sys.exit(0)

    if args.list_loaders:
        print("Available Dataset Loaders:")
        for name in dataset_loader_map.keys():
            print(f"- {name}")
        sys.exit(0)

    # --- Run Pipeline or Enter Interactive Mode ---
    if args.model and args.dataset:
        # Command-line mode
        set_seed(GlobalConfig.RANDOM_SEED) # Set seed once at the start
        run_pipeline(args.model, args.dataset, args.dataset_loader)
    else:
        # Interactive mode (if no model or dataset argument provided)
        print("Running in interactive mode.")
        print("Use --help for command-line options.")
        set_seed(GlobalConfig.RANDOM_SEED) # Set seed once at the start for interactive runs

        while True:
            display_menu()
            choice = input("Enter your choice: ").strip()
            model_names_list = list(model_pipeline_map.keys())

            if choice.isdigit() and 1 <= int(choice) <= len(model_names_list):
                selected_model_name = model_names_list[int(choice)-1]
                print(f"\nYou selected: {selected_model_name}")

                # Prompt for dataset path in interactive mode
                dataset_path = input(f"Enter the path to your dataset for {selected_model_name} (e.g., ./my_data/): ").strip()

                # In interactive mode, we use the default dataset loader linked to the model
                # Or you could prompt the user to choose a loader too if needed
                run_pipeline(selected_model_name, dataset_path, dataset_loader_str=None) # Pass None to use default linked loader


            elif choice == str(len(model_names_list)+1):
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
