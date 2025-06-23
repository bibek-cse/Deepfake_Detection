# main.py (Modified for command-line arguments)

import argparse # Import argparse
import os
import traceback
import sys

# Import models - make sure these files exist in the Detection/ directory
# and each has a test_model function that takes a dataset_path and returns metrics
from Detection import XceptionNet
from Detection import InceptionNet
from Detection import ResNet50
from Detection import EfficientNet
from Detection import MesoNet
from Detection import ConvxNet
from Detection import InceptionResNetV2 # <-- The new model

# Map user-friendly string names to the actual model test functions
model_map = {
    'XceptionNet': XceptionNet.test_model,
    'InceptionNet': InceptionNet.test_model,
    'ResNet50': ResNet50.test_model,
    'EfficientNet': EfficientNet.test_model,
    'MesoNet': MesoNet.test_model,
    'MesoNet': MesoNet.test_model, # Corrected typo from original README
    'ConvxNet': ConvxNet.test_model,
    'InceptionResNetV2': InceptionResNetV2.test_model # <-- Added the new model
}

def display_menu():
    # ... (existing menu display code) ...
     print("\n================================================")
     print("       Deepfake Detection Models Menu")
     print("================================================")
     # Use model_map keys for display names
     for i, name in enumerate(model_map.keys(), 1):
         print(f"{i}. {name}")
     print(f"{len(model_map)+1}. Exit")
     print("================================================")


def run_model_pipeline(model_name_str, dataset_path):
     """Finds the model and runs its test_model function."""
     test_func = model_map.get(model_name_str)

     if test_func is None:
         print(f"Error: Unknown model name '{model_name_str}'. Available models are: {list(model_map.keys())}", file=sys.stderr)
         return # Exit the function

     # Validate dataset path exists
     if not os.path.isdir(dataset_path):
          print(f"Error: Dataset path not found or is not a directory: {dataset_path}", file=sys.stderr)
          print("Please check the path and try again.")
          return # Exit the function

     print(f"\n--- Running {model_name_str} Pipeline for Dataset: {dataset_path} ---")

     try:
         # Call the test function for the selected model
         # This function handles its own pipeline including training, evaluation, and visualization
         metrics = test_func(dataset_path)

         print(f"\n--- Final Test Results for {model_name_str} ---")
         # Check if metrics dictionary is valid (e.g., not the default zero dictionary)
         # We assume a confusion matrix shape of (2,2) indicates valid results
         if 'conf_matrix' in metrics and len(metrics['conf_matrix']) == 2 and len(metrics['conf_matrix'][0]) == 2:
             print(f"Accuracy:  {metrics['accuracy']:.4f}")
             print(f"Precision: {metrics['precision']:.4f}")
             print(f"Recall:    {metrics['recall']:.4f}")
             print(f"F1 Score:  {metrics['f1']:.4f}")
             print("Confusion Matrix:")
             try:
                  cm = np.array(metrics['conf_matrix'])
                  print(f"[[{cm[0,0]} {cm[0,1]}]\n [{cm[1,0]} {cm[1,1]}]]")
                  print("Labels: 0: Real, 1: Fake")
                  print("Rows are True Labels, Columns are Predicted Labels")
                  print(f"  TP (Fake Correct): {cm[1,1]}, FP (Real Predicted Fake): {cm[0,1]}")
                  print(f"  FN (Fake Predicted Real): {cm[1,0]}, TN (Real Correct): {cm[0,0]}")
             except Exception as e:
                  print("Could not parse confusion matrix for detailed print:", e, file=sys.stderr)

         else:
             print("No valid test results were returned by the model pipeline.")
             print("Please check the console output above for potential errors during execution.", file=sys.stderr)

         print(f"--- {model_name_str} Pipeline Finished ---")

     except FileNotFoundError as e:
         print(f"Error: Required file not found - {e}", file=sys.stderr)
         print("Please ensure model weights or dataset subfolders exist.", file=sys.stderr)
     except ImportError as e:
         print(f"Error: Could not import required module - {e}", file=sys.stderr)
         print("Please check your Python environment and installed libraries (requirements.txt).", file=sys.stderr)
     except Exception as e:
         print(f"An unexpected error occurred during the {model_name_str} pipeline:", file=sys.stderr)
         traceback.print_exc()
         print(f"Error details: {e}", file=sys.stderr)


def main():
    # Check if required libraries are installed (basic check)
    try:
        # Import all modules needed by any model or main script
        import torch, timm, sklearn, matplotlib, seaborn, cv2, numpy, tqdm, PIL
        from PIL import Image
    except ImportError as e:
        print("------------------------------------------------------------", file=sys.stderr)
        print("ERROR: Missing required libraries.", file=sys.stderr)
        print(f"Please install dependencies using: pip install -r requirements.txt", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        print("------------------------------------------------------------", file=sys.stderr)
        sys.exit(1) # Exit if dependencies are not met


    # --- Command Line Argument Handling ---
    parser = argparse.ArgumentParser(description='Run deepfake detection models on a dataset.')
    parser.add_argument('--model', type=str, choices=model_map.keys(),
                        help=f'Specify the model to run. Choices: {list(model_map.keys())}')
    parser.add_argument('--dataset', type=str,
                        help='Path to the dataset directory containing "real" and "fake" subfolders.')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models and exit.')

    args = parser.parse_args()

    if args.list_models:
        print("Available Models:")
        for name in model_map.keys():
            print(f"- {name}")
        sys.exit(0) # Exit after listing

    # --- Interactive Mode or Command-line Mode ---
    if args.model and args.dataset:
        # Command-line mode
        run_model_pipeline(args.model, args.dataset)
    else:
        # Interactive mode (if no or insufficient arguments provided)
        print("Running in interactive mode.")
        print("Use --help for command-line options.")
        while True:
            display_menu()
            choice = input("Enter your choice: ").strip()
            if choice == str(len(model_map)+1): # Adjust exit choice number
                print("Exiting the program. Goodbye!")
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(model_map):
                # Get model name from dictionary keys based on index
                selected_model_name = list(model_map.keys())[int(choice)-1]
                print(f"\nYou selected: {selected_model_name}")
                dataset_path = input(f"Enter the path to your dataset for {selected_model_name} (e.g., dataset/): ").strip()
                run_model_pipeline(selected_model_name, dataset_path)
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
