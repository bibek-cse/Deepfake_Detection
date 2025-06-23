import os
import sys
import traceback
import numpy as np

# Import your model test functions
from Detection import XceptionNet
from Detection import InceptionNet
from Detection import ResNet50
from Detection import EfficientNet
from Detection import MesoNet
from Detection import ConvxNet
from Detection import InceptionResNetV2  # Newly added model

# List of available models (name, function)
models = [
    ('XceptionNet', XceptionNet.test_model),
    ('InceptionNet', InceptionNet.test_model),
    ('ResNet50', ResNet50.test_model),
    ('EfficientNet', EfficientNet.test_model),
    ('MesoNet', MesoNet.test_model),
    ('ConvxNet', ConvxNet.test_model),
    ('InceptionResNetV2', InceptionResNetV2.test_model),
]


def display_menu():
    """
    Displays the selection menu for available deepfake detection models.
    """
    print("\n================================================")
    print("       Deepfake Detection Models Menu")
    print("================================================")
    for idx, (name, _) in enumerate(models, start=1):
        print(f"{idx}. {name}")
    print(f"{len(models)+1}. Exit")
    print("================================================")


def main():
    """
    Main CLI loop to run deepfake detection pipelines.
    """
    print("=== Welcome to the Deepfake Detection Toolkit ===")

    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()

        if choice == str(len(models) + 1):
            print("Exiting the program. Goodbye!")
            break

        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected_name, test_func = models[int(choice) - 1]
            print(f"\nSelected Model: {selected_name}")

            dataset_path = input("Enter dataset path (e.g., dataset/): ").strip()
            if not os.path.isdir(dataset_path):
                print(f"[ERROR] Invalid dataset path: {dataset_path}")
                continue

            try:
                print(f"\n--- Running {selected_name} Pipeline ---")
                metrics = test_func(dataset_path)

                print(f"\n--- {selected_name} Test Results ---")
                if 'conf_matrix' in metrics and len(metrics['conf_matrix']) == 2:
                    print(f"Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall:    {metrics['recall']:.4f}")
                    print(f"F1 Score:  {metrics['f1']:.4f}")
                    cm = np.array(metrics['conf_matrix'])
                    print("Confusion Matrix:")
                    print(f"[[{cm[0][0]} {cm[0][1]}]\n [{cm[1][0]} {cm[1][1]}]]")
                    print("Labels: 0 - Real, 1 - Fake")
                    print(f"TP: {cm[1][1]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TN: {cm[0][0]}")
                else:
                    print("[WARNING] No valid results returned. Check model execution output.")

                print(f"--- {selected_name} Pipeline Finished ---")

            except FileNotFoundError as e:
                print(f"[ERROR] File not found: {e}")
            except ImportError as e:
                print(f"[ERROR] Import error: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error in {selected_name} pipeline:")
                traceback.print_exc()

        else:
            print("[ERROR] Invalid choice. Please try again.")


if __name__ == "__main__":
    # Dependency check before execution
    try:
        import torch
        import timm
        import sklearn
        import matplotlib
        import seaborn
        import cv2
        import numpy
        import tqdm
        from PIL import Image
    except ImportError as e:
        print("\n[ERROR] Missing required Python libraries.")
        print("Please install them using: pip install -r requirements.txt")
        print(f"Missing module: {e.name}")
        sys.exit(1)

    main()
