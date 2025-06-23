import XceptionNet
import InceptionNet
import ResNet50
import EfficientNet
import MesoNet
import ConvxNet

models = [
    ('XceptionNet', XceptionNet.test_model),
    ('InceptionNet', InceptionNet.test_model),
    ('ResNet50', ResNet50.test_model),
    ('EfficientNet', EfficientNet.test_model),
    ('MesoNet', MesoNet.test_model),
    ('ConvxNet', ConvxNet.test_model)
]

def display_menu():
    print("\nSelect a model to test your dataset:")
    for i, (name, _) in enumerate(models, 1):
        print(f"{i}. {name}")
    print(f"{len(models)+1}. Exit")

def main():
    print("Welcome to the Deepfake Detection Toolkit")
    while True:
        display_menu()
        choice = input("Enter your choice: ").strip()
        if choice == str(len(models)+1):
            print("Exiting the program. Goodbye!")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(models):
            dataset_path = input("Enter the path to your dataset: ").strip()
            try:
                _, test_func = models[int(choice)-1]
                metrics = test_func(dataset_path)
                print(f"\nResults from {models[int(choice)-1][0]}:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print("Confusion Matrix:")
                print(metrics['conf_matrix'])
            except FileNotFoundError:
                print("Dataset not found. Please check the path and try again.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
