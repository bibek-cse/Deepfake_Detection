import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def test_model(dataset_path, batch_size=32, image_size=512):
    """
    Tests the Inception-ResNetV2 model on the provided dataset and returns evaluation metrics.

    Args:
        dataset_path (str): Path to the dataset folder containing 'real' and 'fake' subfolders.
        batch_size (int): Batch size for data loading. Default is 32.
        image_size (int): Size to resize images to. Default is 299 for Inception-ResNetV2.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1 score, and confusion matrix.
    """
    weights_file = 'best_inception_resnet_v2_model.pth'
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Model weights not found. Please download from [link] and place in the current directory.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations for the test set
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    dataset = ImageFolder(root=dataset_path, transform=test_transform)

    # Adjust labels to ensure 'real' is 0 and 'fake' is 1
    if 'real' in dataset.class_to_idx and 'fake' in dataset.class_to_idx:
        real_idx = dataset.class_to_idx['real']
        fake_idx = dataset.class_to_idx['fake']
        for i in range(len(dataset.targets)):
            if dataset.targets[i] == real_idx:
                dataset.targets[i] = 0
            elif dataset.targets[i] == fake_idx:
                dataset.targets[i] = 1
    else:
        raise ValueError("Dataset must have 'real' and 'fake' subfolders.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix.tolist()
    }

    return metrics
