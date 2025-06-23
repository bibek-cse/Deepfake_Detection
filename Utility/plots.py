# Utility/plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
import sys

def plot_confusion_matrix(true_labels, binary_preds, class_names=['Real', 'Fake'], title='Confusion Matrix'):
    """Plots a confusion matrix."""
    try:
        cm = confusion_matrix(true_labels, binary_preds)
        if cm.shape != (2, 2):
             print(f"Warning: Confusion matrix shape is {cm.shape}, expected (2, 2). Skipping plot.", file=sys.stderr)
             return

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Predicted {class_names[0]}', f'Predicted {class_names[1]}'],
                    yticklabels=[f'True {class_names[0]}', f'True {class_names[1]}'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.show()

        # Print FP/FN counts explicitly
        tn, fp, fn, tp = cm.ravel()
        print("\n--- Confusion Matrix Details ---")
        print(f"True Negatives (Actual {class_names[0]}, Predicted {class_names[0]}): {tn}")
        print(f"False Positives (Actual {class_names[0]}, Predicted {class_names[1]}): {fp}")
        print(f"False Negatives (Actual {class_names[1]}, Predicted {class_names[0]}): {fn}")
        print(f"True Positives (Actual {class_names[1]}, Predicted {class_names[1]}): {tp}")
        print("------------------------------")

    except Exception as e:
        print(f"Error plotting confusion matrix: {e}", file=sys.stderr)
        # traceback.print_exc() # Uncomment for detailed debugging


def plot_roc_pr_curves(true_labels, probabilities, title_suffix=''):
    """Plots ROC and Precision-Recall curves."""
    if len(np.unique(true_labels)) < 2 or len(probabilities) == 0:
        print("Skipping ROC/PR plots: Need at least 2 different labels and >0 samples.", file=sys.stderr)
        return

    try:
        # ROC Curve
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        print(f"\nROC AUC{title_suffix}: {roc_auc:.4f}")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(true_labels, probabilities)
        pr_auc = average_precision_score(true_labels, probabilities)
        print(f"PR AUC (Average Precision){title_suffix}: {pr_auc:.4f}")


        plt.figure(figsize=(12, 5))

        # ROC Plot
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random classifier baseline
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve{title_suffix}')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)

        # PR Plot
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve{title_suffix}')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error generating AUC/PR curves: {e}", file=sys.stderr)
        # traceback.print_exc()


def plot_tsne(features, labels, title='t-SNE Visualization', class_names=['Real', 'Fake']):
    """Plots t-SNE visualization of features."""
    if len(features) < 2 or len(np.unique(labels)) < 2:
        print("Skipping t-SNE plot: Need at least 2 samples and 2 different labels.", file=sys.stderr)
        return

    try:
        n_samples_for_tsne = len(features)
        # Adjust perplexity based on number of samples
        perplexity = min(30, n_samples_for_tsne - 1 if n_samples_for_tsne > 1 else 1)
        perplexity = max(5, perplexity) if n_samples_for_tsne > 5 else (n_samples_for_tsne - 1 if n_samples_for_tsne > 1 else 1)

        # TSNE requires perplexity > 1 and n_samples > perplexity
        min_samples_for_tsne_lib = max(3 * 2, perplexity + 1) # n_components=2
        if n_samples_for_tsne < min_samples_for_tsne_lib or perplexity <= 1:
             print(f"Skipping t-SNE: Too few samples ({n_samples_for_tsne}) or invalid perplexity ({perplexity}) for reliable T-SNE (requires at least {min_samples_for_tsne_lib} samples and perplexity > 1).", file=sys.stderr)
             return


        print(f"\nRunning t-SNE on {n_samples_for_tsne} samples with perplexity={perplexity}...")
        # Use PCA init if feature dimension is high
        init_method = 'pca' if features.shape[1] > 50 else 'random'
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init=init_method)
        tsne_results = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        # Use labels obtained during feature extraction
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        # Create a legend mapping colors to class names
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{class_names[0]} (0)',
                                      markerfacecolor=plt.cm.coolwarm(0.0), markersize=10),
                           plt.Line2D([0], [0], marker='o', color='w', label=f'{class_names[1]} (1)',
                                      markerfacecolor=plt.cm.coolwarm(1.0), markersize=10)]

        plt.legend(handles=legend_elements, loc="best")
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    except Exception as e:
        print(f"Error running or plotting t-SNE: {e}", file=sys.stderr)
        # traceback.print_exc()
