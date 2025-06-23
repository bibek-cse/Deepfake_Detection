# Deepfake Detection Models Toolkit

This project provides a Python-based toolkit for training, evaluating, and analyzing deepfake detection models. It separates concerns into dedicated modules for Data Handling, Model Implementations, and shared Utilities.

The `main.py` script acts as the central command-line interface, allowing you to select a model and dataset path to run a full pipeline. The pipeline typically involves:
1.  Loading and splitting your dataset into training, validation, and test sets using a specified dataset loader module.
2.  Setting up the selected model (potentially loading pre-trained weights).
3.  Training the model on the training data.
4.  Evaluating the model on the validation set during training (if validation data is available) and saving the best model weights.
5.  Loading the best (or last epoch's) model.
6.  Performing comprehensive evaluation on the test set.
7.  Displaying detailed metrics and generating visualizations (Confusion Matrix, ROC/PR Curves, Grad-CAM, t-SNE) using utility functions.

---

## Detection Models

The following table summarizes the publicly available models, their descriptions, pre-trained weight links, and citations to the original research papers:

|    Model Name    |                                              Description                                                   |        Model Weights           |      Research Paper         |
|------------------|------------------------------------------------------------------------------------------------------------|--------------------------------|-----------------------------|
| **XceptionNet**  | A deep CNN using depthwise separable convolutions for efficient and accurate deepfake detection.           |  [Download](link_to_weights)   |  [Paper](link_to_paper)     |
| **InceptionNet** | Uses inception modules to capture multi-scale features for detecting deepfake manipulations.               |  [Download](link_to_weights)   |  [Paper](link_to_paper)     |
| **ResNet50**     | A 50-layer residual network with skip connections for learning complex deepfake patterns.                  |  [Download](link_to_weights)   |  [Paper](link_to_paper)     |
| **EfficientNet** | Scales efficiently in depth, width, and resolution for high-accuracy deepfake detection.                   |  [Download](link_to_weights)   |  [Paper](link_to_paper)     |
| **MesoNet**      | Focuses on mesoscopic features specifically for distinguishing real and fake images.                       |  [Download](link_to_weights)   |  [Paper](link_to_paper)     |
| **ConvxNet**     | A custom convolutional network tailored for deepfake detection in this project.                            |  [Download](link_to_weights)   |  [Paper](link_to_paper)     |

> **Note:** Download Models weight path from `Model Weights Link` and `Paper Link` For Offocial paper citation.

---

## Usage

### 1. Prerequisites

- Python 3.7+
- Install dependencies:
```
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
Organize your dataset directory with subfolders for each class:
```
dataset/
├── real/
│   ├── image1.jpg
│   └── ...
└── fake/
    ├── image1.jpg
    └── ...
```
### 3. Run the Main Script
```
python main.py
```
* Choose a model while testing.
* Enter the path to your dataset.

---

## Adding New Models
To include a **new model** in this project:

Create a new Python file under the Detection/ directory (e.g., NewModel.py).

**Update the model_map dictionary in main.py:**
```
model_map = {
    "1": ("XceptionNet", "Detection.XceptionNet"),
    "2": ("NewModel", "Detection.NewModel"),
    ...
}
```

---

## Project Structure 
```
deepfake-detection-toolkit/
├── Detection/
│   ├── __init__.py             
│   ├── XceptionNet.py          
│   ├── InceptionNet.py       
│   ├── ResNet50.py           
│   ├── EfficientNet.py        
│   ├── MesoNet.py             
│   ├── ConvxNet.py            
│   ├── InceptionResNetV2.py   
├── Dataset/
│   ├── __init__.py             # To make Dataset a Python package
│   ├── base_dataset.py         # Base classes/helpers for datasets (Optional but good)
│   ├── Celeb_DF_v2.py          # Celeb-DF v2 specific loading/splitting
│   ├── Deepfake_Dataset.py     
├── Utility/
│   ├── __init__.py             # To make Utility a Python package
│   ├── plots.py                # Functions for generating plots (Conf matrix, ROC/PR, t-SNE)
│   ├── gradcam_helper.py       # Grad-CAM class and visualization function
│   ├── metrics.py              # Helpers Function for metrics if needed
├── main.py
├── requirements.txt
└── README.md
```

---

## Datasets

The following standard datasets can be used for training and testing with deepfake detection models. While download scripts might be provided, manual steps like video frame extraction and organizing the data into real/fake folders are typically required.

| Dataset Name       | Description                                                  | Loader Script                 |
|--------------------|--------------------------------------------------------------|-------------------------------|
| Celeb_DF_v2        | High-quality celebrity deepfakes, widely used in research.   | `Dataset/Celeb_DF_v2.py`      |
| Deepfake_Dataset   | Real and fake face images for binary classification tasks.   | `Dataset/Deepfake_Dataset.py` |

To download datasets automatically:

```bash
# Celeb_DF_v2
python -c "from Dataset.Celeb_DF_v2 import download_celeb_df_v2; download_celeb_df_v2()"

# Deepfake_Dataset
python -c "from Dataset.Deepfake_Dataset import download_deepfake_dataset; download_deepfake_dataset()"
```

---

## Running the Toolkit

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone <repository_url> # Replace with your repo URL
cd deepfake-detection-toolkit
pip install -r requirements.txt
```
### 2. Execute main.py
```bash
python main.py --model InceptionResNetV2 --dataset /path/to/your/prepared_dataset/
```
Replace /path/to/your/prepared_dataset/ with the actual path to your dataset folder containing real/ and fake/.

Specifying a different dataset loader (if you created one or want to force a specific one):
```
python main.py --model InceptionResNetV2 --dataset /path/to/my_other_dataset/ --dataset-loader Deepfake_Dataset
```
This would run InceptionResNetV2 but use the load_and_split_data function from Dataset/Deepfake_Dataset.py.

---

## Output and Visualizations

After running a model pipeline (e.g., for InceptionResNetV2), the script will print summaries of data loading, training progress (if training), validation metrics (if validation data exists), and final test metrics.

It will then use the utility functions to generate plots in separate windows:
* Confusion Matrix
* ROC Curve and PR Curve
* Grad-CAM visualization for a random test image
* t-SNE visualization of test set features

Inspect these plots to understand the model's performance and behavior. The confusion matrix and ROC/PR curves provide quantitative insights, while Grad-CAM and t-SNE offer qualitative understanding of where the model looks and how well it separates classes in its feature space.

---

### Acknowledgements
We thank the authors of the respective research papers for their contributions to the field of deepfake detection.
