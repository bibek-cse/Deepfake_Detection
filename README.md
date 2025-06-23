# Deepfake Detection Models

This project provides a collection of deepfake detection models implemented in Python. It allows users to easily evaluate their datasets against various models to detect deepfake images or videos.

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
├── Detection/
│   ├── XceptionNet.py
│   ├── InceptionNet.py
│   ├── ...
├── Dataset/
|   ├── __init__.py
|   ├── Celeb_DF_v2.py
|   ├── Deepfake_Dataset.py
|   ├── utils.py            
├── main.py
├── requirements.txt
└── README.md
```

---

## Datasets

The following standard datasets can be used for training and testing:

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

### Acknowledgements
We thank the authors of the respective research papers for their contributions to the field of deepfake detection.
