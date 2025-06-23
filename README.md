# Deepfake Detection Models

This project provides a collection of deepfake detection models implemented in Python. It allows users to easily evaluate their datasets against various models to detect deepfake images or videos.

---

## Detection Models

The following table summarizes the publicly available models, their descriptions, pre-trained weight links, and citations to the original research papers:

|    Model Name    |                                              Description                                                   |          Model Weights                 |         Research Paper          |
|------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------|---------------------------------|
| **XceptionNet**  | A deep CNN using depthwise separable convolutions for efficient and accurate deepfake detection.           |    [Download](link_to_weights)         |      [Paper](link_to_paper)     |
| **InceptionNet** | Uses inception modules to capture multi-scale features for detecting deepfake manipulations.               |    [Download](link_to_weights)         |      [Paper](link_to_paper)     |
| **ResNet50**     | A 50-layer residual network with skip connections for learning complex deepfake patterns.                  |    [Download](link_to_weights)         |      [Paper](link_to_paper)     |
| **EfficientNet** | Scales efficiently in depth, width, and resolution for high-accuracy deepfake detection.                   |    [Download](link_to_weights)         |      [Paper](link_to_paper)     |
| **MesoNet**      | Focuses on mesoscopic features specifically for distinguishing real and fake images.                       |    [Download](link_to_weights)         |      [Paper](link_to_paper)     |
| **ConvxNet**     | A custom convolutional network tailored for deepfake detection in this project.                            |    [Download](link_to_weights)         |      [Paper](link_to_paper)     |

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

## Project Structure 
```
├── Detection/
│   ├── XceptionNet.py
│   ├── InceptionNet.py
│   ├── ...
├── main.py
├── requirements.txt
└── README.md
```

### Acknowledgements
We thank the authors of the respective research papers for their contributions to the field of deepfake detection.
