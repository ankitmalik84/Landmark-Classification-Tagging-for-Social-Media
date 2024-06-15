# Landmark Classification with Convolutional Neural Networks

Welcome to the Convolutional Neural Networks (CNN) project! In this project, we build a pipeline to process real-world, user-supplied images and create a model to predict the most likely locations where the image was taken. The final app will suggest the top 5 most relevant landmarks from 50 possible landmarks from across the world.

## Project Overview

Photo sharing and storage services often lack location metadata for uploaded images. This project addresses the challenge by detecting and classifying landmarks in the images using a CNN-powered app.

### Key Objectives

- Build a CNN from scratch for landmark classification.
- Experiment with different architectures, hyperparameters, and training strategies.
- Implement transfer learning using pre-trained models.
- Develop a simple app to predict landmarks from user-uploaded images.

## Data Preprocessing

### Procedure

1. **Resize Images:**

   - First resize images to 256x256 pixels.
   - Then crop images to 224x224 pixels.
   - Chosen 224x224 as it is the recommended input size for using PyTorch's pre-trained models.

2. **Data Augmentation:**
   - Applied `RandAugment` to augment the dataset with translations, flips, and rotations.
   - Aimed to improve model robustness and test accuracy.

## CNN Architecture (From Scratch)

### Steps to Final Architecture

1. **Initial Setup:**

   - 7 convolutional blocks to ensure expressiveness.
   - Dropout layers to reduce overfitting.
   - Output a 50-dimensional vector to match the 50 landmark classes.

2. **Training Parameters:**
   - Batch size: 64
   - Validation size: 20%
   - Epochs: 50
   - Learning rate: 0.01
   - Optimizer: Adam
   - Weight decay: \(1 \times 10^{-8}\)

### Results

- **Test Loss:** 1.632997
- **Test Accuracy:** 60% (760/1250)

## Transfer Learning

### Procedure

1. **Model Choice:**

   - Used `ResNet152` for transfer learning due to its performance on ImageNet and suitability for natural scene images.

2. **Training Parameters:**
   - Batch size: 64
   - Validation size: 20%
   - Epochs: 50
   - Learning rate: 0.001
   - Optimizer: Adam
   - Weight decay: 0.0

### Results

- **Test Loss:** 0.807198
- **Test Accuracy:** 79% (998/1250)

## Simple App for Landmark Classification

### Functionality

- Upload an image.
- The app predicts and displays the top 5 landmarks with their probabilities.

### Code Overview

```python
from ipywidgets import VBox, Button, FileUpload, Output, Label
from PIL import Image
from IPython.display import display
import io
import numpy as np
import torchvision.transforms as T
import torch

# Load the exported model
learn_inf = torch.jit.load("checkpoints/transfer_exported.pt")

def on_click_classify(change):
    fn = io.BytesIO(btn_upload.data[-1])
    img = Image.open(fn)
    img.load()
    out_pl.clear_output()
    with out_pl:
        ratio = img.size[0] / img.size[1]
        c = img.copy()
        c.thumbnail([ratio * 200, 200])
        display(c)
    timg = T.ToTensor()(img).unsqueeze_(0)
    softmax = learn_inf(timg).data.cpu().numpy().squeeze()
    idxs = np.argsort(softmax)[::-1]
    for i in range(5):
        p = softmax[idxs[i]]
        landmark_name = learn_inf.class_names[idxs[i]]
        labels[i].value = f"{landmark_name} (prob: {p:.2f})"

btn_upload = FileUpload()
btn_run = Button(description="Classify")
btn_run.on_click(on_click_classify)
labels = [Label() for _ in range(5)]
out_pl = Output()
out_pl.clear_output()
wgs = [Label("Please upload a picture of a landmark"), btn_upload, btn_run, out_pl] + labels
VBox(wgs)
```

## Model Code

### CNN from Scratch

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Head(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_classes: int, p: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=p),
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Linear(out_channels, n_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

class MyModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        channels = [3,] + [2**(4+i) for i in range(7)]
        self.model = nn.Sequential()
        for i in range(7):
            self.model.add_module(f"ConvBlock_{i}", ConvBlock(channels[i], channels[i+1]))
        self.model.add_module("Head", Head(channels[-1], channels[-2], num_classes, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

### Transfer Learning

```python
import torch
import torchvision.models as models

def get_model_transfer_learning(model_name: str):
    if model_name == "resnet152":
        model = models.resnet152(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 50)
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model
```

## Usage

1. Clone the repository.
2. Install the necessary dependencies.
3. Run the Jupyter notebooks to train the models and launch the app.

```bash
git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
jupyter notebook
```

## Conclusion

This project demonstrates the process of building a CNN for landmark classification, utilizing transfer learning for improved accuracy, and developing a simple app to classify landmarks from user-uploaded images. The transfer learning model achieved a test accuracy of 79%, indicating its suitability for the task.

This project provides me solid foundation for further development and application of CNNs in real-world scenarios.
