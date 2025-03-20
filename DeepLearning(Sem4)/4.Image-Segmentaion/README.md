# U-Net Image Segmentation

This project implements an image segmentation model using the U-Net architecture in PyTorch. The model is designed to segment objects in images based on a given dataset of images and masks.

## Features
- U-Net architecture for semantic segmentation
- Custom dataset handling with `torch.utils.data.Dataset`
- Data augmentation using Albumentations
- Training and evaluation with loss tracking
- Model saving and loading functionality
- Prediction on new images

## Requirements

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
torch
torchvision
numpy
pillow
albumentations
opencv-python
```

## Directory Structure
```
project_root/
│── sub_images/      # Directory containing input images
│── sub_masks/       # Directory containing corresponding masks
│── unet_model.pth   # Saved model weights (after training)
│── script.py        # Main training and evaluation script
│── predicted_mask.png # Output of the segmentation model
│── requirements.txt  # Dependencies
│── README.md        # This file
```

## How to Use

### 1. Train the Model
Run the following command to train the U-Net model:
```bash
python script.py
```

This will:
- Load the dataset (images & masks)
- Apply transformations (resizing, normalization)
- Train the model for the specified number of epochs
- Save the trained model to `unet_model.pth`

### 2. Load the Model for Prediction
The model can be loaded and used for prediction as follows:
```python
loaded_model = UNet(in_channels=3, out_channels=1).to(DEVICE)
load_model(loaded_model, "unet_model.pth", DEVICE)
```

### 3. Perform Prediction
To segment a new image:
```python
predicted_mask = predict_image(loaded_model, "my_car.jpg", transform, DEVICE)
```
The predicted mask will be saved as `predicted_mask.png`.

## Dataset
The dataset consists of images and their corresponding masks. Ensure:
- Image and mask filenames match.
- Masks have pixel values of 0 and 255, which will be normalized to [0, 1].

## Model Architecture
The U-Net model consists of:
- Encoder: Extracts features using convolutional layers.
- Bottleneck: Dense representation of features.
- Decoder: Upsamples to restore the segmented mask.
- Skip connections: Preserve spatial details.

## Hyperparameters
You can modify hyperparameters like:
```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 1
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## License
This project is open-source under the MIT License.

## Acknowledgements
- PyTorch
- Albumentations
- U-Net paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)

