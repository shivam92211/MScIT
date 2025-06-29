import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A  # Added Albumentations import

# 1. Define the U-Net Model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

# 2. Custom Dataset for Loading Images and Masks
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # Normalize mask to [0, 1]

        if self.transform:
            # Apply Albumentations transforms to both image and mask
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# 3. Training Function
def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)  # Add channel dimension
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# 4. Evaluation Function
def evaluate_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# 5. Save Model Function
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# 6. Load Model Function
def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")

# 7. Prediction Function
def predict_image(model, image_path, transform, device, threshold=0.5):
    model.eval()
    image = np.array(Image.open(image_path).convert("RGB"))
    original_shape = image.shape[:2]  # Save original shape for resizing later

    if transform:
        # Apply Albumentations transforms (only image needed here)
        augmented = transform(image=image)
        image = augmented['image']

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    # Convert output to binary mask
    output = (output.squeeze().cpu().numpy() > threshold).astype(np.uint8)
    # Resize mask back to original image size
    output = np.array(Image.fromarray(output).resize(original_shape[::-1], Image.NEAREST))
    return output

# 8. Main Script
if __name__ == "__main__":
    # Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NUM_EPOCHS = 1
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PIN_MEMORY = True
    TRAIN_VAL_SPLIT = 0.8  # 80% for training, 20% for validation
    MODEL_SAVE_PATH = "unet_model.pth"

    # Directories
    IMAGES_DIR = "data/sub_images/"
    MASKS_DIR = "data/sub_masks/"

    # Albumentations Transform (Corrected)
    transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

    # Load all image and mask paths
    image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith(('.jpg', '.png'))])

    # Ensure filenames match
    image_names = [os.path.splitext(f)[0] for f in image_files]
    mask_names = [os.path.splitext(f)[0] for f in mask_files]
    assert set(image_names) == set(mask_names), "Mismatch between image and mask filenames"

    # Combine full paths
    image_paths = [os.path.join(IMAGES_DIR, f) for f in image_files]
    mask_paths = [os.path.join(MASKS_DIR, f) for f in mask_files]

    # Shuffle and split into train/validation
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    split_index = int(len(combined) * TRAIN_VAL_SPLIT)
    train_data, val_data = combined[:split_index], combined[split_index:]
    train_image_paths, train_mask_paths = zip(*train_data)
    val_image_paths, val_mask_paths = zip(*val_data)

    # Create datasets and dataloaders
    train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform=transform)
    val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    # Initialize Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss = evaluate_fn(val_loader, model, loss_fn, DEVICE)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the trained model
    save_model(model, MODEL_SAVE_PATH)

    # Load the model for prediction
    loaded_model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    load_model(loaded_model, MODEL_SAVE_PATH, DEVICE)

    # Perform prediction on a new image
    test_image_path = "my_car.jpeg"  # Replace with your test image path
    predicted_mask = predict_image(loaded_model, test_image_path, transform, DEVICE)

    # Save the predicted mask as an image
    predicted_mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
    predicted_mask_image.save("predicted_mask.png")
    print("Prediction complete! Predicted mask saved as 'predicted_mask.png'")