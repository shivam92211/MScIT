from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Define the path to the model weights file
model_path = "yolov5su.pt"

# Check if the model weights file exists locally
if os.path.exists(model_path):
    model = YOLO(model_path)  # Load the local model if it exists
else:
    model = YOLO("yolov5s.pt")  # 'yolov5s' is a lightweight model that will be downloaded automatically
# Read the image
image_path = "dogs.jpeg"
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Display the results
annotated_image = results[0].plot()  # Annotated image with bounding boxes
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

output_path = "annotated_image.jpg"
cv2.imwrite(output_path, annotated_image)
print(f"Annotated image saved to {output_path}")
