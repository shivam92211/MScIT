# Use Colab with GPU

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Parameters
IMG_SIZE = 224  # Resize MNIST images to 224x224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Define data preprocessing function
def preprocess(image, label):
    # Expand dims for grayscale, resize, normalize, and convert to RGB
    image = tf.image.resize(tf.expand_dims(image, axis=-1), (IMG_SIZE, IMG_SIZE)) / 255.0
    image = tf.image.grayscale_to_rgb(image)
    label = tf.one_hot(label, depth=10)  # One-hot encode labels
    return image, label

# Create TensorFlow datasets
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Dropout for regularization
x = Dense(128, activation="relu")(x)  # Add a dense layer
predictions = Dense(10, activation="softmax")(x)  # Output layer for 10 classes

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# Save the model
model.save("mnist_transfer_learning_model.h5")

# Evaluate the model on the test dataset
evaluation = model.evaluate(test_dataset, verbose=1)

# Print the evaluation metrics
print(f"Test Loss: {evaluation[0]:.4f}")
print(f"Test Accuracy: {evaluation[1]:.4f}")
