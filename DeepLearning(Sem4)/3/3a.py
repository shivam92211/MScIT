# Implementing deep neural network for performing binary classification task

import numpy as np
import tensorflow as tf 
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer dataset (binary classification: malignant vs. benign)
data = sklearn.datasets.load_breast_cancer()
X, y = data.data, data.target

# Standardize features to improve training performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Build a deep neural network with two hidden layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # First hidden layer with 16 neurons
    tf.keras.layers.Dense(8, activation='relu'),                                     # Second hidden layer with 8 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')                                   # Output layer with sigmoid for binary classification
])

# Compile the model with the Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for 50 epochs with a batch size of 16
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model on the test set and print the accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy: {:.2f}%".format(accuracy * 100))

# Predict the class for a single test sample
sample = X_test[0:1]
pred_prob = model.predict(sample)
pred_class = (pred_prob > 0.5).astype("int32")  # Convert probability to class label (0 or 1)
print("Predicted class:", pred_class[0][0], "Actual class:", y_test[0])
