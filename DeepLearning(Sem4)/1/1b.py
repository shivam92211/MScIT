'''
Program to solve the XOR problem.
'''
import tensorflow as tf
import numpy as np

# Define XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=1000, verbose=0)  # Training for 500 epochs

# Evaluate the model
predictions = model.predict(X)
print("Predictions:", np.round(predictions).flatten())
