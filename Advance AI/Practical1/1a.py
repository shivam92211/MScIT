import tensorflow as tf
import numpy as np

# Step 1: Prepare the Data
# Input: [1, 2, 3, 4] -> Output: [5]
sequence = np.array([1, 2, 3, 4, 5], dtype=np.float32)
x = sequence[:-1].reshape(1, -1, 1)  # Shape (batch_size, timesteps, features)
y = sequence[1:].reshape(1, -1, 1)  # Target sequence (shifted by one)

# Step 2: Define the Model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(10, activation='relu', input_shape=(None, 1)),
    tf.keras.layers.Dense(1)
])

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='mse')

# Step 4: Train the Model
model.fit(x, y, epochs=200, verbose=0)

# Step 5: Make a Prediction
test_input = np.array([2, 3, 4, 5], dtype=np.float32).reshape(1, -1, 1)
predicted = model.predict(test_input)
print(f"Next number prediction: {predicted.flatten()[0]:.2f}")
