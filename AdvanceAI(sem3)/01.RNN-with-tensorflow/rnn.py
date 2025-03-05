import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate a sequence of natural numbers
sequence = np.array([i for i in range(100)]) 

# Prepare input (X) and output (y) for the RNN
X = sequence[:-1].reshape(-1, 1, 1)  # Reshape to [samples, time steps, features]
y = sequence[1:]  # Next number in the sequence

# Define a more complex RNN model
model = Sequential([
    SimpleRNN(20, return_sequences=True, input_shape=(1, 1)),  # First RNN layer with 20 units
    SimpleRNN(20),  # Second RNN layer with 20 units
    Dense(10, activation='relu'),  # Additional Dense layer with 10 units
    Dense(1)  # Output layer to predict the next number
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=200, verbose=0)

# Predict the next number in the sequence
test_input = np.array([9]).reshape(1, 1, 1)  # Predict the next number after 9
predicted = model.predict(test_input, verbose=0)

print(f"Input: {test_input.flatten()[0]}, Predicted Next Number: {predicted[0][0]}")