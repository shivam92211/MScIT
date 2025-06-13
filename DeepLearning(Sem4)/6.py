# Applying the Autoencoder algorithms for encoding real-world data

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
# from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# Load real-world dataset (MNIST handwritten digits)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0,1]
x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)  # Flatten

# Define Autoencoder architecture
input_dim = 28 * 28  # Image size (784 pixels)
encoding_dim = 32    # Compressed representation size

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Autoencoder model
autoencoder = tf.keras.models.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Encode and decode test images
encoded_imgs = autoencoder.predict(x_test)

# Display original vs. reconstructed images
n = 2
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original images
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    
    # Reconstructed images
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
