# with chatgpt prompt act like
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Create a simple dataset (1D Gaussian distribution)
def generate_real_data(n):
    return np.random.normal(loc=0.0, scale=1.0, size=(n, 1))

# Step 2: Build the Generator Network
def build_generator():
    model = Sequential([
        Dense(16, activation='relu', input_dim=1),
        Dense(1, activation='linear')
    ])
    return model

# Step 3: Build the Discriminator Network
def build_discriminator():
    model = Sequential([
        Dense(16, activation='relu', input_dim=1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Combine Generator and Discriminator into a GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator
    model = Sequential([generator, discriminator])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Step 5: Training Loop
def train_gan(generator, discriminator, gan, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Generate real and fake data
        real_data = generate_real_data(batch_size)
        fake_data = generator.predict(np.random.normal(0, 1, (batch_size, 1)))

        # Train the discriminator
        x_combined = np.vstack((real_data, fake_data))
        y_combined = np.vstack((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))
        discriminator.train_on_batch(x_combined, y_combined)

        # Train the generator (via GAN model)
        noise = np.random.normal(0, 1, (batch_size, 1))
        gan.train_on_batch(noise, np.ones((batch_size, 1)))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Discriminator Loss = {discriminator.evaluate(x_combined, y_combined, verbose=0)[0]:.4f}")

# Step 6: Set up and train the GAN
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
train_gan(generator, discriminator, gan)