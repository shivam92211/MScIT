# finalized practical 9
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Step 1: Create a simple dataset (1D Gaussian distribution)
def generate_real_data(n_samples):
    X = np.random.normal(loc=0.0, scale=1.0, size=n_samples)  # Mean=0, Stddev=1
    y = np.ones((n_samples, 1))  # Labels for real data
    return X, y

# Step 2: Define the Generator
def build_generator(latent_dim):
    model = Sequential([
        Dense(16, activation='relu', input_dim=latent_dim),
        Dense(1, activation='linear')  # Output is a single number (1D point)
    ])
    return model

# Step 3: Define the Discriminator
def build_discriminator():
    model = Sequential([
        Dense(16, activation='relu', input_dim=1),
        Dense(1, activation='sigmoid')  # Output is a probability
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator during GAN training
    model = Sequential([generator, discriminator])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Step 5: Train the GAN
def train_gan(generator, discriminator, gan, latent_dim, n_epochs=10000, n_batch=128, n_eval=1000):
    for epoch in range(n_epochs):
        # Train the Discriminator
        # 1. Real data
        X_real, y_real = generate_real_data(n_batch // 2)
        d_loss_real = discriminator.train_on_batch(X_real, y_real)

        # 2. Fake data
        X_fake = generator.predict(np.random.randn(n_batch // 2, latent_dim))
        y_fake = np.zeros((n_batch // 2, 1))  # Labels for fake data
        d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)

        # Train the Generator (via the GAN model)
        z_input = np.random.randn(n_batch, latent_dim)  # Random noise
        y_gan = np.ones((n_batch, 1))  # Labels for GAN training are always 1
        g_loss = gan.train_on_batch(z_input, y_gan)

        # Evaluate progress
        if (epoch + 1) % n_eval == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, D Loss Real: {d_loss_real[0]:.3f}, D Loss Fake: {d_loss_fake[0]:.3f}, G Loss: {g_loss:.3f}")
            summarize_performance(epoch, generator, latent_dim)

# Step 6: Evaluate and Plot Results
def summarize_performance(epoch, generator, latent_dim, n_samples=100):
    X_fake = generator.predict(np.random.randn(n_samples, latent_dim))
    plt.hist(X_fake, bins=30, alpha=0.7, label='Generated')
    plt.hist(np.random.normal(0.0, 1.0, n_samples), bins=30, alpha=0.7, label='Real')
    plt.title(f'Epoch {epoch+1}')
    plt.legend()
    plt.show()

# Parameters
latent_dim = 5  # Size of random noise vector
n_epochs = 1000
n_batch = 32

# Build and compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, latent_dim, n_epochs, n_batch, n_eval=1000)