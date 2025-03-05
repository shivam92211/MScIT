import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate real data (simple 2D Gaussian distribution)
def generate_real_data(n_samples):
    mean = [2, 3]
    cov = [[1, 0.5], [0.5, 1]]
    real_data = np.random.multivariate_normal(mean, cov, n_samples)
    return torch.FloatTensor(real_data)

# Generator Network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training parameters
n_samples = 1000
input_dim = 10
output_dim = 2
n_epochs = 2000
batch_size = 32

# Initialize networks and optimizers
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

criterion = nn.BCELoss()

# Training loop
for epoch in range(n_epochs):
    # Train Discriminator
    real_data = generate_real_data(batch_size)
    noise = torch.randn(batch_size, input_dim)
    fake_data = generator(noise).detach()
    
    d_optimizer.zero_grad()
    
    # Real data
    real_labels = torch.ones(batch_size, 1)
    real_output = discriminator(real_data)
    d_loss_real = criterion(real_output, real_labels)
    
    # Fake data
    fake_labels = torch.zeros(batch_size, 1)
    fake_output = discriminator(fake_data)
    d_loss_fake = criterion(fake_output, fake_labels)
    
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()
    
    # Train Generator
    noise = torch.randn(batch_size, input_dim)
    fake_data = generator(noise)
    
    g_optimizer.zero_grad()
    
    output = discriminator(fake_data)
    g_loss = criterion(output, real_labels)
    
    g_loss.backward()
    g_optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Visualize results
def plot_distributions():
    real_data = generate_real_data(500)
    noise = torch.randn(500, input_dim)
    fake_data = generator(noise).detach()
    
    plt.figure(figsize=(10, 5))
    
    # Plot real data
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.5, label='Real Data')
    plt.title('Real Data Distribution')
    plt.legend()
    
    # Plot generated data
    plt.subplot(1, 2, 2)
    plt.scatter(fake_data[:, 0], fake_data[:, 1], c='red', alpha=0.5, label='Generated Data')
    plt.title('Generated Data Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Plot the results
plot_distributions()