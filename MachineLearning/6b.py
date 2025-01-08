import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
# Step 1: Generate Synthetic Data (Multiple Gaussian Distributions)
np.random.seed(42)

# Create synthetic data: 3 Gaussian distributions
n_samples = 500
X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.60, random_state=42)

# Step 2: Fit a Gaussian Mixture Model (GMM) to the data
# Fit a GMM with 3 components (since we know there are 3 clusters)
gmm = GaussianMixture(n_components=3)
gmm.fit(X)

# Step 3: Visualize the data points and the GMM
# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c='black', s=40, label="Data")
# Plot the GMM components (mean of each Gaussian)
means = gmm.means_
covariances = gmm.covariances_
# Plot the ellipses for each Gaussian component (representing the covariance)
for mean, cov in zip(means, covariances):
    v, w = np.linalg.eigh(cov)
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi
    ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], angle=angle, color='red', alpha=0.4)
    plt.gca().add_patch(ell)
plt.title('Gaussian Mixture Model Components (Density Estimation)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 4: Use the GMM for clustering (predict the cluster for each point)
labels = gmm.predict(X)

# Step 5: Visualize the clustering result
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', label="Clustered Data")
plt.title('Clustering Results from GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 6: Density estimation - Plot the GMM's predicted density on a grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_data = np.column_stack([xx.ravel(), yy.ravel()])
# Get the log likelihood (density) of the data points under the GMM
Z = np.exp(gmm.score_samples(grid_data))
Z = Z.reshape(xx.shape)
# Plot the density
plt.contourf(xx, yy, Z, levels=10, cmap='Blues')
plt.scatter(X[:, 0], X[:, 1], c='black', s=40, alpha=0.5, label="Data")
plt.title('Density Estimation with GMM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()