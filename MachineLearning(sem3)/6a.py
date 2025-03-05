import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
# Set random seed for reproducibility
np.random.seed(42)
# 1. Generate synthetic data (linear relationship with noise)
n_samples = 50
X = np.linspace(0, 10, n_samples)
y_true = 2 * X + 1  # y = 2x + 1 (true coefficients)
noise = np.random.normal(0, 1, n_samples)
y = y_true + noise  # observed data with noise
# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label="Observed data", color='blue')
plt.plot(X, y_true, label="True line", color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data and True Line')
plt.legend()
plt.show()
# 2. Define prior distributions for the parameters (weights)
# Assuming a Normal prior distribution for the weights (slope and intercept)
# We assume a zero-mean prior with a high variance (uninformative prior)
prior_mean = np.array([0, 0])  # [slope, intercept]
prior_cov = np.array([[10, 0], [0, 10]])  # Uncorrelated, high variance

# 3. Compute the posterior distribution using Bayesian formula
# We use the equation: p(w | X, y) ~ p(y | X, w) * p(w)
# p(y | X, w) is the likelihood: N(y | Xw, sigma^2)
# p(w) is the prior: N(w | prior_mean, prior_cov)
# Design matrix (with a column of ones for the intercept)
X_design = np.vstack([X, np.ones_like(X)]).T
# Likelihood covariance (assuming noise variance is known)
sigma_squared = 1  # Variance of the Gaussian noise
likelihood_cov = sigma_squared * np.identity(n_samples)
# Posterior mean and covariance (using the closed-form solution for Bayesian LR)
X_transpose = X_design.T
posterior_cov = np.linalg.inv(np.linalg.inv(prior_cov) + X_transpose @ np.linalg.inv(likelihood_cov) @ X_design)
posterior_mean = posterior_cov @ (np.linalg.inv(prior_cov) @ prior_mean + X_transpose @ np.linalg.inv(likelihood_cov) @ y)

# 4. Plot the prior and posterior distributions
# We will visualize the prior and posterior distributions of the slope and intercept
sns.set(style="whitegrid")

# Plot prior distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
slope_range = np.linspace(-5, 5, 100)
intercept_range = np.linspace(-5, 5, 100)
slope_grid, intercept_grid = np.meshgrid(slope_range, intercept_range)
prior_pdf = multivariate_normal.pdf(
    np.dstack([slope_grid, intercept_grid]), mean=prior_mean, cov=prior_cov)
plt.contour(slope_grid, intercept_grid, prior_pdf, levels=10, cmap='Blues')
plt.title('Prior Distribution')
plt.xlabel('Slope')
plt.ylabel('Intercept')
# Plot posterior distribution
plt.subplot(1, 2, 2)
posterior_pdf = multivariate_normal.pdf(
    np.dstack([slope_grid, intercept_grid]), mean=posterior_mean, cov=posterior_cov)
plt.contour(slope_grid, intercept_grid, posterior_pdf, levels=10, cmap='Reds')
plt.title('Posterior Distribution')
plt.xlabel('Slope')
plt.ylabel('Intercept')
plt.tight_layout()
plt.show()
# 5. Make predictions based on the posterior distribution
# Draw samples from the posterior distribution to make predictions
n_samples_posterior = 500
posterior_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, n_samples_posterior)
# Generate predictions using the posterior samples
predictions = np.array([X_design @ sample for sample in posterior_samples])
# Plot the data and the predictive distribution
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Observed data')
plt.plot(X, y_true, label='True Line', color='red', linewidth=2)
plt.plot(X, np.mean(predictions, axis=0), label='Posterior mean prediction', color='green', linewidth=2)
# Plot the uncertainty in predictions (shaded area)
std_dev = np.std(predictions, axis=0)
plt.fill_between(X, np.mean(predictions, axis=0) - 1.96 * std_dev, np.mean(predictions, axis=0) + 1.96 * std_dev, color='green', alpha=0.2, label='95% prediction interval')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Posterior Predictions with Uncertainty')
plt.legend()
plt.show()