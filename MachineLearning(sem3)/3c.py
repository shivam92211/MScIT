# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
# Step 1: Generate a synthetic dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Apply Ridge, Lasso, and ElasticNet regression models
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio = 0.5 is a good start for mixing L1 and L2

# Train the models
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)
# Step 3: Predictions
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
y_pred_elastic_net = elastic_net.predict(X_test)
# Step 4: Evaluate the models using MSE and R² score
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_elastic_net = mean_squared_error(y_test, y_pred_elastic_net)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_lasso = r2_score(y_test, y_pred_lasso)
r2_elastic_net = r2_score(y_test, y_pred_elastic_net)

# Print the results
print(f"Ridge Regression MSE: {mse_ridge:.4f}, R²: {r2_ridge:.4f}")
print(f"Lasso Regression MSE: {mse_lasso:.4f}, R²: {r2_lasso:.4f}")
print(f"ElasticNet Regression MSE: {mse_elastic_net:.4f}, R²: {r2_elastic_net:.4f}")

# Step 5: Plot the results (optional, just for visualization)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True values', linestyle='-', marker='o', markersize=5)
plt.plot(y_pred_ridge, label='Ridge predictions', linestyle='-', marker='x', markersize=5)
plt.plot(y_pred_lasso, label='Lasso predictions', linestyle='-', marker='x', markersize=5)
plt.plot(y_pred_elastic_net, label='ElasticNet predictions', linestyle='-', marker='x', markersize=5)
plt.legend()
plt.title('Regression Model Predictions vs True Values')
plt.show()
