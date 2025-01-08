# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.datasets import load_diabetes

# Load the Diabetes dataset
diabetes = load_diabetes()

# Convert it into a pandas DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Add the target variable (diabetes progression) to the dataframe
df['PROGRESSION'] = diabetes.target

# Display the first few rows of the dataset
print("Dataset Overview:")
print(df.head(), "\n")

# Check for multicollinearity using the correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Check Variance Inflation Factor (VIF) to detect multicollinearity
X = df.drop(columns=['PROGRESSION'])
X_with_const = add_constant(X)  # Add constant to the features
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

print("\nVariance Inflation Factor (VIF):")
print(vif_data, "\n")

# Remove features with high VIF (>10), or use Lasso Regression to handle feature selection
X_selected = X.drop(columns=['s5', 's4'])  # Drop highly correlated features, based on VIF analysis

# Split the dataset into features (X) and target (y)
X = X_selected
y = df['PROGRESSION']

# Standardize the features (important for regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the model's coefficients and intercept
intercept = model.intercept_
coefficients = model.coef_

print(f"\nIntercept: {intercept}")
print(f"Coefficients: {coefficients}\n")

# Make predictions using the test set
y_pred = model.predict(X_test)

# Compare actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted Progression (for the first 5 records):")
print(comparison_df.head(), "\n")

# Calculate R-squared (R²) and Mean Squared Error (MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R-squared: {r2}")
print(f"Mean Squared Error (MSE): {mse}\n")

# Visualize the regression line (if it were 2D) or scatter plot of actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Progression')
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression')
plt.show()

# Regularization using Lasso (L1) Regression to perform feature selection
lasso = Lasso(alpha=0.1)  # Regularization strength (alpha)
lasso.fit(X_train, y_train)

# Display Lasso coefficients (some of them might be zero, meaning the feature is not important)
lasso_coefficients = lasso.coef_
print(f"Lasso Coefficients: {lasso_coefficients}\n")

# Make predictions using Lasso model
y_pred_lasso = lasso.predict(X_test)

# Calculate R-squared and MSE for Lasso model
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f"Lasso R-squared: {r2_lasso}")
print(f"Lasso Mean Squared Error (MSE): {mse_lasso}\n")

# Plot Lasso regression predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Progression (Lasso Regression)')
plt.xlabel('Actual Progression')
plt.ylabel('Predicted Progression (Lasso)')
plt.show()