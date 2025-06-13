import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
# Ensure the CSV file is in the same directory as your script or provide the full path
df = pd.read_csv('2/Housing.csv')

# Display the first few rows of the dataframe
print(df.head())

# Select relevant features and target variable
# Assuming 'area' is the feature and 'price' is the target
X = df[['area']].values  # Feature: area of the house
y = df['price'].values   # Target: price of the house

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Define a simple linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
    ])
model.compile(optimizer='sgd', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=100, verbose=0, validation_data=(X_test_scaled, y_test_scaled))

# Visualize the loss function
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Over Time')
plt.legend()
plt.show()

# Plot the learned linear relationship
plt.scatter(X_train_scaled, y_train_scaled, label='Training Data')
plt.scatter(X_test_scaled, y_test_scaled, label='Testing Data')
plt.plot(X_train_scaled, model.predict(X_train_scaled), 'r', label='Model Prediction')
plt.xlabel('Normalized Area')
plt.ylabel('Normalized Price')
plt.legend()
plt.title('Linear Regression Fit')
plt.show()

# Make a prediction on new data
new_area = np.array([[3000]])  # Example: 3000 sq ft
new_area_scaled = scaler_X.transform(new_area)
predicted_price_scaled = model.predict(new_area_scaled)
predicted_price = scaler_y.inverse_transform(predicted_price_scaled)
print(f'Predicted price for {new_area[0][0]} sq ft: ${predicted_price[0][0]:.2f}')