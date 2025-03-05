import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Prepare and normalize data
data = pd.DataFrame(np.sin(np.linspace(0, 100, 1000)), columns=['value'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 2. Create dataset for LSTM
X, y = [], []
for i in range(len(scaled_data) - 10):
    X.append(scaled_data[i:i+10, 0])
    y.append(scaled_data[i+10, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 3. Split data into train and test
train_size = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# 4. Build and train model
model = Sequential([LSTM(50, input_shape=(X_train.shape[1], 1)), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)  # Increased epochs to 20

# 5. Predict and plot
predictions = scaler.inverse_transform(model.predict(X_test))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape y_test to 2D for inverse transform
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()