import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Fetch historical stock price data using yfinance
ticker = 'GOOG'
df = yf.download(ticker, start='2010-01-01', end='2025-02-16')
data = df[['Close']].values  # Use closing prices

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split data into training and test sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Prepare training data
X_train, y_train = [], []
time_steps = 60  # Use last 60 days to predict next day
for i in range(time_steps, len(train_data)):
    X_train.append(train_data[i-time_steps:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # Reshape for LSTM

# Prepare test data
X_test, y_test = [], []
for i in range(time_steps, len(test_data)):
    X_test.append(test_data[i-time_steps:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build RNN model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50),
    Dense(units=1)
])

# Compile and train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict future stock prices
future_steps = 30
future_input = test_data[-time_steps:].reshape(1, time_steps, 1)
future_predictions = []
for _ in range(future_steps):
    pred = model.predict(future_input)
    future_predictions.append(pred[0, 0])
    future_input = np.append(future_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Visualization of future predictions
plt.figure(figsize=(14, 5))
future_dates = pd.date_range(df.index[-1], periods=future_steps + 1, freq='B')[1:]
plt.plot(future_dates, future_predictions, label='Future Predictions', color='green')
plt.legend()
plt.title("Future Stock Price Predictions")
plt.show()

print("Future Stock Price Prediction Completed!")