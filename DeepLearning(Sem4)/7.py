# Write a program for character recognition using RNN and compare it with CNN

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape for RNN (samples, timesteps, features)
x_train_rnn, x_test_rnn = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)
# Reshape for CNN (samples, height, width, channels)
x_train_cnn, x_test_cnn = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)

# RNN Model
def create_rnn():
    model = models.Sequential([
        layers.SimpleRNN(128, activation='relu', input_shape=(28, 28)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# CNN Model
def create_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate RNN
rnn_model = create_rnn()
rnn_model.fit(x_train_rnn, y_train, epochs=3, batch_size=64, validation_data=(x_test_rnn, y_test))
rnn_loss, rnn_acc = rnn_model.evaluate(x_test_rnn, y_test)

# Train and evaluate CNN
cnn_model = create_cnn()
cnn_model.fit(x_train_cnn, y_train, epochs=3, batch_size=64, validation_data=(x_test_cnn, y_test))
cnn_loss, cnn_acc = cnn_model.evaluate(x_test_cnn, y_test)

# Compare results
print(f'RNN Accuracy: {rnn_acc:.4f}, CNN Accuracy: {cnn_acc:.4f}')
