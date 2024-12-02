import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load and Prepare the IMDb Dataset
max_features = 10000  # Use the top 10,000 most frequent words
maxlen = 100  # Limit each review to 100 words

# Load the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure all inputs are the same length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Step 2: Define the RNN Model
model = Sequential([
    Embedding(max_features, 32, input_length=maxlen),  # Embedding layer
    SimpleRNN(32, activation='relu'),                 # RNN layer
    Dense(1, activation='sigmoid')                    # Output layer
])

# Step 3: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
