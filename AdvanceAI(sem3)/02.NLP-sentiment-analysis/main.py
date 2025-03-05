import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample data: Text and corresponding labels (0 = negative, 1 = positive)
texts = [
    "I love this movie, it's fantastic!",
    "This film was terrible, I hated it.",
    "What a great experience, highly recommended.",
    "The worst movie I've ever seen.",
    "Absolutely wonderful, I enjoyed every moment."
]
labels = [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative

# Tokenize the text data
tokenizer = Tokenizer(num_words=1000)  # Limit vocabulary to the top 1000 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # Convert text to sequences of integers

# Pad sequences to ensure uniform input size
max_length = 10  # Maximum length of each sequence
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to a numpy array
labels = np.array(labels)

# Build the NLP model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=max_length),  # Embedding layer
    GlobalAveragePooling1D(),  # Pooling layer to reduce dimensionality
    Dense(16, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, verbose=1)

# Test the model with new text
test_texts = [
    "This was an amazing experience!",
    "I didn't like the movie at all."
]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Predict sentiment
predictions = model.predict(test_padded)

# Print predictions
for i, text in enumerate(test_texts):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {'Positive' if predictions[i] > 0.5 else 'Negative'}")
    print(f"Confidence: {predictions[i][0]:.4f}\n")