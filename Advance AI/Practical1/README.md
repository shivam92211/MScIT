Here’s a **README.md** file explaining the two programs: 

1. **Simple RNN for Sequence Prediction**  
2. **RNN for Sentiment Analysis on IMDb Dataset**

---

## README.md

# RNN Examples in TensorFlow

This repository contains two simple examples of using Recurrent Neural Networks (RNNs) with TensorFlow:

1. A simple program that predicts the next number in a sequence.  
2. A sentiment analysis model using the IMDb movie reviews dataset.

---

## 1. Simple RNN for Sequence Prediction

This program trains an RNN to predict the next number in a sequence. For example, given `[1, 2, 3, 4]`, the model predicts `5`.

### Functions and Components

- **Import Libraries**  
  ```python
  import tensorflow as tf
  import numpy as np
  ```
  - `tensorflow`: The deep learning framework used to build and train the RNN.  
  - `numpy`: Used for handling data arrays.

- **Prepare the Data**  
  ```python
  sequence = np.array([1, 2, 3, 4, 5], dtype=np.float32)
  x = sequence[:-1].reshape(1, -1, 1)
  y = sequence[1:].reshape(1, -1, 1)
  ```
  - A simple sequence `[1, 2, 3, 4, 5]` is prepared.  
  - `x` is the input `[1, 2, 3, 4]`.  
  - `y` is the target `[2, 3, 4, 5]`.  
  - The data is reshaped into a format compatible with RNNs (`batch_size, timesteps, features`).

- **Define the Model**  
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.SimpleRNN(10, activation='relu', input_shape=(None, 1)),
      tf.keras.layers.Dense(1)
  ])
  ```
  - `SimpleRNN`: A basic recurrent layer with 10 units and ReLU activation.  
  - `Dense`: A fully connected layer with one output to predict the next number.

- **Compile the Model**  
  ```python
  model.compile(optimizer='adam', loss='mse')
  ```
  - `adam`: Optimizer for efficient gradient descent.  
  - `mse`: Mean Squared Error, a suitable loss function for regression problems.

- **Train the Model**  
  ```python
  model.fit(x, y, epochs=200, verbose=0)
  ```
  - The model is trained for 200 epochs.  

- **Make a Prediction**  
  ```python
  test_input = np.array([2, 3, 4, 5], dtype=np.float32).reshape(1, -1, 1)
  predicted = model.predict(test_input)
  ```
  - Given a new sequence, the model predicts the next number.

---

## 2. RNN for Sentiment Analysis on IMDb Dataset

This program builds a sentiment analysis model using an RNN on the IMDb movie reviews dataset.

### Functions and Components

- **Import Libraries**  
  ```python
  import tensorflow as tf
  from tensorflow.keras import Sequential
  from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
  from tensorflow.keras.datasets import imdb
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  ```
  - TensorFlow and Keras modules are used for building and training the model.  
  - `imdb`: Provides the dataset of 50,000 labeled movie reviews.  
  - `pad_sequences`: Ensures all input sequences have the same length.

- **Load and Prepare the Dataset**  
  ```python
  max_features = 10000
  maxlen = 100
  (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
  x_train = pad_sequences(x_train, maxlen=maxlen)
  x_test = pad_sequences(x_test, maxlen=maxlen)
  ```
  - `max_features`: Use the 10,000 most frequent words.  
  - `maxlen`: Limit each review to 100 words.  
  - Reviews are tokenized into integers and padded to have the same length.

- **Define the Model**  
  ```python
  model = Sequential([
      Embedding(max_features, 32, input_length=maxlen),
      SimpleRNN(32, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  ```
  - `Embedding`: Maps words to dense 32-dimensional vectors (word embeddings).  
  - `SimpleRNN`: A recurrent layer with 32 units and ReLU activation.  
  - `Dense`: Outputs a single probability value (0 = negative, 1 = positive).  
  - `sigmoid`: Used for binary classification.

- **Compile the Model**  
  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```
  - `adam`: Optimizer for efficient gradient descent.  
  - `binary_crossentropy`: Loss function for binary classification.  
  - `accuracy`: Metric to evaluate model performance.

- **Train the Model**  
  ```python
  model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
  ```
  - Trains the model for 5 epochs, using a batch size of 64.  
  - Splits 20% of the training data for validation.

- **Evaluate the Model**  
  ```python
  test_loss, test_acc = model.evaluate(x_test, y_test)
  print(f"Test Accuracy: {test_acc:.2f}")
  ```
  - Evaluates the model on the test data and prints the accuracy.

---

## How to Run

1. Install TensorFlow:
   ```bash
   pip install tensorflow
   ```

2. Save each program in separate files (e.g., `simple_rnn.py` and `imdb_rnn.py`).

3. Run the programs:
   ```bash
   python simple_rnn.py
   python imdb_rnn.py
   ```

---

## Outputs

### Simple RNN for Sequence Prediction
- The program predicts the next number in a given sequence.

### IMDb Sentiment Analysis
- The program outputs test accuracy, e.g.,:
  ```
  Test Accuracy: 0.84
  ```

---

## Notes

- **Extensions**:
  - Replace `SimpleRNN` with `LSTM` or `GRU` for better performance in both examples.
  - For IMDb, experiment with different `max_features` and `maxlen` values.

- **References**:
  - IMDb Dataset: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/imdb_reviews)  
  - TensorFlow Documentation: [tensorflow.org](https://www.tensorflow.org/)
