# Sentiment Analysis with TensorFlow/Keras

This project demonstrates a simple **Natural Language Processing (NLP)** model for **sentiment analysis** using TensorFlow and Keras. The model is trained on a small dataset of text samples and predicts whether a given text expresses positive or negative sentiment.

---

## **Code Overview**

The code performs the following steps:

1. **Data Preparation**:
   - A small dataset of text samples and corresponding labels (1 = positive, 0 = negative) is created.
   - The text data is tokenized and converted into sequences of integers.
   - Sequences are padded to ensure uniform input size.

2. **Model Architecture**:
   - The model consists of:
     - An **Embedding Layer** to convert integer-encoded words into dense vectors.
     - A **GlobalAveragePooling1D Layer** to reduce the sequence of word vectors into a single vector.
     - A **Dense Layer** with ReLU activation for hidden processing.
     - A **Dense Output Layer** with sigmoid activation for binary classification.

3. **Training**:
   - The model is compiled using the Adam optimizer and binary cross-entropy loss.
   - It is trained on the prepared dataset for 10 epochs.

4. **Prediction**:
   - The model predicts the sentiment of new text inputs and outputs the probability of the text being positive.

---

## **How to Run the Code**

1. Ensure you have the required libraries installed:
   ```bash
   pip install tensorflow numpy
   ```

2. Copy the code into a Python file (e.g., `sentiment_analysis.py`).

3. Run the script:
   ```bash
   python sentiment_analysis.py
   ```

4. The script will:
   - Train the model on the sample dataset.
   - Predict the sentiment of new text inputs and display the results.

---

## **Sample Output**

```
Text: This was an amazing experience!
Predicted Sentiment: Positive
Confidence: 0.9567

Text: I didn't like the movie at all.
Predicted Sentiment: Negative
Confidence: 0.1234
```

---

## **Viva Questions**

### 1. **What is the purpose of the `Tokenizer` in this code?**
   - **Answer**: The `Tokenizer` converts text data into sequences of integers. It maps each word in the vocabulary to a unique integer, which is necessary for feeding text data into the model.

### 2. **Why is padding used in this code?**
   - **Answer**: Padding ensures that all input sequences have the same length (`max_length`). This is required because neural networks expect fixed-size inputs.

### 3. **What is the role of the `Embedding` layer?**
   - **Answer**: The `Embedding` layer converts integer-encoded words into dense vectors of fixed size. It helps the model learn meaningful representations of words.

### 4. **Why is `GlobalAveragePooling1D` used in the model?**
   - **Answer**: `GlobalAveragePooling1D` reduces the sequence of word vectors into a single vector by averaging. This simplifies the output and reduces dimensionality.

### 5. **What does the `sigmoid` activation function in the output layer do?**
   - **Answer**: The `sigmoid` activation function outputs a value between 0 and 1, representing the probability of the input text being positive (1) or negative (0).

---

## **Dependencies**

- Python 3.x
- TensorFlow 2.x
- NumPy

---

## **License**

This project is open-source and available under the MIT License.

---

This README provides a clear overview of the project, how to run it, and answers to potential viva questions. You can customize it further as needed!