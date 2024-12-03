
# Sentiment Analysis with NLTK

A simple sentiment analysis model built using Python and the Natural Language Toolkit (NLTK). This project uses the **Movie Reviews** dataset provided by NLTK to classify text as either **positive** or **negative** based on word presence.

---

## Features

- Utilizes **Naive Bayes Classifier** for simplicity and efficiency.
- Preprocesses text data to extract word-based features.
- Demonstrates high interpretability by showing the most informative features.
- Allows sentiment prediction for custom input text.

---

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.7 or higher
- Required Python packages:
  - `nltk`

Install the required package using:
```bash
pip install nltk
```

---

## How It Works

1. **Dataset**: The model uses the **Movie Reviews** dataset from NLTK, which contains labeled positive and negative movie reviews.
2. **Feature Extraction**: Each review is represented as a dictionary of word presence (`True` if the word is in the review).
3. **Training**: A **Naive Bayes Classifier** is trained on the extracted features.
4. **Evaluation**: The model's accuracy is evaluated on a test set.
5. **Prediction**: The trained classifier predicts the sentiment of custom input text.

---

## Running the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-nltk.git
cd sentiment-analysis-nltk
```

### 2. Run the Code
Run the Python script to train the model, evaluate accuracy, and test custom inputs:
```bash
python sentiment_analysis.py
```

---

## Example Usage

```python
# Predict sentiment for a custom input
predict_sentiment("This movie was amazing and inspiring.")  # Output: pos
predict_sentiment("The film was boring and poorly acted.")  # Output: neg
```

---

## Accuracy

The model achieves an accuracy of approximately **75%** on the Movie Reviews test dataset. This can vary slightly depending on the train-test split.

---

## Files

- `sentiment_analysis.py`: Main Python script containing the model code.
- `README.md`: Documentation for the project.

---

## To Do

- Improve feature extraction (e.g., add bigrams or TF-IDF).
- Test with external datasets.
- Implement a GUI or web interface for easy sentiment prediction.

---

## License

This project is licensed under the MIT License. Feel free to use and modify it.
