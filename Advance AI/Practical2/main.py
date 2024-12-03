# Use Colab 

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Download movie reviews data
nltk.download("movie_reviews")

# Load movie reviews as labeled data
def load_data():
    labeled_data = [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    return labeled_data


# Extract features
def word_features(words):
    return {word: True for word in words}


# Load data and shuffle
from random import shuffle

data = load_data()
shuffle(data)

# Train-test split
train_data = data[:1600]
test_data = data[1600:]

# Train classifier
train_features = [(word_features(words), sentiment) for words, sentiment in train_data]
classifier = NaiveBayesClassifier.train(train_features)


# Evaluate on the test set
test_features = [(word_features(words), sentiment) for words, sentiment in test_data]
print("Accuracy:", accuracy(classifier, test_features))

# Show most informative features
classifier.show_most_informative_features(10)


# Test with custom input
def predict_sentiment(text):
    words = text.split()
    features = word_features(words)
    return classifier.classify(features)

print(predict_sentiment("This movie was amazing and inspiring."))
print(predict_sentiment("The film was boring and poorly acted."))
