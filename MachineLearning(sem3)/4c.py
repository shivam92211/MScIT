import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Step 1: Load a Sample Dataset
# For example, we use the iris dataset available in scikit-learn
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and Train a Decision Tree Classifier (Control max depth to avoid overfitting)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)  # max_depth controls overfitting
clf.fit(X_train, y_train)

# Step 4: Evaluate the Model (Optional)
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
