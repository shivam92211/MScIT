# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load and Prepare the Dataset
# For demonstration purposes, let’s use a sample dataset like the famous Iris dataset or any binary classification dataset.
# Sample dataset: load it using scikit-learn
from sklearn.datasets import load_iris
iris = load_iris()

# We will use only two classes (binary classification) from the Iris dataset
X = iris.data[iris.target != 2]  # Features
y = iris.target[iris.target != 2]  # Target variable

# Step 3: Split the Data into Training and Testing Sets
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Logistic Regression Model
# Initialize the Logistic Regression model
model = LogisticRegression()
# Train the model
model.fit(X_train, y_train)

# Step 5: Make Predictions
# Make predictions on the test set
y_pred = model.predict(X_test)

# Predict probabilities for ROC curve (output probabilities for class 1)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluate the Model
# We will now compute the accuracy, precision, recall, and plot the ROC curve.

# 1.	Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
# 2.	Precision
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision:.4f}')
# 3.	Recall
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')
# 4.	ROC Curve
# To understand the ROC curve, we need to plot it using roc_curve and calculate the AUC (Area Under the Curve).
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute the Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line (no skill)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

