# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# Load the dataset (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Train a single Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions with the Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Calculate evaluation metrics for the Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='macro')
recall_dt = recall_score(y_test, y_pred_dt, average='macro')

# Print performance for Decision Tree
print("Decision Tree Performance:")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}\n")

# 2. Train Random Forest models with different numbers of trees (n_estimators) and feature sampling
n_estimators_list = [10, 50, 100, 200]
max_features_list = ['sqrt', 'log2', None]  # Corrected feature sampling strategies
results = []

for n_estimators in n_estimators_list:
    for max_features in max_features_list:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions with the Random Forest
        y_pred_rf = rf_model.predict(X_test)

        # Calculate evaluation metrics for Random Forest
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf, average='macro')
        recall_rf = recall_score(y_test, y_pred_rf, average='macro')
        
        # Save the results
        results.append({
            'n_estimators': n_estimators,
            'max_features': max_features,
            'accuracy': accuracy_rf,
            'precision': precision_rf,
            'recall': recall_rf
        })

# Convert the results to a DataFrame for easier comparison
results_df = pd.DataFrame(results)

# Print Random Forest results for comparison
print("\nRandom Forest Performance with different hyperparameters:")
print(results_df)

# 3. Compare the performance of Decision Tree and Random Forest models
# We will plot the accuracy scores for both models to visualize the performance
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['accuracy'], label="Random Forest Accuracy", marker='o')
plt.axhline(y=accuracy_dt, color='r', linestyle='--', label="Decision Tree Accuracy")
plt.title('Random Forest vs Decision Tree Accuracy')
plt.xlabel('Number of Trees in Random Forest')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 4. Plot Precision vs Recall for Random Forest models
plt.figure(figsize=(10, 6))
plt.plot(results_df['n_estimators'], results_df['precision'], label="Random Forest Precision", marker='o', color='b')
plt.plot(results_df['n_estimators'], results_df['recall'], label="Random Forest Recall", marker='s', color='g')
plt.axhline(y=precision_dt, color='r', linestyle='--', label="Decision Tree Precision")
plt.axhline(y=recall_dt, color='orange', linestyle='--', label="Decision Tree Recall")
plt.title('Random Forest Precision and Recall vs Decision Tree')
plt.xlabel('Number of Trees in Random Forest')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()