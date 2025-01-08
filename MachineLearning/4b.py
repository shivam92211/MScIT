import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Read training data from a CSV file
# Let's assume the CSV file has feature columns and the target column is named 'class'.
data = pd.read_csv('data.csv')  # Update with your actual file path

# Split the data into features (X) and target labels (y)
X = data.drop(columns=['class'])  # All columns except 'class'
y = data['class']  # The target column

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the k-NN classifier and train the model
k = 3  # You can change k to any number for the number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Step 4: Predict the labels for the test set
y_pred = knn.predict(X_test)

# Step 5: Print correct and wrong predictions with sample data
for i in range(len(X_test)):
    sample = X_test.iloc[i]
    true_label = y_test.iloc[i]
    predicted_label = y_pred[i]
    
    if true_label == predicted_label:
        print(f"Correct: Sample {i+1} -> Features: {sample.values} | True label: {true_label} | Predicted: {predicted_label}")
    else:
        print(f"Wrong: Sample {i+1} -> Features: {sample.values} | True label: {true_label} | Predicted: {predicted_label}")

# Step 6: Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
