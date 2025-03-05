# Develop an API to deploy your model and perform predictions

# This is a step-by-step guide for training a machine learning model, saving it, and then deploying it using FastAPI for predictions.

# Step 1: Train a Model and Save It
# Create a file called train.py. This file will train a model, save it using joblib, and visualize feature importance.
# train.py

# Code:

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()
# Convert the data into a DataFrame for easier handling
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
# Quick inspection of the dataset
print(df.head())
print(df.info())
# Data Preprocessing
# Features (X) are all columns except 'target'
X = df.drop(columns=['target'])
y = df['target']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the model (RandomForestClassifier in this case)
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model using joblib
joblib.dump(model, 'trained_model.joblib')

# Optionally: Visualize feature importance (if the model allows it)
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(X.shape[1]), importances[indices], align="center")
plt.yticks(range(X.shape[1]), features[indices])
plt.xlabel("Relative Importance")
plt.show()

print("Model saved as 'trained_model.joblib'")

# Step 2: Load the Model and Create the API Using FastAPI

# Once you have the model saved, create an API using FastAPI. This API will accept inputs via POST requests and return predictions.

# Create a file called main.py:
# main.py

# Code:

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
import pandas as pd

# Load the trained model from the .joblib file
model = joblib.load('trained_model.joblib')
# Define the input data structure using Pydantic
class FlowerInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
# Initialize the FastAPI app
app = FastAPI()

# Define a function to make predictions
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Map prediction to actual species names
    iris = load_iris()
    species = iris.target_names[prediction]
    return species[0]

# Define an endpoint for prediction
@app.post("/predict/")
async def predict(flowers: FlowerInput):
    # Call the prediction function
    predicted_species = predict_species(flowers.sepal_length, flowers.sepal_width, flowers.petal_length, flowers.petal_width)
    
    # Return the result as a JSON response
    return {"predicted_species": predicted_species}

# Step 3: Run the API

# To run the API, use the following command:
# uvicorn main:app --reload
# This will start a FastAPI app on http://127.0.0.1:8000. You can now make predictions by sending POST requests.

# Step 4: Test Your API

# You can now test your API using a terminal command (in PowerShell or Command Prompt on Windows).
# Here's how you can call your FastAPI using PowerShell with Invoke-WebRequest:
# $headers = @{
#     "accept" = "application/json"
#     "Content-Type" = "application/json"
# }

# $data = '{"sepal_length": 9, "sepal_width": 9, "petal_length": 0, "petal_width": 9}'

# Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict/" -Method Post -Headers $headers -Body $data

# Expected Output:

# The API will return a response with the predicted species. The response will look like:
# {
#   "predicted_species": "setosa"
# }
# This confirms that the API is working, and the model is making predictions based on the input features.

# First, create a requirements.txt file for your project dependencies.
# requirements.txt
# Copy code
# fastapi==0.95.1
# uvicorn==0.23.1
# scikit-learn==1.2.0
# pandas==1.5.3
# joblib==1.2.0
# numpy==1.23.5
# matplotlib==3.6.3
# seaborn==0.12.2
