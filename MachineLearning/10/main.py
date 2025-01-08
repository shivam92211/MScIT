import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
import pandas as pd

# Load the trained model from the .joblib file
model = joblib.load('p10/trained_model.joblib')
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

print("Done")