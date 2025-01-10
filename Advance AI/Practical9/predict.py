# **predict.py (separate file):**

import pickle
import numpy as np
import pandas as pd

# Load the trained model
pipe = pickle.load(open('best_model.pkl', 'rb'))

# Sample user input (replace with actual user input)
test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)

# Create a DataFrame from the user input
columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
test_input2_df = pd.DataFrame(test_input2, columns=columns)

# Make prediction on the user input
prediction = pipe.predict(test_input2_df)
print(f"Predicted Survival: {prediction[0]}")