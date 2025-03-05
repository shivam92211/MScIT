# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Binarizer

# Step 2: Create a synthetic dataset
data = {
    "Numerical_Column": [10, 20, 30, 40, 50],
    "Categorical_Column": ["Low", "Medium", "High", "Medium", "Low"],
    "Binary_Column": [1, 0, 1, 0, 1]
}

# Convert to a DataFrame
df = pd.DataFrame(data)

# Step 3: Handle missing values (if any)
# Here we simulate missing values and fill them
df.loc[1, "Numerical_Column"] = np.nan
df["Numerical_Column"].fillna(df["Numerical_Column"].mean(), inplace=True)

# Step 4: Apply Label Encoding to the categorical column
label_encoder = LabelEncoder()
df["Categorical_Column_Encoded"] = label_encoder.fit_transform(df["Categorical_Column"])

# Step 5: Scale numerical columns
scaler = StandardScaler()
df["Numerical_Scaled"] = scaler.fit_transform(df[["Numerical_Column"]])

# Min-Max Scaling for comparison
min_max_scaler = MinMaxScaler()
df["Numerical_MinMax_Scaled"] = min_max_scaler.fit_transform(df[["Numerical_Column"]])

# Step 6: Apply Binarization to numerical columns
binarizer = Binarizer(threshold=25)
df["Numerical_Binarized"] = binarizer.fit_transform(df[["Numerical_Column"]])

# Print the resulting DataFrame
print(df)
