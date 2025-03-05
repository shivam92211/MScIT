import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Load the dataset from a CSV file
file_path = 'data.csv'  # Replace with your file path
df = pd.read_csv(file_path)
# Display the first few rows
print(df.head())
# Handle Missing Values
print("Missing values before cleaning:")

print(df.isnull().sum())
# Fill missing values with mean for numerical columns and mode for categorical columns
df['numerical_column'] = df['numerical_column'].fillna(df['numerical_column'].mean())
df['category_column'] = df['category_column'].fillna(df['category_column'].mode()[0])
# Handle Inconsistent Formatting (Example: Strip spaces and convert to lowercase)
df.columns = df.columns.str.strip()  # Strip spaces from column names
df['category_column'] = df['category_column'].str.lower()  # Convert text to lowercase
# Handle Outliers (Using IQR)
Q1 = df['numerical_column'].quantile(0.25)
Q3 = df['numerical_column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df[(df['numerical_column'] >= lower_bound) & (df['numerical_column'] <= upper_bound)]
# Display cleaned data
print("Cleaned data:")
print(df_cleaned.head())
# Visualize with Boxplot to check for outliers
sns.boxplot(x=df_cleaned['numerical_column'])
plt.title('Boxplot of Numerical Column After Cleaning')
plt.show()

# Visualize the distribution with a histogram
df_cleaned['numerical_column'].hist(bins=30)
plt.title('Histogram of Numerical Column After Cleaning')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
