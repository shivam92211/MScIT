import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
df = sns.load_dataset("iris")

# 1. Descriptive Statistics
print(df.describe())

# 2. Univariate Visualizations
# a. Histogram for 'sepal_length'
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal_length'], kde=True, bins=20, color='blue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# b. Boxplot for 'petal_length'
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['petal_length'], color='green')
plt.title('Boxplot of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.show()

# c. Density plot for 'sepal_width'
plt.figure(figsize=(8, 6))
sns.kdeplot(df['sepal_width'], fill=True, color='red')  # Replaced `shade` with `fill`
plt.title('Density Plot of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Density')
plt.show()

# 3. Bivariate Visualizations
# a. Scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', data=df, hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# b. Pairplot of all numerical variables
sns.pairplot(df, hue='species')
plt.show()

# c. Correlation Heatmap (only numeric columns)
corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()  # Exclude non-numeric columns
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()