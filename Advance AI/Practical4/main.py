import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy  # Import accuracy from surprise

#userId,movieId,rating,timestamp
# Load the dataset
data = pd.read_csv('ratings.csv')  # UserID, ItemID, Rating
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)

# Split the data
trainset, testset = train_test_split(surprise_data, test_size=0.2)

# Use SVD (Singular Value Decomposition)
model = SVD()
model.fit(trainset)

# Evaluate
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))  # Use the correct accuracy module

# Make a prediction
user_id, item_id = 1, 10
pred = model.predict(user_id, item_id)
print(f"Predicted rating for User {user_id} on Item {item_id}: {pred.est}")
