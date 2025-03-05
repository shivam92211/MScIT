### README: Movie Recommendation System Using SVD

---

## Overview

This project implements a **Movie Recommendation System** using the **Surprise Library** and **Singular Value Decomposition (SVD)**. The system predicts user ratings for movies based on historical data. It evaluates the model's accuracy using **Root Mean Squared Error (RMSE)** and provides predictions for specific user-movie pairs.

---

## Features

- **Load and preprocess data**: Reads a CSV file containing user-movie ratings.
- **Train-Test split**: Splits the dataset into training and testing sets.
- **Model Training**: Uses the SVD algorithm to predict user ratings.
- **Evaluation**: Computes the RMSE to measure model accuracy.
- **Prediction**: Predicts the rating a user would give to a specific movie.

---

## Requirements

### Python Libraries:
Install the required libraries using:
```bash
pip install pandas scikit-surprise
```

---

## Dataset Format

The system expects a CSV file named `ratings.csv` with the following structure:

| userId | movieId | rating | timestamp     |
|--------|---------|--------|---------------|
| 1      | 10      | 4.0    | 964982703     |
| 2      | 15      | 3.5    | 964982931     |

- **userId**: Unique identifier for each user.
- **movieId**: Unique identifier for each movie.
- **rating**: Rating given by the user (scale: 1 to 5).
- **timestamp**: (Optional) Timestamp of the rating.

---

## How to Use

1. **Prepare the Dataset**:
   - Ensure the dataset (`ratings.csv`) is in the same directory as the script.
   - The dataset should have columns: `userId`, `movieId`, `rating`.

2. **Run the Script**:
   ```bash
   python main.py
   ```

3. **Output**:
   - **RMSE**: Displays the Root Mean Squared Error of the predictions.
   - **Prediction**: Prints the predicted rating for a specific user-movie pair.

---

## Example Output

```plaintext
RMSE: 0.8567
Predicted rating for User 1 on Item 10: 3.89
```

---

## Customization

- **Change the user and movie for predictions**:
   Modify these lines in the script:
   ```python
   user_id, item_id = <USER_ID>, <MOVIE_ID>
   ```

- **Adjust test size**:
   Modify the `test_size` parameter in `train_test_split`:
   ```python
   trainset, testset = train_test_split(surprise_data, test_size=0.2)
   ```

---

## References

- [Surprise Library Documentation](https://surprise.readthedocs.io/)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
