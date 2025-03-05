# README: Bayesian Optimization for Random Forest Hyperparameter Tuning

This project demonstrates the use of Bayesian Optimization to tune hyperparameters of a Random Forest Classifier using the `BayesSearchCV` module from the `scikit-optimize` library. The code uses the Iris dataset, a standard dataset for classification tasks, and performs optimization to achieve the best cross-validation score.

---

## Features

- **Dataset**: Iris dataset is used as a benchmark dataset for multiclass classification.
- **Model**: Random Forest Classifier from `sklearn.ensemble`.
- **Hyperparameter Tuning**: Bayesian Optimization is applied to find the optimal hyperparameters.
- **Search Space**:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `min_samples_split`: Minimum fraction of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `max_features`: Fraction of features considered for splitting.
- **Cross-Validation**: 5-fold cross-validation is used to evaluate the model performance for each set of hyperparameters.

---

## Prerequisites

- Python 3.8 or above
- Required libraries:
  - `numpy`
  - `scikit-learn`
  - `scikit-optimize`

You can install the dependencies using:

```bash
pip install numpy scikit-learn scikit-optimize
```

---

## How to Use

1. Clone the repository or copy the script to your working directory.
2. Run the script in an environment with GPU support (optional but recommended for faster execution).
3. The script will:
   - Load the Iris dataset.
   - Define a Random Forest model.
   - Set up a search space for hyperparameter optimization.
   - Perform Bayesian optimization with cross-validation.
   - Print the best parameters and corresponding cross-validation score.

---

## Output Example

After running the script, you will see the best hyperparameters and the best cross-validation score:

```plaintext
Best Parameters: {'n_estimators': 150, 'max_depth': 10, 'min_samples_split': 0.05, 'min_samples_leaf': 2, 'max_features': 0.75}
Best Cross-Validation Score: 0.96
```

---


## Notes

- This example uses the Iris dataset, but the code can be easily adapted to other datasets by replacing the `load_iris()` function with your dataset.
- Bayesian Optimization efficiently narrows down the search space, making it ideal for hyperparameter tuning compared to grid or random search.
- The GPU usage is optional for this script but recommended when scaling to larger datasets or more complex models.

---

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this code.