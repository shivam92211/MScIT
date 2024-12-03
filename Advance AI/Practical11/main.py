# Use colab with GPU

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the search space for hyperparameters
param_space = {
    'n_estimators': Integer(10, 200),        # Number of trees
    'max_depth': Integer(1, 20),            # Maximum depth of a tree
    'min_samples_split': Real(0.01, 0.3),   # Minimum fraction of samples required to split
    'min_samples_leaf': Integer(1, 10),     # Minimum samples at a leaf node
    'max_features': Real(0.1, 1.0),         # Fraction of features to consider for split
}

# Bayesian Optimization with Cross-Validation
opt = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=50,  # Number of parameter settings to try
    cv=5,       # Number of cross-validation folds
    n_jobs=-1,  # Use all processors
    random_state=42
)

# Perform the optimization
opt.fit(X, y)

# Results
print("Best Parameters:", opt.best_params_)
print("Best Cross-Validation Score:", opt.best_score_)
