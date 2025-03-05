from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
# Load dataset
data = load_iris()
X, y = data.data, data.target

# Initialize model
model = DecisionTreeClassifier(random_state=42)

# Define k-Fold Cross-Validation
k = 5  # Number of folds
kfold = KFold(n_splits=k, shuffle=True, random_state=42)

# Evaluate with k-Fold CV
kfold_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"k-Fold Cross-Validation Accuracy: {kfold_scores}")
print(f"Mean Accuracy (k-Fold): {kfold_scores.mean():.2f}")

# Define Stratified k-Fold Cross-Validation
stratified_kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# Evaluate with Stratified k-Fold CV
stratified_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
print(f"Stratified k-Fold Cross-Validation Accuracy: {stratified_scores}")
print(f"Mean Accuracy (Stratified k-Fold): {stratified_scores.mean():.2f}")

# Advanced: Custom Scorer with F1-Score
f1_scorer = make_scorer(f1_score, average='weighted')
f1_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring=f1_scorer)
print(f"Stratified k-Fold Cross-Validation F1-Score: {f1_scores}")
print(f"Mean F1-Score (Stratified k-Fold): {f1_scores.mean():.2f}")