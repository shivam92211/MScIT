import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load the Zoo dataset from a local CSV file (downloaded from Kaggle).
# Make sure the 'zoo.csv' file is in your working directory, or provide the full path.
df = pd.read_csv("3\zoo\zoo.csv")

# Prepare features and labels:
# Drop 'animal_name' as it's not a feature for classification.
# Adjust class labels from 1-7 to 0-6 (0-indexed) for proper one-hot encoding.
X = df.drop(["animal_name", 'class_type'], axis=1).values
y = to_categorical(df['class_type'].values - 1)

# 3. Split (80/20) with stratification, then scale properly
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# 4. Build the model: two hidden layers + softmax output
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(y.shape[1], activation='softmax'),
    tf.keras.layers.Dense(16, activation='relu'),
])

# 5. Compile & set up early stopping on val_loss
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 6. Train with validation split and early stopping
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# 7. Evaluate on test set
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.3f}")

# 8. Predict & compare for first 10 test samples
sample_X = X_test[:10]
pred_probs = model.predict(sample_X)
pred_classes = pred_probs.argmax(axis=1)
actual_classes = y_test[:10].argmax(axis=1)
print("Predicted:", pred_classes)
print("Actual:   ", actual_classes)