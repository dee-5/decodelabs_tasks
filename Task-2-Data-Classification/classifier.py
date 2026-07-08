"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit - Batch 2026

Goal: Build a basic classification model using a small dataset (Iris),
following the Input -> Process -> Output pipeline:
  INPUT   -> Load and understand the dataset
  PROCESS -> Split into train/test, scale features, train a KNN model
  OUTPUT  -> Predict on unseen data and validate with metrics
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # allows saving plots without opening a window
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# -------------------------------------------------
# STEP 1 (INPUT): Load and understand the dataset
# -------------------------------------------------
iris = load_iris()
X = iris.data          # features: sepal length, sepal width, petal length, petal width
y = iris.target        # labels: 0 = setosa, 1 = versicolor, 2 = virginica

print("=" * 50)
print("STEP 1: DATASET OVERVIEW")
print("=" * 50)
print(f"Total samples : {X.shape[0]}")
print(f"Features      : {iris.feature_names}")
print(f"Classes       : {list(iris.target_names)}")
print()

# -------------------------------------------------
# STEP 2 (PROCESS): Split into train/test sets
# -------------------------------------------------
# shuffle=True randomizes order before splitting (removes order bias)
# stratify=y keeps the same class balance in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

print("=" * 50)
print("STEP 2: TRAIN / TEST SPLIT")
print("=" * 50)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print()

# -------------------------------------------------
# STEP 3 (PROCESS): Feature scaling
# -------------------------------------------------
# KNN relies on distance, so features must be on the same scale.
# Fit the scaler ONLY on training data, then apply it to both sets
# (this avoids "leaking" info from the test set into training).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------
# STEP 4 (PROCESS): Train the classification model
# -------------------------------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

print("=" * 50)
print("STEP 3: MODEL TRAINED")
print("=" * 50)
print("Algorithm: K-Nearest Neighbors (K=5)")
print()

# -------------------------------------------------
# STEP 5 (OUTPUT): Predict & validate
# -------------------------------------------------
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("=" * 50)
print("STEP 4: RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(cm)
print()

print("Classification Report (Precision / Recall / F1-score):")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# Save confusion matrix as an image (nice for the GitHub portfolio)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Iris KNN Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved chart: confusion_matrix.png")

# -------------------------------------------------
# BONUS: Find the best K value ("elbow method")
# -------------------------------------------------
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(np.mean(pred_k != y_test))

best_k = list(k_range)[int(np.argmin(error_rates))]

plt.figure()
plt.plot(list(k_range), error_rates, marker="o")
plt.title("Choosing K: Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.tight_layout()
plt.savefig("k_tuning.png")
plt.close()
print(f"Saved chart: k_tuning.png (best K found: {best_k})")

# -------------------------------------------------
# BONUS: Predict a brand new, made-up flower
# -------------------------------------------------
sample_flower = [[5.1, 3.5, 1.4, 0.2]]  # sepal_len, sepal_wid, petal_len, petal_wid
sample_scaled = scaler.transform(sample_flower)
sample_prediction = model.predict(sample_scaled)[0]

print()
print("=" * 50)
print("BONUS: PREDICTING A NEW FLOWER")
print("=" * 50)
print(f"Input measurements : {sample_flower[0]}")
print(f"Predicted species  : {iris.target_names[sample_prediction]}")
