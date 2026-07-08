# Project 2: Data Classification Using AI

**DecodeLabs Industrial Training Kit — Batch 2026**

A supervised learning project that trains a K-Nearest Neighbors (KNN) model
to classify iris flowers into one of three species (Setosa, Versicolor,
Virginica) based on their measurements. This is the classic "Hello World"
of machine learning classification.

## 🧠 How It Works (the pipeline)

1. **Load the dataset** – the built-in Iris dataset (150 samples, 4 features:
   sepal length/width, petal length/width, 3 balanced classes).
2. **Train/Test split** – 80% of the data is used to train the model, 20% is
   held back to test it on data it has never seen (shuffled to avoid order
   bias, stratified to keep class balance equal in both sets).
3. **Feature scaling** – `StandardScaler` puts every feature on the same
   scale (mean = 0, variance = 1), which matters because KNN is
   distance-based.
4. **Train the model** – a `KNeighborsClassifier` (K=5) is fit on the
   training data.
5. **Evaluate** – predictions are made on the test set and scored with
   accuracy, a confusion matrix, and precision/recall/F1-score.
6. **Bonus** – finds the best K value using the "elbow method," and predicts
   the species of one brand-new, made-up flower.

## 🚀 How to Run

```bash
pip install -r requirements.txt
python classifier.py
```

This will print the full pipeline output to your terminal and save two
charts: `confusion_matrix.png` and `k_tuning.png`.

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~93% |

![Confusion Matrix](confusion_matrix.png)

The confusion matrix shows how many test flowers were correctly classified
per species (diagonal = correct, off-diagonal = mistakes).

![K Tuning](k_tuning.png)

This chart shows how the model's error rate changes as K increases — used
to pick the best number of neighbors.

## 🛠️ Skills Demonstrated

- Loading and understanding a real dataset
- Train/test splitting with stratification
- Feature scaling
- Supervised learning with KNN
- Model evaluation: accuracy, confusion matrix, precision, recall, F1-score
- Hyperparameter tuning (choosing K)

## 📌 Notes / Possible Extensions

- Try other classification algorithms (Decision Tree, Logistic Regression,
  SVM) and compare accuracy
- Try the model on a different dataset
- Visualize decision boundaries for two features at a time

---
Built as part of the DecodeLabs AI Engineering Internship, 2026.
