import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RBF Network using SVM with RBF Kernel
rbf_model = SVC(kernel='rbf', gamma=1.0, random_state=42)
rbf_model.fit(X_train, y_train)

# Predictions
y_pred = rbf_model.predict(X_test)

# Accuracy
print("RBF Network Accuracy:", accuracy_score(y_test, y_pred))
