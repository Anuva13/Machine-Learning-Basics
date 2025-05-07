from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Example data: [hours studied], target: 0 = fail, 1 = pass
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Predict probability for 6.5 study hours
hours = np.array([[6.5]])
prob = model.predict_proba(hours)
print(f"Probability of passing for 6.5 study hours: {prob[0][1]:.2f}")