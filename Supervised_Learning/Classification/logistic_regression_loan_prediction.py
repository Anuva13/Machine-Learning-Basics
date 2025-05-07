from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Features: [Income in $1000s, Credit Score, Loan Amount in $1000s, Employment Status]
X = np.array([
    [25, 600, 5, 1],
    [30, 650, 10, 1],
    [45, 700, 20, 1],
    [15, 550, 5, 0],
    [35, 720, 15, 1],
    [20, 580, 8, 0],
    [50, 750, 25, 1],
    [28, 610, 7, 0],
    [42, 690, 18, 1],
    [22, 590, 9, 0]
])

# Labels: 1 = loan approved, 0 = loan denied
y = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Predict on a new applicant
new_applicant = np.array([[40, 680, 12, 1]])  # good income, decent credit, employed
prob = model.predict_proba(new_applicant)
print(f"Loan Approval Probability: {prob[0][1]:.2f}")