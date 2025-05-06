from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample Data
X = np.array([[1], [2], [3], [4]])
y = np.array([[2], [4], [6], [8]])

# Create and train model
model = LinearRegression()
model.fit(X,y)

# Make prediction
x_pred = np.array([[5]])
y_pred = model.predict(x_pred)
print(f"Prediction for x = 5: {y_pred[0]}")

# Plot the training data and the regression line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X, model.predict(X), color='green', label='Regression line')
plt.scatter(x_pred, y_pred, color='red', label='Prediction (x=5)')
plt.text(5, y_pred[0], f"({x_pred[0][0]}, {float(y_pred[0]):.1f})", fontsize=12, ha='left', va='bottom')

plt.title("Linear Regression Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()