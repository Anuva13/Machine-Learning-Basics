from sklearn.linear_model import LinearRegression
import numpy as np

# X has multiple features (columns)
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40]
])
y = np.array([12, 24, 36, 48])

# Create and train model
model = LinearRegression()
model.fit(X,y)

# Predict for new data point [5, 50]
print(model.predict([[5,50]]))