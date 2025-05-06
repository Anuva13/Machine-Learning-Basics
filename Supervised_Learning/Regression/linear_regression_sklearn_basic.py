from sklearn.linear_model import LinearRegression

# Sample Data
X = [[1], [2], [3], [4]]
y = [[2], [4], [6], [8]]

# Create and train model
model = LinearRegression()
model.fit(X,y)

# Make prediction
print(f"Prediction for x = 5: {model.predict([[5]])}")