import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset: [experience, education, age]
# X represents the features: experience, education, age
# y represents the target variable: salary
X = np.array([
    [5, 16, 30],  # 5 years experience, 16 years education, 30 years old
    [10, 18, 40], # 10 years experience, 18 years education, 40 years old
    [3, 14, 25],  # 3 years experience, 14 years education, 25 years old
    [7, 16, 35],  # 7 years experience, 16 years education, 35 years old
    [2, 12, 22],  # 2 years experience, 12 years education, 22 years old
    [15, 20, 45], # 15 years experience, 20 years education, 45 years old
])

y = np.array([25000, 40000, 22000, 35000, 20000, 55000])  # corresponding salaries

# Create the linear regression model
model = LinearRegression()

# Train the model using the data
model.fit(X,y)

# Predict salary based on user input
experience = float(input("Enter years of experience: "))
education = float(input("Enter year of education: "))
age = int(input("Enter age: "))

# Predict salary using the trained model
predicted_salary = model.predict([[experience, education, age]])

print(f"Predicted Salary: ${predicted_salary[0]:.2f}")
