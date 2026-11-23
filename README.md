# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Matplotlib and Scikit-learn.
2.Create a dataset containing employee experience and corresponding salary.
3.Create and train the Decision Tree Regressor model using the dataset.
4.Predict the salary of a new employee and display the result with a graph.
## Program:
```py
# Program to implement Decision Tree Regressor
# for predicting the salary of an employee.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Sample dataset
# Experience in years
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)

# Salary in thousands
Y = np.array([15, 20, 25, 30, 35, 45, 50, 60, 70, 80])

# Create Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X, Y)

# Predict salary for a new employee
experience = np.array([[6.5]])
prediction = model.predict(experience)

# Display output
print("Employee Experience:", experience[0][0], "years")
print("Predicted Salary:", prediction[0], "Thousand")

# Predict training data
Y_pred = model.predict(X)

# Plot graph
plt.scatter(X, Y, color='blue', label='Actual Salary')
plt.plot(X, Y_pred, color='red', label='Predicted Salary')

plt.xlabel("Experience (Years)")
plt.ylabel("Salary (Thousands)")
plt.title("Decision Tree Regressor - Salary Prediction")
plt.legend()
plt.grid()
plt.show()
```

## Output:
~~~
Employee Experience: 6.5 years
Predicted Salary: 50.0 Thousand
~~~
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
