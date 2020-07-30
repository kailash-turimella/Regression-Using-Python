"""
Polynomial Regression
   Exponential growth
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values     # Independent         # All rows of all columns except the last one
y = dataset.iloc[:,2].values       # Dependent           # All rows of the third column

# Not enough observations to be involving training and test set


# Fitting linear regression to dataset
#created just to compare
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualising Linear Regression Results
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Visualising Polynomial Regression Results
X_grid = np.arange(min(X),max(X),0.1)     #all values of X incrimented by 0.1(X_grid = 1,1.1,1.2,1.3...)
# For smoother curve
X_grid = X_grid.reshape((len(X_grid), 1)) # Convert into matrix
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Predicting Result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting Result with Polynomial Regression
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))
