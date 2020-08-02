"""
Random Forest Regression

Ensemble Learning - When you take multiple machine learning algorithms and combine them.

Random Forest is a type of ensemble learning
Ensemble learning is when you take multiple algorithms and put them together to make something more powerful
   or the same algorithm multiple times
Random Forest combines multiple decision trees

Pick random points from your dataset
Build a decision tree of those points(instead of building a decision tree of all the points)
Choose the number of trees you want(n_estimators) and repeat steps 1&2.
You will have multiple decision trees
The prediction = the average value of predictions made by each decision tree

Instead of getting one prediction, like in decision tree regression, you are getting multiple predictions
 and that improves the accuracy
 for example if a tree doesn't turn out to be as good it would not ruin the prediction as we are taking the average
 
Ensemble algorithms are more stable
 because any changes in the dataset could impact one tree but for them to impact multiple tree is harder
"""
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values     # Independent         # All rows of all columns except the last one
y = dataset.iloc[:,2].values       # Dependent           # All rows of the third column

# Fitting the regression model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300)  # Number of trees predicting the value of y
regressor.fit(X,y)

# Predicting Result with Polynomial Regression
y_pred = regressor.predict([[6.5]])

# Visualising Regression Results(Higher resolution)
X_grid = np.arange(min(X),max(X),0.01)     #all values of X incrimented by 0.1(X_grid = 1,1.1,1.2,1.3...)
X_grid = X_grid.reshape((len(X_grid), 1)) #convert into a matrix
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Random Forest Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(y_pred)
