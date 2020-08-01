"""
Decision Tree Regression

The algorithm splits the dependant variables.
It decides where the splits should be based on a mathematical concept called information entropy
Information entropy basically checks if the splits are increasing the amount of information we have about our points
  or adding any value to the way we want to group our points
Each section is called a leaf or a Terminal Node
There are different conditions for deciding when the algorithm stops.
  For example, when there are 5% of the points in a leaf

Summary:
  The independant variable are graphed
  They are then divided into sections(a leaf) based on information entropy
  The predicted value of y is the average value of all the independant variables in that perticular leaf

"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values     # Independent         # All rows of all columns except the last one
y = dataset.iloc[:,2].values       # Dependent           # All rows of the third column

# Use only if you have enough variables
"""
# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)
"""

# Feature scaling
#        bringing the age and salary to a specific range
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)          # Already fitted
y_train = sc_y.fit_transform(y_train)
"""


# Fitting the regression model to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)

# Predicting Result with Polynomial Regression
y_pred = regressor.predict([[6.5]])

# Visualising Regression Results
X_grid = np.arange(min(X),max(X),0.01)     #all values of X incrimented by 0.1(X_grid = 1,1.1,1.2,1.3...)
X_grid = X_grid.reshape((len(X_grid), 1))  #convert into a matrix
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Decision Tree Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(y_pred)