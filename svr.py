"""
SVR
Support Vector Regression
 Like in linear regression, in svr there is a straight line going through all the points
 but in svr there is a 'tube' around this line with width, epsilon
 epsilon is measured vertically(along the y-axis) and not perpendicular to the tube
 SVR tries to fit as many points as possible in this 'tube'
"""
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values      # Independent variables         # All rows of all columns except the last one
y_ = dataset.iloc[:,2].values       # Dependent variables           # All rows of the third column
y=np.reshape(y_,(-1,1))             # Convert it into an array

# Use only if you have enough variables
"""
# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)
"""

# Feature scaling
from sklearn.preprocessing import StandardScaler
# Bringing both, the independant and dependant variables to the same scale
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the regression model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')  # One of the most popular non-linear kernel
regressor.fit(X,y)               # Learns the correlation


# Predicting Result with SVR Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


# Visualising SVR Results
X_grid = np.arange(min(X),max(X),0.1)     #all values of X incrimented by 0.1(X_grid = 1,1.1,1.2,1.3...)
#for smoother curve
X_grid = X_grid.reshape((len(X_grid), 1)) #convert into a matrix
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title(' Regression Model')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(y_pred)
