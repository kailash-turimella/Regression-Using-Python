"""
Simple Linear Regression
    Straight line with one independant variable

-->  y = b0 + b1*x   <--

y = dependent variable         the value we need to predict
x = independent variable       the value which causes y to change
b0 = constant                  the point where the line crosses the y axis (when x = 0)
b1 = coefficient               the slope of the line
"""

# Importing libraries
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing datasets
dataset = pd.read_csv('Salary_Data.csv') # Dataset
X = dataset.iloc[:,:-1].values           # Independant Variables
y = dataset.iloc[:,1].values             # Dependant Variables

# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 1/3) # Dividing into training and test sets

# Fitting the Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()   # Object of the LinearRegression class
regressor.fit(X_train,y_train)   # The regressor learns the relationship between the two

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Visualising the traing set result
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the test set result
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Evaluating Model Performance
#    Higher the better
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)