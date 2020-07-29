"""
Multiple Linear Regression
      Straight line with multiple independant variables

-->   y = b0 + b1*x1 + b2*x2 + b3*x3...   <--

y = dependent variable         the value we need to predict
x = independent variable       the value which causes y to change
b0 = constant                  the point where the line crosses the y axis (when x = 0)
b1 = coefficient               the slope of the line
"""

# Data preprocessing


# Importing libraries
import numpy as np
import pandas as pd

# Importing datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values     # Independent         # All rows of all columns except the last one
y = dataset.iloc[:,4].values       # Dependent           # All rows of the third column


# Encoding categorical data into dummy variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
# Encoding the independent variables
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])            # change the words into numbers
# Encode(convert into three columns) # Fourth column 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# Avaoiding the dummy variable trap
X = X[:,1:]   # Removing fist column
"""
0,0 - California
1,0 - Florida
0,1 - New York
"""
# Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2,random_state = 0) # test size = 20% 


# Fitting Multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)   # Regressor learns the relationship between the independant and dependant variables

# Predicting Test set results
y_pred = regressor.predict(X_test)


# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis = 1) #axis 1 - column; 2 - row
'''
(b0 + b1*x1 + b2*x2 + b3*x3...) = (b0*x0 + b1*x1 + b2*x2 + b3*x3...)
only when x0 = 1
'''
X_opt = X[:,[0,1,2,3,4,5]]  # Columns
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
# Remove column with highest p value ONE BY ONE
X_opt = X[:,[0,1,3,4,5]]  
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
# Remove column with highest p value ONE BY ONE
X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
# Remove column with highest p value ONE BY ONE
X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
# Remove column with highest p value ONE BY ONE
X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog = y,exog = X_opt).fit()
regressor_ols.summary()
# IF ADJUSTED R-SQUARED REDUCES WHEN YOU REMOVE A VARIABLE, ADD IT BACK
#if the coef is a negative number, the variable is inversely proportional to the dependant variable
#for every unit of increase of the independant variable the dependant variable changes by the magnitude of its coef
