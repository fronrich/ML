#Polynomial Regression
#Start of nonlinear regressor
"""
Created on Sat Sep  7 16:31:09 2019
@author: Abhijit
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Dont need to split if there are not many operations not enough to train and test both
#No training set and test set

#No need to feature scaling since the library takes care of the feature scaling in this cas

#PREPROCESSING DONE -----------------------------------------------------------------

#Build a linear regression and a poly reg to compare them to each other

from sklearn.linear_model import LinearRegression as LR
lin_reg = LR()
lin_reg.fit(X,y)

#Building a Poly reg model
from sklearn.preprocessing import PolynomialFeatures as PF
poly_reg = PF(degree = 2) #Transforms X into X poly contains the idnependant variables with x^2 etc
X_poly = poly_reg.fit_transform(X)
#Lin reg to use the new polynomial terms
lin_reg2 = LR()
lin_reg2.fit(X_poly, y)

#Visualize the linreg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff(Linear Regression Results)')
y_lin_pred = lin_reg.predict(X)
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
#Linear Regression is not the move here straight line cannot correspond to this
#Lin reg prediction is around $300000 for the employee

#Visualize the polyreg
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')#IMPORTANT LINE 
plt.title('Truth or Bluff(Polynomial Regression Results)')
y_lin_pred = lin_reg.predict(X)
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()
#According to the polynomial regression model, he should only get like 200,000
#As  the degree goes up the regression equation is more accurate to the target data


#Prediction with LinReg
#Predicts the salary at the level 6.5 according to the model
#Predicted at 300,000

#Prediction with PolyReg
lin_reg2.predict(poly_reg.fit_transform(6.5))
