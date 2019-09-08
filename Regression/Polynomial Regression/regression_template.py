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

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#PREPROCESSING DONE -----------------------------------------------------------------

#Building a Poly reg model
from sklearn.preprocessing import PolynomialFeatures as PF
poly_reg = PF(degree = 2) #Transforms X into X poly contains the idnependant variables with x^2 etc
X_poly = poly_reg.fit_transform(X)
#Lin reg to use the new polynomial terms
lin_reg2 = LR()
lin_reg2.fit(X_poly, y)

#Visualize the linreg

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
#lin_reg2.predict(poly_reg.fit_transform(6.5))
