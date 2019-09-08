# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:40:22 2019
@author: Abhijit
"""
#Simple Linear Regression Model
 #first we need to preprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # years of experience is independant, one var matrix
y = dataset.iloc[:, 1].values# dependant variabloe

#No missing values, no need for imputer this time

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#We will split 10 to test, 20 to train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#No need for feature scaling

#Fitting Simple Lin Regression model to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#We are fine with default parameters
regressor.fit(X_train, y_train)#machine is the regressor, made it learn on the training set

#Machine can now based on its learning experience predict the new salary
#Regressor learned the correlations between experience and salary

#Predicting the test results - create a vector  of predictions
y_pred = regressor.predict(X_test) #vector of predictions of dependant variable

#The predictions are pretty damn close

#Visualizing the results with matplotlib
plt.scatter(X_train, y_train, color = 'red')#plots the real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue')#shows the comparisn between X_train and predictions
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()#display it

plt.scatter(X_test, y_test, color = 'red')#plots the real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue')#shows the comparisn between X_train and predictions
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()#display it
