import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Truth or Bluff')

#Visualing the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(Polynomial Regression)')

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualing the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(Polynomial Regression)')

#Visualing the Polynomial Regression results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(Polynomial Regression)')

#Predicting a new result with linear regression
lin_reg.predict(X)

#Predicting a new result with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
