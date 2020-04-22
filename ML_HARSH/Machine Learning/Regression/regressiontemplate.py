import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values

#Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"""

#Feature Scaling
from sklearn.preprocessing import StandardScaler
"""sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Regression Model to the dataset
#Create your Regressor

#Predicting the new result with Regression Model
y_pred = regressor.predict(X)

#Visualing the Regression Model results
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(Regression Model)')
plt.show()

#Visualing the Regression Model results(for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.title('Truth or Bluff(Regression Model)')
plt.show()