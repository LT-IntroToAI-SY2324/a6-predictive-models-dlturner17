import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model


data = pd.read_csv("final-project/data.csv")
data['Gender'].replace(['M', 'F'], [0,1], inplace = True )
x = data[["Gender"]].values 
print(x)
y = data[["Probability"]].values 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)
model = LinearRegression().fit(xtrain, ytrain)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
print(coef, intercept, r_squared)

plt.scatter(x,y, c="purple")
plt.scatter(x,y)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.show()