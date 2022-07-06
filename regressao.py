
### REGRESSÃO
### DATASET CPU

from statistics import mean
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('diabetes.csv', delimiter=',')

df['class'].replace(['tested_negative', 'tested_positive'], [0, 1], inplace = True)

X = df.drop('class', axis=1)
y = df['class']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=1)

regressao = LinearRegression()
model = linear_model.LinearRegression()
regressao.fit(X_train, y_train)
y_pred = regressao.predict(X_test)
model.fit(X_train, y_train)
erro = mean_squared_error(y_test, y_pred)

print(erro)
print("_________________________")
print(f"Score:{regressao.score(X_test, y_test)}")
print("_________________________")
print(y_pred)

