
##### KNN como outro valor de K
##### DATASET IRIS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split #dividir o conjunto de dados
from sklearn.metrics import classification_report
import pandas as pd

dt = pd.read_csv('dados.csv', delimiter=',')
X = dt.iloc[:,:4]
y = dt.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state= 13)

### quantos vizinhos o algoritmo vai considerar

k = 10

clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred,))
print(y_pred)