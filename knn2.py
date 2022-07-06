
##### KNN padrão

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split #dividir o conjunto de dados
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dt = pd.read_csv('vote.csv', delimiter=',', na_values='?')


dt['Class'].replace(['republican', 'democrat'], [1, 0], inplace = True)
dt['handicapped-infants'].replace(['y', 'n'], [1, 0], inplace = True)
dt['water-project-cost-sharing'].replace(['y', 'n'], [1, 0], inplace = True)
dt['adoption-of-the-budget-resolution'].replace(['y', 'n'], [1, 0], inplace = True)
dt['physician-fee-freeze'].replace(['y', 'n'], [1, 0], inplace = True)
dt['el-salvador-aid'].replace(['y', 'n'], [1, 0], inplace = True)
dt['religious-groups-in-schools'].replace(['y', 'n'], [1, 0], inplace = True)
dt['anti-satellite-test-ban'].replace(['y', 'n'], [1, 0], inplace = True)
dt['aid-to-nicaraguan-contras'].replace(['y', 'n'], [1, 0], inplace = True)

dt['mx-missile'].replace(['y', 'n'], [1, 0], inplace = True)
dt['immigration'].replace(['y', 'n'], [1, 0], inplace = True)
dt['synfuels-corporation-cutback'].replace(['y', 'n'], [1, 0], inplace = True)
dt['education-spending'].replace(['y', 'n'], [1, 0], inplace = True)
dt['superfund-right-to-sue'].replace(['y', 'n'], [1, 0], inplace = True)

dt['crime'].replace(['y', 'n'], [1, 0], inplace = True)
dt['duty-free-exports'].replace(['y', 'n'], [1, 0], inplace = True)
dt['export-administration-act-south-africa'].replace(['y', 'n'], [1, 0], inplace = True)

dt = dt.dropna(axis = 0)
X = dt.drop('Class', axis=1)
y = dt['Class']



print(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=13)

### quantos vizinhos o algoritmo vai considerar

k = 5

clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(y_pred)
print(classification_report(y_test, y_pred,))   

