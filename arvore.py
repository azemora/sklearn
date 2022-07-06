

#### ARVORE
#### WEATHERNOMINAL

from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import accuracy_score

df = pd.read_csv('weathernominal.csv', delimiter=',')

df['play'].replace(['yes', 'no'], [1, 0], inplace = True)

#print(X.outlook.unique)

df1 = df.outlook.map({'sunny': 0, 'overcast':1, 'rainy':2 })
#print(df1.head())

le = LabelEncoder()
df['temperature'] = le.fit_transform(df['temperature'])
df['humidity'] = le.fit_transform(df['humidity'])
df['windy'] = le.fit_transform(df['windy'])
df['outlook'] = le.fit_transform(df['outlook'])

X = df.drop('play', axis=1)
y = df['play']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

arvoresomosnozes = tree.DecisionTreeClassifier(criterion='gini', max_depth= 5)
arvoresomosnozes = arvoresomosnozes.fit(X_train, y_train)
predicao = arvoresomosnozes.predict(X_test)

print(predicao)

