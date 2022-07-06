

import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.neighbors import DistanceMetric

dt = pd.read_csv('credito.csv', delimiter=',')


dt['class'].replace(['good', 'bad'], [1, 0], inplace = True)

distance =DistanceMetric.get_metric("manhattan")

le = LabelEncoder()
dt['checking_status'] = le.fit_transform(dt['checking_status'])
dt['property_magnitude'] = le.fit_transform(dt['property_magnitude'])
dt['credit_history'] = le.fit_transform(dt['credit_history'])
dt['savings_status'] = le.fit_transform(dt['savings_status'])
dt['employment'] = le.fit_transform(dt['employment'])

print(dt)

X = dt.drop('class', axis=1)
y = dt['class']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state= 13)

### quantos vizinhos o algoritmo vai considerar

k = 5

clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred,))
print(y_pred)
