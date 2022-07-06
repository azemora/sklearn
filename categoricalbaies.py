
from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB


dt = pd.read_csv('glass.csv', delimiter=',', na_values='?')
dt = dt.dropna(axis = 0)
le = LabelEncoder()

X = dt.drop('Type', axis=1)
y = dt['Type']


print(dt)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=5)

glass = CategoricalNB()
glass.fit(X_train, y_train)
import warnings
warnings.filterwarnings('ignore')

y_pred = glass.predict(X_test)
print(accuracy_score(y_test,y_pred))
print("##########")
print(y_pred)
print("##########")
print(classification_report(y_test, y_pred))


'''
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=17)

trevas = CategoricalNB()
trevas.fit(X_train, y_train)

#svcclassifier = SVC(kernel = 'linear')
#svcclassifier.fit(X_train, y_train)

y_pred = trevas.predict(X_test)
print(accuracy_score(y_test,y_pred))
print("##########")
print(y_pred)
print("##########")
print(classification_report(y_test, y_pred))


'''




'''

#### CATEGORICAL COM DATASETCANCER


from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
df = pd.read_csv('cancer.csv', delimiter=',')

print(df)
le = LabelEncoder()
df['menopause'] = le.fit_transform(df['menopause'])
df['age'] = le.fit_transform(df['age'])
df['tumor-size'] = le.fit_transform(df['tumor-size'])
df['inv-nodes'] = le.fit_transform(df['inv-nodes'])
df['node-caps'] = le.fit_transform(df['node-caps'])
df['breast'] = le.fit_transform(df['breast'])
df['breast-quad'] = le.fit_transform(df['breast-quad'])
df['irradiat'] = le.fit_transform(df['irradiat'])
df['Class'].replace(['recurrence-events', 'no-recurrence-events'], [1, 0], inplace = True)
print('##############')
print(df)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=17)

trevas = CategoricalNB()
trevas.fit(X_train, y_train)

#svcclassifier = SVC(kernel = 'linear')
#svcclassifier.fit(X_train, y_train)

y_pred = trevas.predict(X_test)
print(accuracy_score(y_test,y_pred))
print("##########")
print(y_pred)
print("##########")
print(classification_report(y_test, y_pred))

'''