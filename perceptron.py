from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

df = pd.read_csv('hypothyroid.csv', delimiter=',', na_values='?')


le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
df['sick'] = le.fit_transform(df['sick'])
df['goitre'] = le.fit_transform(df['goitre'])
df['TSH'] = le.fit_transform(df['TSH'])
#df['T3m'] = le.fit_transform(df['T3m'])
df['T3m'].replace(['t', 'f'], [1, 0], inplace = True)
df = df.dropna()
print(df)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

clf = Perceptron()
clf.fit(X_train,y_train)

#svcclassifier = SVC(kernel = 'linear')
#svcclassifier.fit(X_train, y_train)
import warnings
warnings.filterwarnings('ignore')
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print("##########")
print(y_pred)
print("##########")
print(classification_report(y_test, y_pred))

'''
from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

clf = Perceptron()
clf.fit(X_train,y_train)

#svcclassifier = SVC(kernel = 'linear')
#svcclassifier.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred))
print("##########")
print(y_pred)
print("##########")
print(classification_report(y_test, y_pred))

'''