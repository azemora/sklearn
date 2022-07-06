

from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
df = pd.read_csv('weathernumeric.csv', delimiter=',')


df['play'].replace(['yes', 'no'], [1, 0], inplace = True)
le = LabelEncoder()
df['windy'] = le.fit_transform(df['windy'])
df['outlook'] = le.fit_transform(df['outlook'])

X = df.drop('play', axis=1)
y = df['play']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=10)

svcclassifier = SVC(kernel = 'sigmoid')
svcclassifier.fit(X_train, y_train)

warnings.filterwarnings('ignore')
y_pred = svcclassifier.predict(X_test)
print(classification_report(y_test, y_pred))

