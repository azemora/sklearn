

from cProfile import label
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('lens.csv', delimiter=',')

df['contact-lenses'].replace(['none', 'soft', 'hard'], [0, 1, 2], inplace = True)


le = LabelEncoder()
df['tear-prod-rate'] = le.fit_transform(df['tear-prod-rate'])
df['spectacle-prescrip'] = le.fit_transform(df['spectacle-prescrip'])
df['astigmatism'] = le.fit_transform(df['astigmatism'])
df['tear-prod-rate'] = le.fit_transform(df['tear-prod-rate'])
df['age'] = le.fit_transform(df['age'])

X = df.drop('contact-lenses', axis=1)
y = df['contact-lenses']

print(y)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=15)

svcclassifier = SVC(kernel = 'linear')
svcclassifier.fit(X_train, y_train)


y_pred = svcclassifier.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred))
