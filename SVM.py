import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score,classification_report
print(df.head())
x=df.drop(['Id','Species'],axis=1)
y=df['Species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratif
y=y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
Standard=StandardScaler()
x_train=Standard.fit_transform(x_train)
x_test=Standard.fit_transform(x_test)
model=svm.SVC(kernel='rbf')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
