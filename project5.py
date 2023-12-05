# Importing the Modules
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#Importing the data set.
data=pd.read_csv(r"D:\Git\Git-Projects\modifiedIris2Classes.csv")
#print(data)
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())
Y=data.iloc[:,-1]
X=data.iloc[:,:-1]
#print(Y)
#Training the data set.
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#print(x_train)
#Fitting to the algoritham.
model=LogisticRegression()
model.fit(x_train,y_train)
#predicting the model.
y_pred=model.predict(x_test)
#print(y_pred)
proba=model.predict_proba(x_test)
#print(proba)
co_eff=model.coef_
#print(co_eff)
inte_cept=model.intercept_
#print(inte_cept)
score=model.score(x_test,y_test)
print(score*100)