#Importing modules  and not used CT Method
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#Reading the data set
data=pd.read_csv(r"D:\Git\Git-Projects\User_Data.csv")
#print(data.head())
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())
X=data[['User ID','Age','EstimatedSalary']]
Y=data['Purchased']
#print(X)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#fitting the model
model=LogisticRegression()
model.fit(x_train,y_train)
#predicting the data..
y_pred=model.predict(x_test)
#probablity
prob=model.predict_proba(x_test)
#coefficiency
co_eff=model.coef_
#print(co_eff)
#intercept
inter=model.intercept_
#print(inter)
#Checking the score
score=model.score(x_test,y_test)
print(score*100)