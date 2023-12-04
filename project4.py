import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split  # Import the train_test_split function
data = pd.read_csv(r"D:\Git\Git-Projects\advertising.csv")
Y = data.iloc[:, -1]
Cat_col = ['Country','City']
ord_col = ['Ad Topic Line','Timestamp']
other_col = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']
X = data[Cat_col + ord_col + other_col]
ct=ColumnTransformer(transformers=[('Cat_encoder',OneHotEncoder(),Cat_col),
                                   ('other_encoder',OrdinalEncoder(),ord_col)],remainder='passthrough')
x=ct.fit_transform(X)
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=0)
#print(x_test)
#print(x_train)
#print(x_train)
#fitting the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=10000)
model.fit(x_train,y_train)
#Predicting
y_pred=model.predict(x_test)
#print(y_pred)
#Checking probablity
prob=model.predict_proba(x_test)
#print(prob)
#Coefficient
coff=model.coef_
#print(coff)
#intercept
inter_cept=model.intercept_
#print(inter_cept)
#Score
score=model.score(x_test,y_test)
print((score*100).round(),'%')