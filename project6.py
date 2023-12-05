#Used Column Transformer method...
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#Reading the dataset
data=pd.read_csv(r"D:\Git\Git-Projects\User_Data.csv")
#print(data)
Y=data.iloc[:,-1]
cat_col=['Gender']
other=['User ID','Age','EstimatedSalary']
X=data[cat_col+other]
#print(X)
ct=ColumnTransformer(transformers=[('Cat_col',OneHotEncoder(),cat_col)],remainder='passthrough')
x=ct.fit_transform(X)
#print(x)
#Training the data.
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=0)
#print(x_train)
#fitting to algoritham
model=LogisticRegression()
model.fit(x_train,y_train)
#Predicting the data set
y_pred=model.predict(x_test)
#print(y_pred)
#Checking the probablity
prob=model.predict_proba(x_test)
#print(prob)
co_eff=model.coef_
#print(co_eff)
in_cept=model.intercept_
#print(in_cept)
#Checking the score..
score=model.score(x_test,y_test)
print(score*100)