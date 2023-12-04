import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

df=pd.read_csv(r"D:\Git\Git-Projects\birnary.csv")
#print(df.head(5))
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
#print(Y)
#Plotting the data set
#plt.scatter(X,Y)
#plt.show()
#splitting the data set for Training.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
#Fitting the data set in logistic model.
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
#Predicting the dta set
y_pred=model.predict(x_test)
#print(y_pred)
#Predicting the probablity
prob=model.predict_proba(x_test)
#print(prob)
#Checking the score
score=model.score(x_test,y_test)
co_eff=model.coef_
#print(co_eff)
intercept=model.intercept_
#print(intercept)
print(score.round(),'%')
