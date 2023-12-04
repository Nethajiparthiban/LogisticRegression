import pandas as pd

#Reading the data set.
data=pd.read_csv(r"D:\Git\Git-Projects\nba_logreg.csv")
#print(data.head())
#print(data.info())
#print(data.describe())
#Filling the Null values
mean=data['3P%'].mean()
data.fillna(mean,axis=0,inplace=True)
#print(data.isnull().sum())
X=data.iloc[:,1:-1].values
Y=data.iloc[:,-1].values
#print(X)
#print(Y)
#Training the data set...
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#print(len(x_train))
#fitting to the model..
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=10000)
model.fit(x_train,y_train)
#prediction
y_pred=model.predict(x_test)
#print(y_pred)
#Coeff
co_eff=model.coef_
#print(co_eff)
#intercept
int_cep=model.intercept_
#print(int_cep)
#problity...
prob=model.predict_proba(x_test)
#print(prob)
#Accuracy
score=model.score(x_test,y_test)
print(score)

