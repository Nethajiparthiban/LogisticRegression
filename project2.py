import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



#Reading the data set.
data=pd.read_csv(r"D:\Git\Git-Projects\data.csv")
#print(data)
data['diagnosis'].replace(['M','B'],[1,0],inplace=True)

Y=data['diagnosis']
X=data[['id','radius_mean','texture_mean','perimeter_mean','area_mean',
        'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
        'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se',
        'area_se','smoothness_se','compactness_se','concavity_se','concave points_se',
        'symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst',
        'area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
        'symmetry_worst','fractal_dimension_worst']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
model=LinearRegression()
model.fit(x_train,y_train)
#prediction..
y_pred=model.predict(x_test)
#print(y_pred)
#Accuracy
co_eff=model.coef_
int_cep=model.intercept_
#print(co_eff,int_cep)
score=model.score(x_test,y_test)
print(score.round(),'%')