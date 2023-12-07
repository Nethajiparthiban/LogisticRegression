import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv(r"D:\Git\Git-Projects\bank-additional-full.csv")

cat_col = ['marital', 'contact', 'poutcome']
ord_col = ['job', 'education', 'default', 'housing', 'loan', 'month', 'day_of_week']
othe_col = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate',
            'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Convert 'y' to numeric values
data['y'].replace(['yes', 'no'], [1, 0], inplace=True)

# Separate features (X) and target variable (Y)
X = data[cat_col + ord_col + othe_col]
Y = data['y']

# Create ColumnTransformer
ct = ColumnTransformer(transformers=[
    ('Cat_encode', OneHotEncoder(), cat_col),
    ('ord_encode', OrdinalEncoder(), ord_col),
    ('other_encode', OrdinalEncoder(), othe_col)
], remainder='passthrough')

# Transform features
x = ct.fit_transform(X)
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,random_state=0)
#fitting to algoritham
stud=LogisticRegression(max_iter=10000)
stud.fit(x_train,y_train)
#prediction
y_pred=stud.predict(x_test)
#co_eff=stud.coef_
#inte_cept=stud.intercept_
score=stud.score(x_test,y_test)
print(score*100)
