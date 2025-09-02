#import the lib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("car_price_july25.csv")
print(data)

features = data.drop(["price"],axis=1)
target = data["price"]

nfeatures = pd.get_dummies(features)

x_train,x_test,y_train,y_test = train_test_split(nfeatures,target,random_state=7)

model = DecisionTreeRegressor(random_state=7)
model.fit(x_train,y_train)

tr_score= model.score(x_train,y_train)
te_score= model.score(x_test,y_test)

print("\n\tDecision Tree Regressor\n")
print("********Training Score********")
print(round(tr_score*100,2),"%")
print("\n")
print("********Testing Score********")
print(round(te_score*100,2),"%")


"""
*************Training**************
100.0
*************Testing***************
98.72708927061477
"""
