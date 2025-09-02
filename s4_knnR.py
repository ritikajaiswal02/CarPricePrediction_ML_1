#import the lib
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#load the data
data = pd.read_csv("car_price_july25.csv")
print(data)

#check for null data
print(data.isnull().sum())

#features and target
features = data.drop(["price"],axis=1)
target = data["price"]

#handle categorical data
nfeatures = pd.get_dummies(features)

#feature scaling
mms = MinMaxScaler()
sfeatures = mms.fit_transform(nfeatures)

#k value
k = int(len(data)**0.5)
if k%2 == 0:
	k=k+1

#training and testing performance
x_train,x_test,y_train,y_test = train_test_split(sfeatures,target,random_state=2)

#model creation
model = KNeighborsRegressor(n_neighbors=k)
model.fit(x_train,y_train)

tr_score = model.score(x_train,y_train)
te_score = model.score(x_test,y_test)

print("\n\tKNN Regression\n")
print("********Training Score********")
print(round(tr_score*100,2),"%")
print()
print()
print("********Testing Score********")
print(round(te_score*100,2),"%")


"""
********Training Score************
96.42997627811903
*********Testing Score*************
96.39089537155374
"""


