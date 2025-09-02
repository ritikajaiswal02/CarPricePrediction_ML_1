#RandomForestRegressor Model

#import the lib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pickle import dump

#load the data
data = pd.read_csv("car_price_july25.csv")

#check for null data
print(data.isnull().sum())

#features and target
features = data.drop(["price"],axis=1)
target = data["price"]

#handle categorical data
nfeatures = pd.get_dummies(features)

#train and testing
x_train,x_test,y_train,y_test = train_test_split(nfeatures,target,random_state=3)

#model creation
model = RandomForestRegressor(n_estimators=200,random_state=3)
model.fit(x_train,y_train)

#saving the model
f = open("rfr_model.pkl","wb")
dump(model,f)
f.close()

