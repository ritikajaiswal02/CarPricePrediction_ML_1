import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("car_price_july25.csv")
print(data)

print(data.isnull().sum())

features = data.drop(["price"],axis=1)
target = data["price"]

nfeatures = pd.get_dummies(features)
print(nfeatures)

x_train,x_test,y_train,y_test = train_test_split(nfeatures.values,target,random_state=3)

model=RandomForestRegressor(n_estimators=200,random_state=3)
model.fit(x_train,y_train)

'''
name = int(input("1-Mahindra	2-Maruti	3-Tata"))
if name == 1:
	d1 = [1,0,0]
elif name == 2:
	d1=[0,0,1]
else:
	d1=[0,0,1]

age = int(input("Enter age :"))

driven = int(input("ENter driven"))

d = [[age,driven]+d1]
res = model.predict(d)
print(res)
'''


print("\n\tRandom Forest Regressor\n")
tr_score=model.score(x_train,y_train)
te_score=model.score(x_test,y_test)

print("********Training Score********")
print(round(tr_score*100,2),"%")
print("\n")
print("********Testing Score********")
print(round(te_score*100,2),"%")


"""
******************Training*******************
99.6061205740592
******************Testing********************
99.59966997577703
"""

