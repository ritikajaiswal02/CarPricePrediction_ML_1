#KNNRegressor

#import the lib
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#load the data
data = pd.read_csv("car_price_july25.csv")
print(data)


data.sort_values(by="kms_driven",inplace=True)

#check for null data
print(data.isnull().sum())


#x and y 
x = data["kms_driven"]
y = data["price"]


#features and target
features = data.drop(["price"],axis=1)
target = data["price"]

#categorical data
nfeatures = pd.get_dummies(features)

#feature scaling 
mms = MinMaxScaler()
sfeatures = mms.fit_transform(nfeatures)

#k - value
k = int(len(data)**0.5)
if k%2 == 0:
	k=k+1

#model
model = KNeighborsRegressor(n_neighbors=k)
model.fit(sfeatures,target)


#visualization
plt.scatter(x,y,color="red")
plt.plot(x,model.predict(sfeatures),color="purple")
plt.xlabel("kms_driven")
plt.ylabel("price")
plt.title("Used Car price prediction")
plt.show()

 
