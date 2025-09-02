#Polynomial Regression visualization

#step1: import the lib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

data = pd.read_csv("car_price_july25.csv")
print(data)
data.sort_values(by="kms_driven",inplace=True)

x = data["kms_driven"]
y = data["price"]

features = data.drop(["price"],axis=1)
target = data["price"]

nfeatures = pd.get_dummies(features)

pf = PolynomialFeatures(degree=3)
pfeature = pf.fit_transform(nfeatures.values)

model = LinearRegression()
model.fit(pfeature,target)

plt.scatter(x,y,color="red")
plt.xlabel("kms_driven")
plt.ylabel("price")
plt.title("Used Car price prediction")
plt.plot(x,model.predict(pfeature),color="purple")
plt.show()

#the visualization of curve is not proper not joining the points