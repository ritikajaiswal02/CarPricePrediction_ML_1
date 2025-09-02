#Plotting the Line of regression and Visualization of scatter points

#step1: import the lib
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("car_price_july25.csv")
print(data)

x = data["kms_driven"]
y = data["price"]

features=data.drop(["price"],axis=1)
target = data["price"]

nfeatures = pd.get_dummies(features)

model = LinearRegression()
model.fit(nfeatures.values,target)

plt.scatter(x,y,color="red")
plt.xlabel("kms_driven")
plt.ylabel("price")
plt.title("Used Car price prediction")
#plt.plot(x,model.predict(nfeatures),color="purple")
plt.show()


#after this we identified that it is Nonlinear