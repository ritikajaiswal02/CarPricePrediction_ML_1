#decision tree Visualization

import pandas as pd
from sklearn.tree import DecisionTreeRegressor,plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv("car_price_july25.csv")
print(data)

features = data.drop(["price"],axis=1)
target = data["price"]

nfeatures = pd.get_dummies(features)

model = DecisionTreeRegressor(max_depth=3)
md = model.fit(nfeatures.values,target)

plt.figure(figsize=(30,15))
plot_tree(md,filled=True,fontsize=10,feature_names=nfeatures.columns)
plt.show()


#class_names=md.classes_  is valid for classification problem