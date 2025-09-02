# 🚗 Car Price Prediction – Machine Learning Project  

## 📖 Project Overview  
This project is about predicting the **price of used cars** based on features such as **kilometers driven, age and other attributes**.  
The goal is to help users or businesses estimate a fair market price for a car using **Machine Learning techniques**.  

The development was done in multiple steps, starting from **data visualization**, **preprocessing**, **model training**, and finally **deploying the trained model using Flask**.  
This project demonstrates the **end-to-end workflow of an ML project**, from dataset exploration to web deployment.  

---

## 📂 Project Structure 

```
carPricePrediction_ML_1/
│
├── s1_visualization/ # Step 1: Data visualization with Linear Regression
│ └── s1_visualization.py # Scatter plot + simple regression line
│
├── s2_polynomialR/ # Step 2: Polynomial Regression visualization
│ └── s2_polynomialR.py # Non-linear curve fitting (degree=3)
│
├── s3_knnR_visualization/ # Step 3: KNN Regressor implementation
│ └── s3_knnR_visualization.py # Model with feature scaling + KNN predictions (curve plotting)
│
├── s4_knnR/ # Step 4: KNN model evaluation with train-test split
│ └── s4_knnR.py # Prints training & testing accuracy scores
│
├── s5_decisionTreeR_visualization/ # Step 5: Decision Tree Regression visualization
│ └── s5_decisionTreeR_visualization.py # Visualizes tree structure for regression
│
├── s6_decisiontreeR/ # Step 6: Decision Tree Regressor evaluation
│ └── s6_decisiontreeR.py # Prints training & testing accuracy scores
│
├── s7_random_forestR/ # Step 7: Random Forest Regressor
│ └── s7_random_forestR/.py # Training & evaluation of Random Forest model
|
├── RFRmodel/                   # Step 8: Final Random Forest Regressor
│   └── RFRmodel.py          # Trains & saves rfr_model.pkl
|
├── model/                      # Step 9: Flask web deployment
│   ├── app.py                          # Flask app entry point
│   ├── templates/                      # HTML templates (UI)
│   │   └── index.html
│   └── rfr_model.pkl                   # Final saved ML model
│
├── requirements.txt # Dependencies (Flask, scikit-learn, pandas, matplotlib)
├── car_price_july25.csv # Dataset for training/testing
└── README.md # Project documentation
```

## 🚀 How to Run the Project
Run the Flask App - Go to the model folder and start the app:

cd ../model
python app.py

📂 You can view the presentation here:  
[Car Price Prediction – Project Presentation](./ML_Task1.pptx)

## 🙌 Acknowledgement  

This project helped me explore different Machine Learning regression models step by step, understand their performance, and finally deploy the best-performing model (**Random Forest Regressor**) using Flask.
