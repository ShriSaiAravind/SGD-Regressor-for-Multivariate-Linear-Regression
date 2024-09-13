import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd


data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["HousePrice"] = data.target

X= df.drop(["AveOccup", "HousePrice","Latitude","Longitude"],axis=1) 
Y= df[["AveOccup", "HousePrice"]]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

sgd = SGDRegressor(max_iter = 1000,tol = 1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(x_train,y_train)

y_pred = multi_output_sgd.predict(x_test)

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print(y_pred[:8,:])