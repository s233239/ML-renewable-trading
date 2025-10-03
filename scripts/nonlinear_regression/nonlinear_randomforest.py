# this is just a trial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.data_generator import data_collection, split_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..','..','..')), 'data')
df_features_targets = data_collection(path_to_data)

# Split dataset
X_train, X_test, y_train, y_test = split_data(df=df_features_targets.head(100), features=['mean_wind_speed'], targets=['Kalby_AP'])
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Initialize and fit the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="True", color='green')
plt.plot(y_pred, label="Model Predictions", color='red')
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.title(f"Random Forest Regression | MSE: {mse:.5f}")
plt.legend()
plt.show()
