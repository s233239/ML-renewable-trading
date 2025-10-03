import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.data_generator import data_collection, split_data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def weighted_regression_model2(data_path, features, target, bandwidth=1, n_samples=100):
    # Load the prepared dataset
    data_path = os.path.join(path_to_data, "model2_dataset.csv")
    data = pd.read_csv(data_path)

    # Check for NaN values in the target variable
    if data[target].isnull().any():
        data = data.dropna(subset=[target])  # Remove rows with NaN in target

    # Check for NaN values in features
    if data[features].isnull().any().any():
        data = data.dropna(subset=features)  # Remove rows with NaN in features

    # Define features and target 
    X = data[features]
    y = data[target]

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df=data.head(n_samples), features=features, targets=[target])
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy()

    def epanechnikov(t):
        res = np.zeros_like(t)
        res[np.abs(t) <= 1] = 0.75 * (1 - t[np.abs(t) <= 1] ** 2)
        return res

    # Initialise array 
    y_pred_wls = np.zeros(len(X_test))

    # Weighted regression
    for i in range(len(X_test)):
        # Distance from the i-th test point to training points
        distances = np.linalg.norm(X_train - X_test[i], axis=1)

        # Weights calculation
        W = np.diagflat(epanechnikov(distances / bandwidth))

        # Calculate coefficients
        theta = np.linalg.inv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train

        # Make prediction
        y_pred_wls[i] = (X_test[i] @ theta).item()

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_wls))
    mae = mean_absolute_error(y_test, y_pred_wls)
    r_squared = r2_score(y_test, y_pred_wls)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label="True Values", color='blue')
    plt.plot(range(len(y_pred_wls)), y_pred_wls, label="Weighted Model Predictions", color='red')
    plt.scatter(range(len(y_train)), y_train, lw=1, s=30, label="Training Samples", color='green')
    plt.xlabel("Sample Index")
    plt.ylabel("Optimal Bid")
    plt.title(f"Weighted Regression\nRMSE: {rmse:.5f}\nMAE: {mae:.5f}\nR²: {r_squared:.5f}", fontsize=10)
    plt.legend()
    plt.show()

path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')
weighted_regression_model2(data_path=path_to_data, 
                    features=['SpotPriceDKK', 'BalancingPowerPriceDownDKK', 'mean_wind_power'], 
                    target='optimal_bid', 
                    bandwidth=1, 
                    n_samples=100)
