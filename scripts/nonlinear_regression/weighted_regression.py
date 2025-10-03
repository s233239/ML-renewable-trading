import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.data_generator import data_collection, split_data

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def weighted_regression(data, features, targets, bandwidth=1, n_samples=100):
    
    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df=data.head(n_samples), features=features, targets=targets)
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
        # Distance from i-th test point to training points
        distances = np.linalg.norm(X_train - X_test[i], axis=1)

        # Weights calculation
        W = np.diagflat(epanechnikov(distances / bandwidth))

        # Calculate coefficients
        theta = np.linalg.inv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train

        # Make prediction
        y_pred_wls[i] = (X_test[i] @ theta).item()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_wls))
    mae = mean_absolute_error(y_test, y_pred_wls)
    r_squared = r2_score(y_test, y_pred_wls)

    # Plotting
    plt.plot(X_test[:,0], y_test, label="True Values", color='blue')
    plt.plot(X_test[:,0], y_pred_wls, label="Weighted Model Predictions", color='red')
    plt.scatter(X_train[:,0], y_train, lw=1, s=30, label="Training Samples", color='green')
    plt.xlabel(f"{features[0]}")
    plt.ylabel(f"{targets[0]}")
    plt.title(f"Weighted Regression\nRMSE: {rmse:.5f}\nMAE: {mae:.5f}\nR²: {r_squared:.5f}", fontsize=10)
    plt.legend()
    plt.show()

    return y_pred_wls

# path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')
# data = data_collection(path_to_data)
# weighted_regression(data=data, features=['mean_wind_speed','max_wind_speed_10min','max_wind_speed_3sec'], targets=['Kalby_AP'], bandwidth=1, n_samples=100)


# path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')
# data_path = os.path.join(path_to_data, "model2_dataset.csv")
# data = pd.read_csv(data_path)
# features=['SpotPriceDKK', 'BalancingPowerPriceDownDKK', 'mean_wind_power'] 
# targets='optimal_bid'
# if data[targets].isnull().any():
#     data = data.dropna(subset=[targets])  # Remove rows with NaN in target
# # Check for NaN values in features
# if data[features].isnull().any().any():
#     data = data.dropna(subset=features)  # Remove rows with NaN in features
# weighted_regression(data=data, 
#                     features=features, 
#                     targets=targets, 
#                     bandwidth=1, 
#                     n_samples=100)