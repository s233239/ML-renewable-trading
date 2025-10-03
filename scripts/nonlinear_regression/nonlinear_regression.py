import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.data_generator import data_collection, split_data, data_generation_model2

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def nonlinear_regression(data, features, targets, degrees=[1, 4, 13], n_samples=100, plots=True):
    '''
    Args:
        features (list of string): Must specify the principal feature as first in the list: features[0].
        targets (list of string):
        degrees (list): Needs at least to input two degrees.

    Returns:
        (poly_features_train, poly_features_test, y_pred)
    '''
    # Split dataset
    X_train_df, X_test, y_train, y_test = split_data(df=data.head(n_samples), targets=targets, features=features)
    X_train = X_train_df.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    fig, axs = plt.subplots(1, len(degrees), sharey=False, figsize=(12, 4))

    for ax, degree in zip(axs.flatten(), degrees):
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression(fit_intercept=True)

        # Transform features
        poly_features_train = polynomial_features.fit_transform(X_train)
        poly_features_test = polynomial_features.transform(X_test)

        # Fit model
        linear_regression.fit(poly_features_train, y_train)

        # Predict
        y_pred = linear_regression.predict(poly_features_test)

        # Get the weights (coefficients) associated with the predictions
        weights = linear_regression.coef_
        feature_weights = pd.DataFrame(weights.T, columns=['Weights'])
        if plots:
            print(f"Feature weights for polynomial degree {degree}:\n", feature_weights)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)

        # Plotting
        ax.plot(range(len(y_test)), y_test, label="True Values", color='blue')
        ax.plot(range(len(y_pred)), y_pred, label="Predicted Values", color='orange')
        ax.scatter(X_train[:,0], y_train, lw=1, s=30, label="Training Samples", color='green')
        ax.set_xlabel(f"{features[0]}")
        ax.set_ylabel(f"{targets[0]}")
        ax.legend(fontsize=10)
        ax.set_title(f"Polynomial Degree: {degree}\nRMSE: {rmse:.5f}\nMAE: {mae:.5f}\nR²: {r_squared:.5f}", fontsize=10)

    plt.suptitle("Non-linear Regression using Polynomial Features", fontsize=14, y=1)
    plt.tight_layout()
    plt.show()

    return poly_features_train, poly_features_test, y_pred, feature_weights


# path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')

# # Model 1
# data = data_collection(path_to_data)
# nonlinear_regression(data=data, features=['mean_wind_speed'], targets=['Kalby_AP'], degrees=[1, 4, 13])

# # Model 2
# features=['SpotPriceDKK', 'BalancingPowerPriceDownDKK', 'mean_wind_power']
# targets=['optimal_bid']
# data = data_generation_model2(path_to_data, features, targets)
# nonlinear_regression(data=data,
#                      features=features, 
#                      targets=targets, 
#                      degrees=[1, 4, 5]) 