import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def nonlinear_regression_model2(data_path, features, target, degrees=[1, 2, 3], n_samples=100):
    # Load the prepared dataset
    data_path = os.path.join(data_path, "model2_dataset.csv")
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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    fig, axs = plt.subplots(1, len(degrees), figsize=(15, 5))

    for ax, degree in zip(axs.flatten(), degrees):
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression()

        # Transform features
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.transform(X_test)

        # Fit the model
        linear_regression.fit(X_train_poly, y_train)

        # Predict
        y_pred = linear_regression.predict(X_test_poly)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)

        # Plotting
        ax.scatter(y_test, y_pred, color='orange')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Degree: {degree}\nRMSE: {rmse:.5f}\nMAE: {mae:.5f}\nR²: {r_squared:.5f}")

    plt.tight_layout()
    plt.suptitle("Non-linear Regression using Polynomial Features", fontsize=16)
    plt.show()

path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')
nonlinear_regression_model2(data_path=path_to_data, 
                           features=['SpotPriceDKK', 'BalancingPowerPriceDownDKK', 'mean_wind_power'], 
                           target='optimal_bid', 
                           degrees=[1, 4, 5])  
