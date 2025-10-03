import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.data_generator import data_collection, split_data

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def nonlinear_regression_metrics(data_path, features, targets, degrees=range(1, 21), n_samples=100):
    # Load dataset
    df_features_targets = data_collection(data_path)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(df=df_features_targets.head(n_samples), features=features, targets=targets)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    results = []

    fig, ax = plt.subplots(1, sharey=False, figsize=(12, 5))

    # Loop through polynomial degrees
    for degree in degrees:
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression(fit_intercept=True)

        # Transform features
        poly_features = polynomial_features.fit_transform(X_train.reshape(-1, 1))
        poly_features_test = polynomial_features.transform(X_test.reshape(-1, 1))

        # Fit model
        linear_regression.fit(poly_features, y_train)

        # Predict
        y_pred = linear_regression.predict(poly_features_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r_squared = r2_score(y_test, y_pred)

        # Store metrics in results
        results.append({
            'Degree': degree,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r_squared
        })

    results_df = pd.DataFrame(results)

    # Plot results for all degrees
    ax.plot(results_df['Degree'], results_df['RMSE'], label='RMSE', marker='o')
    ax.plot(results_df['Degree'], results_df['MAE'], label='MAE', marker='x')
    ax.plot(results_df['Degree'], results_df['R²'], label='R²', marker='s')
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Metrics")
    ax.set_title("Evaluation Metrics")
    ax.legend()

    plt.tight_layout()
    plt.suptitle("Non-linear Regression Metrics", fontsize=14, y=1.02)
    plt.show()

    # Identify best model
    best_rmse_degree = results_df.loc[results_df['RMSE'].idxmin()]
    best_mae_degree = results_df.loc[results_df['MAE'].idxmin()]
    best_r_squared_degree = results_df.loc[results_df['R²'].idxmax()]

    best_degrees = pd.DataFrame([best_rmse_degree, best_mae_degree, best_r_squared_degree]).drop_duplicates(subset=['Degree'])

    print("\nBest Polynomial Degrees based on RMSE, MAE, and R² (including Degree 1):")
    print(best_degrees)

path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')
nonlinear_regression_metrics(data_path=path_to_data, features=['mean_wind_speed'], targets=['Kalby_AP'])
