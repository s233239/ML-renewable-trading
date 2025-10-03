import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def nonlinear_regression_model2(path_to_data=os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')):
    # Load the prepared dataset
    data_path = os.path.join(path_to_data, "model2_dataset.csv")
    data = pd.read_csv(data_path)

    # Define features and target
    features = [
        'SpotPriceDKK',
        'BalancingPowerPriceDownDKK',
        'mean_wind_power'
    ]
    
    target = 'optimal_bid'  # The target is the optimal offering strategy

    # Check for NaN values in the target variable
    if data[target].isnull().any():
        data = data.dropna(subset=[target])  # Remove rows with NaN in target

    # Check for NaN values in features
    if data[features].isnull().any().any():
        data = data.dropna(subset=features)  # Remove rows with NaN in features

    # Define features and target again after cleaning
    X = data[features]
    y = data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    degrees = range(1, 21)  # Polynomial degrees to evaluate
    results = []

    fig, ax = plt.subplots(figsize=(12, 5))

    # Loop through polynomial degrees
    for degree in degrees:
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = polynomial_features.fit_transform(X_train)
        X_test_poly = polynomial_features.transform(X_test)

        # Initialize and train the regression model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Make predictions
        y_pred = model.predict(X_test_poly)

        # Evaluate the model using RMSE, MAE, and R-squared metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store metrics in results
        results.append({
            'Degree': degree,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        })

    results_df = pd.DataFrame(results)

    # Plot results for all degrees
    ax.plot(results_df['Degree'], results_df['RMSE'], label='RMSE', marker='o')
    ax.plot(results_df['Degree'], results_df['MAE'], label='MAE', marker='x')
    ax.plot(results_df['Degree'], results_df['R²'], label='R²', marker='s')
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Metrics")
    ax.set_title("Evaluation Metrics for Non-linear Regression")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Identify best model based on each metric
    best_rmse_degree = results_df.loc[results_df['RMSE'].idxmin()]
    best_mae_degree = results_df.loc[results_df['MAE'].idxmin()]
    best_r_squared_degree = results_df.loc[results_df['R²'].idxmax()]

    best_degrees = pd.DataFrame([best_rmse_degree, best_mae_degree, best_r_squared_degree]).drop_duplicates(subset=['Degree'])

    print("\nBest Polynomial Degrees based on RMSE, MAE, and R²:")
    print(best_degrees)

nonlinear_regression_model2()
