import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from scripts.data_collection.data_generator import data_collection, plot_feature_correlation, split_data, data_generation_model2
from scripts.linear_regression.linear_regression import linear_regression, closed_form
from scripts.nonlinear_regression.nonlinear_regression import nonlinear_regression
from scripts.nonlinear_regression.weighted_regression import weighted_regression
from scripts.regularization.regularization import regularization
from scripts.regularization.regularization_model2 import regularization_model2
from scripts.optimization.bid_optimization import bid_optimization
from scripts.optimization.revenue_calc import revenue_calculation, revenue_calculation_model2

def run():
    """
    Runs every relevant program for assignment 1.
    """
    # Specify path to 'data', this is currently done automatically 
    # based on the location of the executed file
    path_to_data = os.path.join(os.path.abspath(os.path.join(__file__,'..')), 'data')
    
    # Collect features and targets using data_collection function
    # If more features are decided, the changes should occur in the function
    data = data_collection(path_to_data)
    
    # Plot the correlation between our features and targets
    plot_feature_correlation(data)
    
    # Set duration of our study
    dates_start = '2021-01-01 00:00:00'
    dates_end = '2022-01-01 23:00:00'
    data_model1 = data.loc[dates_start:dates_end]
    num_rows = data_model1.shape[0]
    print(f'\nModels are studied between {dates_start} and {dates_end} for {num_rows} hourly data.\n')

    # Get training and testing sets
    X_train, X_test, y_train, y_test = split_data(df=data_model1, features=['mean_wind_speed','max_wind_speed_10min','max_wind_speed_3sec'], targets=['Kalby_AP'])

    # Run linear_regression script
    linear_regression(path_to_data)

    # Run regularization script
    y_predict_lin, y_predict_lin_L1, y_predict_lin_L2, y_predict_nl, y_predict_nonlin_L1, y_predict_nonlin_L2 = regularization(
        data=data_model1,
        model='model1',
        features=['mean_wind_speed','max_wind_speed_10min','max_wind_speed_3sec'],
        target=['Kalby_AP'], 
        plots = True,
        _lambda_opt = False)

    # Run revenue calculation script
    methods = ['Linear','Non-linear', 'Linear L1','Linear L2','Non-linear L1','Non-linear L2']
    y_predictions = [y_predict_lin, y_predict_nl, y_predict_lin_L1, y_predict_lin_L2, y_predict_nonlin_L1, y_predict_nonlin_L2]

    revenue_table = revenue_calculation(y_predictions, methods, data_model1, plots=False)

    # Find the row with the maximum revenue
    revenue_table['Actual Revenue'] = pd.to_numeric(revenue_table['Actual Revenue'])
    max_revenue_row = revenue_table.loc[revenue_table['Actual Revenue'].idxmax()]
    max_revenue_model = max_revenue_row['Model']
    print(f"\nModel with Max Revenue: {max_revenue_model}")
    print(f"Max Actual Revenue: {max_revenue_row['Actual Revenue']} DKK")
    

    # Model 2 -----------------------------------------------------------------------
    print('\n MODEL 2 RESULTS \n')

    features_model2 = ['SpotPriceDKK','BalancingPowerPriceDownDKK','BalancingPowerPriceUpDKK','mean_wind_power']
    target_model2 = ['optimal_bid']

    optimal_predictions = y_predict_lin_L2
    dates_model2 = data_model1.tail( len(optimal_predictions) ).index

    """
    # Generate the new data
    optimal_bids = bid_optimization(
        data=data_model1, 
        dates=dates_model2,
        y=optimal_predictions,
        plots=False)[0]
    
    optimal_bids = pd.DataFrame(
        data=optimal_bids,
        index=dates_model2,
        columns=['optimal-bid']
    )

    optimal_bids.to_csv('optimal_bids_v2.csv')
    """

    path_to_data_model2 = path_to_data
    data_model2 = data_generation_model2(path_to_data_model2, features_model2, target_model2)

    dates_start = data_model2['ts'].iloc[0]
    dates_end = data_model2['ts'].iloc[-1]
    num_rows = data_model2.shape[0]
    print(f'\nModels are studied between {dates_start} and {dates_end} for {num_rows} hourly data.\n')

    
    X_train, X_test, y_train, y_test = split_data(df=data_model2, features=features_model2, targets=target_model2)

    y_predict_lin_model2, y_predict_lin_L1_model2, y_predict_lin_L2_model2, y_predict_nl_model2, y_predict_nonlin_L1_model2, y_predict_nonlin_L2_model2 = regularization(
        data=data_model2, 
        model='model2',
        features = features_model2,
        target = target_model2,
        plots = True,
        _lambda_opt = False)

    methods_model2 = methods
    y_predictions_model2 = [y_predict_lin_model2, y_predict_nl_model2, y_predict_lin_L1_model2, y_predict_lin_L2_model2, y_predict_nonlin_L1_model2, y_predict_nonlin_L2_model2]

    revenue_calculation_model2(y_predictions_model2, methods_model2, data_model2, plots=False)


    return

if __name__ == '__main__':
    run()