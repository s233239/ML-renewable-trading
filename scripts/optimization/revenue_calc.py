import pandas as pd
import numpy as np
from optimization.bid_optimization import bid_optimization

def calc_revenue(data, dates, bids_DA):
    """
    Gets the actual revenue from production.
    """
    data = data.reset_index()  # 'ts' will now be a column if it was the index
    data = data.loc[data['ts'].isin(dates)]

    price_DA, price_up, price_down = list(data['SpotPriceDKK']), list(data['BalancingPowerPriceUpDKK']), list(data['BalancingPowerPriceDownDKK'])
    actual_power = list(data["Kalby_AP"])

    TIME = range(len(dates))

    revenue = 0

    for t in TIME:
        revenue +=  float(price_DA[t])*bids_DA[t] + float(price_up[t])*max((actual_power[t]-bids_DA[t]),0) + float(price_down[t])*(max((actual_power[t]-bids_DA[t]),0))

    return revenue


def revenue_calculation(y_prediction, method:str, data, plots = True):
    '''
    Computes the assignment step on revenue calculation for evaluation.

    Args:
        y_prediction (dataframe):   Contains 1 column corresponding to the model predictions  
            (for any method).
        method (string):    Associated method of prediction.
        data (dataframe):   Index are the timestamps. Contains many columns corresponding to 
            all the relevant data for our project.
        model (string):     Either 'model1' or 'model2'.
        plots (boolean):    Print the plots if set to True, as default.
    
    Returns:
        revenue_table (float, float): Revenues based on the optimized 
            bids of the day-ahead market & the power production predictions (input) and realizations (data).
    '''
    print('\nRevenue calculation results:\n')

    # Table to store the actual revenue for each model
    revenue_table = pd.DataFrame(columns = ['Model','Expected Revenue','Actual Revenue'])
    revenue_table['Model'] = method

    # Get the dates corresponding to the power productions (index of the last 20% of our data)
    num_rows = int(0.2 * len(data))
    testing_data = data.tail(num_rows)
    dates = testing_data.index 
    
    # Compute actual revenue for each model
    for m in range(len(method)):
        bids_DA, expected_revenue = bid_optimization(data, dates, y_prediction[m], wind_capacity=6000, plots=plots)
        
        actual_revenue = calc_revenue(data, dates, bids_DA)

        revenue_table.at[m, 'Expected Revenue'] = round(expected_revenue, 2)
        revenue_table.at[m, 'Actual Revenue'] = round(actual_revenue, 2)
        
        if plots:
            print(f"The {method[m]} method results in an actual revenue of {actual_revenue} DKK, from {dates[0]} to {dates[-1]}.")

    if plots:
        print()
    print(revenue_table.sort_values(by='Actual Revenue', ascending=False))

    return revenue_table


def revenue_calculation_model2(y_prediction, method:str, data, plots = True):
    '''
    Computes the assignment step on revenue calculation for evaluation. For model 2, 
    the predictions are the optimal bid to place on the day-ahead market. Such that,
    we don't need to run the optimization model of model 1 as they are already data 
    in our model.

    Args:
        y_prediction (dataframe):   Contains 1 column corresponding to the model predictions  
            (for any method). Here, they correspond to the optimal bids predicted.
        method (string):    Associated method of prediction.
        data (dataframe):   Index are the timestamps. Contains many columns corresponding to 
            all the relevant data for our project.
        plots (boolean):    Print the plots if set to True, as default.
    
    Returns:
        revenue_table (float, float): Revenues based on the optimized 
            bids of the day-ahead market & the power production predictions (input) and realizations (data).
    '''
    print('\nRevenue calculation results:\n')

    # Table to store the actual revenue for each model
    revenue_table = pd.DataFrame(columns = ['Model','Actual Revenue'])
    revenue_table['Model'] = method

    # Get the dates corresponding to the predictions (index of the last 20% of our data)
    num_rows = int(0.2 * len(data))
    testing_data = data.tail(num_rows)
    dates = testing_data['ts'].to_list()
    
    # Compute actual revenue for each model
    for m in range(len(method)): 
        
        bids_DA = y_prediction[m]
        
        # Format the predictions
        if not (isinstance(bids_DA, np.ndarray)): # it would be a single-column panda.dataframe otherwise
            bids_DA = bids_DA.iloc[:, 0].to_numpy()
        elif bids_DA.ndim > 1:
            bids_DA = bids_DA.flatten()
            
        actual_revenue = calc_revenue(data, dates, bids_DA)

        revenue_table.at[m, 'Actual Revenue'] = round(actual_revenue, 2)
        
        if plots:
            print(f"The {method[m]} method results in an actual revenue of {actual_revenue} DKK, from {dates[0]} to {dates[-1]}.")

    if plots:
        print()
    print(revenue_table.sort_values(by='Actual Revenue', ascending=False))

    return revenue_table