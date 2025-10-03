import pandas as pd
import os
from data_collection.data_generator import load_features
def delimiter_correction(df):
    """process dataset from "," decimal seperation to "."

    Args:
        path (_type_, DataFrame): Dataframe with the specific columns that need updating
       
    Returns:
        pd.DataFrame: With "." decimal seperation
    """
    for col in df:
            df[col] = df[col].astype(str).apply(lambda x: x.replace(",", "."))
    
    return df

def calc_revenue(strategy):
    """Gets the actual revenue from production.
       Production should end on 2023-12-30 22:00 to get the correct result.

    Args:
        path (_type_, List): List describing the amount of energy promised to the grid
       
    Returns:
        Actual revenue
    """
     #import data
    path_to_data = os.path.join(os.path.abspath(os.path.join(__file__,'..')), 'data')
    df_features = pd.read_csv(os.path.join(path_to_data, 'features-targets.csv'))
    data = load_features(df_features)

    target = data["Kalby_AP"]
    BP_up = data["BalancingPowerPriceUpDKK"]
    BP_down = data["BalancingPowerPriceDownDKK"]
    SP = data["SpotPriceDKK"]

    #import actual values
    true_value = target.tail(len(strategy))

    #cut BP and SP to contain only value relevant to strategy
    BP_up = (BP_up.tail(len(strategy)))
    BP_down = (BP_up.tail(len(strategy)))
    SP = (SP.tail(len(strategy)))

    revenue = 0

    for i in range(1,len(strategy)):
        revenue +=  float(SP[i])*strategy[i] + float(BP_up[i])*(max((true_value[i]-strategy[i]),0)) + float(BP_down[i])*(max((strategy[i]-true_value[i]),0))
    return revenue


strategy = [5, 23, 5, 67,  67,  1, 2]

print(calc_revenue(strategy))