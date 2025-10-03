"""
Authors: Albert R. H. and Mikkel V. K. A.
Edited by Nicole
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt

# power curve of Kalby turbines:
# y = 0.0063x6 - 0.2817x5 + 4.5191x4 - 31.955x3 + 111.24x2 - 109.06x - 119.25
def power_curve_function(X):
    """ Returns power generation of Kalby_AP turbine for a list of wind speeds.
    
    Args:
        X (list): Wind speeds with a given frequency

    Returns:
        list: power generation with wind speeds X
    """
    # list for generated values
    result = []
    # for each element in x:
    for x in X:
        if x<3.5 or x>25:
            # append 0 if wind speed is lower than operational limits of turbine
            result.append(0)
        else:
            # calculate the power generation of x
            y = 0.0063*x**6 - 0.2817*x**5 + 4.5191*x**4 - 31.955*x**3 + 111.24*x**2 - 109.06*x - 119.25
            # only append y if it is smaller than max power generation (2000)
            result.append(min(2000, y))
    
    return np.multiply(result, 3)
    

def create_features(folder=None, file_name='processed_DMI_data.npy', 
                    features=['mean_pressure', 'mean_wind_speed', 'max_wind_speed_10min', 
                              'max_wind_speed_3sec'], save=False):
    
    """ Creates features from "processed_DMI_data.npy" file.

    Args:
        save (bool, optional): Whether DataFrame should be saved. Defaults to False.

    Returns:
        DataFrame: df_total containing features and target
    """
    
    if folder is None:
        data_folder = os.path.join(os.path.abspath(os.path.join( __file__, '..', '..', '..')), 'data')
        dmi_folder = os.path.join(data_folder, 'wind-data', 'DMI-data')
    else:
        dmi_folder = folder
    
    # load dmi_data
    dmi_data = np.load(os.path.join(dmi_folder, file_name), allow_pickle="TRUE").item()

    # trim dmi_data to only include features
    dmi_data = [dmi_data[i]['properties'] for i in dmi_data]
    
    # empty variables
    df_features = pd.DataFrame()
    features_df_list = []

    # for each feature
    for f in features:
        print('adding feature {0}'.format(f))
        df_features_helper = []
        
        # add all values for each (time, value)
        for d in dmi_data:
            df_features_helper.append([(pd.to_datetime(j['from']), j['value']) for j in d if j['parameterId'] == f])
        # unpack inner-list
        df_features_helper = [x for xs in df_features_helper for x in xs]
        # convert to dataframe
        df_features_helper = pd.DataFrame(df_features_helper, columns=['ts', f]).sort_values('ts').set_index('ts')
        # add to list of feature dataframes
        features_df_list.append(df_features_helper)
        
        if df_features.size != 0:
            df_features.insert(1, f, df_features_helper, True)
        else:
            df_features = df_features_helper

    # load turbine data
    df_turbine_generation = pd.read_csv(os.path.join(data_folder, 'turbine-data', 'Kalby_AP.csv'))
    df_turbine_generation['ts'] = pd.to_datetime(df_turbine_generation['ts'], utc=True)
    df_turbine_generation.sort_values('ts', inplace=True)
    df_turbine_generation.set_index('ts', inplace=True)

    # calculate and append power generated per turbine for Kalby_AP using mean_wind_speed
    df_features['mean_wind_power'] = power_curve_function(df_features['mean_wind_speed'])

    # remove daily calculated values done by DMI
    # they are (luckily) easily identifiable by having 1000ms in "from" property
    idx_drops = [i for i in df_features.index if i.microsecond > 0]
    df_features = df_features.drop(idx_drops)

    # combined features and targets:
    df_total = pd.concat([df_features, df_turbine_generation], axis=1)
    df_total.index = pd.to_datetime(df_total.index)

    # scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_total[features])
    scaled_features_df = pd.DataFrame(scaled_features, columns=features, index=df_total.index)
    
    # save df_features and
    if save:
        df_total.to_csv('features-targets.csv')
    
    return df_total

def load_features(file=None):
    """Loads features-targets.csv

    Args:
        file (path, optional): path to file. Defaults to None.

    Returns:
        dataframe: features and targets
    """
    
    if file is None:
        data_folder = os.path.join(os.path.abspath(os.path.join( __file__, '..', '..', '..')), 'data')
        df_features = pd.read_csv(os.path.join(data_folder, 'features-targets.csv'))
        df_features.set_index(df_features.columns[0], inplace=True)
    else:
        df_features = pd.read_csv(file)
    
    df_features['ts'] = pd.DataFrame([datetime.strptime(s, '%Y-%m-%d %H:%M:%S') for s in df_features['ts']])
    df_features.set_index('ts', inplace=True)
    
    return df_features

def process_spotprices(path, format: str='%Y-%m-%d %H:%M', keyword: str='HOUR', time_columns=None) -> pd.DataFrame:
    """process Energinet dataset to DataFrame

    Args:
        path (_type_): _description_
        format (_type_, optional): _description_. Defaults to '%Y-%m-%d %H:%M:%S'.
        keyword (str, optional): _description_. Defaults to 'HOUR'.
        time_columns (_type_, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: ENDK dataset as DataFrame with HourDK as index.
    """
    # Remove index
    dataframe = pd.read_csv(path, sep=';')
    dataframe.reset_index(drop=True, inplace=True)

    # Check time columns are in DateTime format
    time_columns = [col for col in dataframe.columns if keyword.upper() in col.upper()]
    price_columns = [col for col in dataframe.columns if 'Spot' in col]
    
    # Convert str to datetime
    for c in time_columns:
        if isinstance(c[0], str):
            dataframe[c] = pd.DataFrame([datetime.strptime(s, format) for s in dataframe[c]])
    
    for c in price_columns:
        if isinstance(c[0], str):
            dataframe[c] = pd.DataFrame([float(s.replace(",", ".")) for s in dataframe[c]])
    
    dataframe.set_index(dataframe['HourDK'], inplace=True)
    dataframe.index.rename('ts',inplace=True)
    dataframe.drop(columns=['HourUTC', 'HourDK', 'PriceArea'], inplace=True)
    #dataframe = dataframe.loc[(dataframe.index < pd.Timestamp(year=2024, day=1, month=1)) & (dataframe.index>pd.Timestamp(year=2020, month=1, day=1) )]
    
    return dataframe

"""
def process_balancing_data(path):
    # Load dataframe
    dataframe = pd.read_csv(path, sep=';')
    dataframe.reset_index(drop=True, inplace=True)
    
    # define relevant/irrelevant columns
    time_columns = [col for col in dataframe.columns if 'HOUR' in col.upper()]
    data_columns = [c for c in dataframe if 'BalancingPower' in c]
    keep_columns = time_columns+data_columns
    drop_columns = [c for c in dataframe if c not in keep_columns]
    
    dataframe.drop(columns=drop_columns, inplace=True)
    
    # Convert str to datetime
    for c in time_columns:
        if isinstance(c[0], str):
            dataframe[c] = pd.DataFrame([datetime.strptime(s, '%Y-%m-%d %H:%M') for s in dataframe[c]])
    
    for c in data_columns:
        c_list = []
        for s in dataframe[c].values:
            if isinstance(s, str):
                c_list.append(float(s.replace(',', '.')))
            else:
                c_list.append(s)
        dataframe[c] = pd.DataFrame(c_list)
    
    #dataframe['HourUTC'] = pd.DataFrame([datetime.strptime(s, '%Y-%m-%d %H:%M:%S')for s in dataframe['HourUTC']])
    dataframe.set_index(dataframe['HourUTC'], inplace=True)
    dataframe.index.rename('ts',inplace=True)
    dataframe.drop(columns=['HourUTC', 'HourDK'], inplace=True)
    
    #dataframe = dataframe.loc[dataframe.index]
    
    return dataframe
"""


def plot_feature_correlation(df):
    """Plot features correlation to actual wind generation.

    Args:
        df (pd.DataFrame): DataFrame
    """
    
    fig, ax = plt.subplots(figsize=(10,5))
    ylabels = [i for i in df]
    ax.scatter(y=ylabels, x=df.corrwith(np.abs(df['Kalby_AP'].drop(columns='ts'))), marker='x')
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")
    ax.set_xticks(np.arange(-1,1.1,0.1))
    ax.set_xticklabels(np.round(np.arange(-1,1.1,0.1),2), rotation=0)
    ax.set_title('Correlation between features and "Kalby_AP" production')
    ax.axvline(x=0, color='black')
    ax.grid()
    fig.show()    
    return


def split_data(df, targets=['Kalby_AP'], features=None, test_split=0.2):
    """Constructs X_train, X_test, Y_train, Y_test of a pd.DataFrame.

    Args:
        df (DataFrame): DataFrame containing features and targets
        targets (list, optional): list of targets in df (strings). Defaults to ['Kalby_AP'].
        features (list, optional): list of features in df (strings). Defaults to ['mean_pressure', 'mean_temp', 'wind_generation', 'mean_wind_speed'].
        test_split (float, optional): Test_size (e.g. 0.2 = 20% of df). Defaults to 0.2.

    Returns:
        X_train, X_test, Y_train, Y_test: Returns from train_test_split function.
    """
    if features is None:
        features = [i for i in df.columns if i not in targets]
    
    return train_test_split(df[features], df[targets], test_size=test_split, shuffle=False)

def data_collection(data_path: str) -> pd.DataFrame:
    """Function to run all relevant commands for loading data.

    Args:
        data_path (string, optional): Path to datafolder. Defaults to None.
    """
    # User MUST provide function with the path to the data_folder.
    feature_path = os.path.join(data_path, 'features-targets.csv')
    df_features = load_features(feature_path)
    df_features['Kalby_AP'] = np.abs(df_features['Kalby_AP'])
    
    return df_features


def data_generation_model2(path_to_data, features, target):

    data_path = os.path.join(path_to_data, "model2_dataset.csv")
    data = pd.read_csv(data_path)
    
    # Check for NaN values in target
    if data[target].isnull().any().any():
        data = data.dropna(subset=target)  # Remove rows with NaN in target

    # Check for NaN values in features
    if data[features].isnull().any().any():
        data = data.dropna(subset=features)  # Remove rows with NaN in features

    return data