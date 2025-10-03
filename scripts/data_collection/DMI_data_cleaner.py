import pandas as pd
import os
import numpy as np

def read_filter_dmi_data(directory, features=['mean_pressure', 'mean_wind_speed', 'max_wind_speed_10min', 
                                              'max_wind_speed_3sec']):
    """ Reads DMI data and filters for features

    Args:
        directory (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    data = pd.read_json(directory, lines=True)
    filtered_data = data[(data['properties'].apply(lambda x: x['stationId'] == '06190')) & 
                         (data['properties'].apply(
                             lambda x: x['parameterId'] in features)
                          )]
    filtered_data.reset_index(drop=True, inplace=True)
    return filtered_data

def filter_dmi_repeat(folder=None, save=False):
    """ filters DMI data for a folder

    Args:
        folder (_type_, optional): _description_. Defaults to None.
        save (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    data_collected = {}
    
    if folder is None:
        data_folder = os.path.join(os.path.abspath(os.path.join( __file__, '..', '..', '..')), 'data')
        folder_dir = os.path.join(data_folder, 'wind-data', 'DMI-data')
    else:
        folder_dir = folder
    
    folder_names = os.listdir(folder_dir)

    for folder in folder_names:
        print('Initiated folder:', folder)
        
        try:
            current_dir = folder_dir+ '\{0}'.format(folder)
            file_names = os.listdir(current_dir)
            
            for file in file_names:
                # Read date from file name:
                date = file[:-4]
                print('Initiated date:', date)
                
                # Construct directory string;
                directory = current_dir + '\{0}'.format(file)
                
                # Read filtered data
                filtered_data = read_filter_dmi_data(directory)
                
                # Add filetered_data to dictionary
                data_collected[date] = filtered_data
        except Exception as e:
            print(e)
            
    # save the entire collection:
    if save:
        np.save(folder_dir+"\processed_DMI_data.npy", data_collected)

    return data_collected
