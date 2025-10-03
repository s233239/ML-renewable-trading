import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_collection.data_generator import data_collection, split_data
from sklearn.preprocessing import StandardScaler

# Load the original dataset
path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..')), 'data')
data_path = os.path.join(path_to_data, "features-targets.csv")  # Adjust with your dataset filename
data = pd.read_csv(data_path)

# Load the optimal bids
optimal_bids_path = os.path.join(path_to_data, "optimal_bids.csv")
optimal_bids = pd.read_csv(optimal_bids_path)

# Assuming optimal bids have a timestamp column to join with the main dataset
# Merge optimal bids with the original dataset
data = data.merge(optimal_bids, on='ts', how='left')  # Adjust 'ts' as needed

# Select relevant features, including the timestamp
selected_features = [
    'ts',                            # Include the timestamp
    'mean_wind_speed',              # Feature 1
    'SpotPriceDKK',                 # Feature 2: Spot price
    'BalancingPowerPriceUpDKK',     # Feature 3: Up-regulation price
    'BalancingPowerPriceDownDKK',   # Feature 4: Down-regulation price
    'mean_wind_power',
    'max_wind_speed_3sec',
    'max_wind_speed_10min',
    'optimal_bid'                   # Adding optimal bids to the features
]

# Create a new DataFrame with only selected features
model2_data = data[selected_features].copy()

# Normalize features for stable model training (excluding 'ts' and 'optimal_bid')
scaler = StandardScaler()
model2_data[selected_features[1:-1]] = scaler.fit_transform(model2_data[selected_features[1:-1]])

# Save the modified dataset to CSV
output_file_path = os.path.join(path_to_data, "model2_dataset.csv")
model2_data.to_csv(output_file_path, index=False)

print("Data prepared and saved for Model 2.")
