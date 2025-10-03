
    data = data.dropna(subset=[target])  # Remove rows with NaN in target

# Check for NaN values in features
if data[features].isnull().any().any():
    data = data.dropna(subset=features)  # Remove rows with NaN in features