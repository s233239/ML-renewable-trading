import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

#Note: This might not work if script is run virtually.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.data_generator import data_collection, split_data

def gradient_descent (X, y, learning_rate:float, iterations:int, tolerance: float=0.01):
    """Gradient Descent algorithm.

    Args:
        X (list): list of features
        y (list): list of targets
        learning_rate (float): learning rate
        iterations (int): iterations
        tolerance (float: minimum change in error to stop iteration

    Returns:
        Theta.
    """
    m = len(y)
    theta = np.random.randn(X.shape[1], 1)  # initialization
    prev_error = float('inf') 

    for i in range(iterations):
        # calculate the gradient
        gradient = -(2/m) * X.T.dot(y-X.dot(theta))
        theta = theta - learning_rate * gradient
        
        current_error = mean_squared_error(y, X.dot(theta))
        if abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error 
    return theta

def closed_form (X, y):
    """Calculates closed form solutions

    Args:
        X (list): list of features
        y (list): list of targets

    Returns:
        solution
    """
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

def evaluate_model(X, y, theta):
    """Evaluates the model using RMSE, MAE, and R-squared metrics.

    Args:
        X (list): list of features
        y (list): list of targets
        theta (numpy array): model parameters

    Returns:
        Tuple of (RMSE, MAE, R-squared)
    """
    y_predicted = X.dot(theta)
    rmse = np.sqrt(mean_squared_error(y, y_predicted))
    mae = mean_absolute_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)
    return rmse, mae, r2

def linear_regression(path_to_data = os.path.join(os.path.abspath(os.path.join(__file__, '..','..','..')), 'data')):
    trainingdata = data_collection(path_to_data)

    # load split data from training data;
    X_train, X_test, y_train, y_test = split_data(df=trainingdata.head(100), features=['mean_wind_speed'], targets=['Kalby_AP'])

    X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

    # learning rate and iteration
    learning_rate = 0.01
    iterations = 1000

    # calculate the solutions
    gradientdescent_solution = gradient_descent(X_train_bias, y_train, learning_rate, iterations)
    closedform_solution = closed_form (X_train_bias, y_train)

    # print solutions
    print ("Gradient descent solution:", gradientdescent_solution)
    print ("Closed form solution: ", closedform_solution)

    #######
    #trainingdatalarger = pd.read_csv('trainingdatalarger.csv')
    #testingdatalarger = pd.read_csv('testingdatalarger.csv')
    # load split data from training data;
    # Run with 1000 data points.
    X_train, X_test, y_train, y_test = split_data(df=trainingdata.head(1000), features=['mean_wind_speed'], targets=['Kalby_AP'])


    #X_train_largerset = trainingdatalarger.iloc[:, :-1].values  
    #y_train_largerset = trainingdatalarger.iloc[:, -1].values.reshape(-1, 1)
    X_train_largerset_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

    closedform_solution = closed_form(X_train_largerset_bias, y_train)

    print("Closed form solution with more samples:", closedform_solution)

    #load testing data
    trainingdata = data_collection(path_to_data)

    X_train, X_test, y_train, y_test = split_data(df=trainingdata, features=['mean_wind_speed'], targets=['Kalby_AP'])

    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    closedform_solution = closed_form(X_test_bias, y_test)

    y_predicted = X_test_bias.dot(closedform_solution)

    # evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    mae = mean_absolute_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    print("Closed form model evaluation metrics on testing data:")
    print("Root mean squared error:", rmse)
    print("Mean absolute error:", mae)
    print("R-squared:", r2)
    return




# linear_regression()