import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.data_collection.data_generator import split_data, data_collection
from scripts.linear_regression.linear_regression import closed_form, evaluate_model
from scripts.nonlinear_regression.nonlinear_regression import nonlinear_regression
from sklearn.metrics import mean_absolute_error, r2_score

def prediction_closed_form_L1_model2(x_train, x_test, y_train, y_test, _lambda=1000, _print=True):
    num_data, num_features = x_train.shape
    x_train = x_train.to_numpy()
    y_train = y_train.iloc[:, 0].to_numpy()

    # Create Gurobi model for L1 regularization
    model = gp.Model("L1_regularization_model2")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 100 

    # Add variables for model coefficients and absolute values
    beta = {m: model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY) for m in range(num_features)}
    abs_beta = {m: model.addVar(lb=0, ub=gp.GRB.INFINITY) for m in range(num_features)}

    # Define the objective with L1 regularization
    objective = gp.quicksum(
        (y_train[i] - gp.quicksum(beta[j] * x_train[i, j] for j in range(num_features))) ** 2
        for i in range(num_data)
    ) + _lambda * gp.quicksum(abs_beta[j] for j in range(num_features))
    model.setObjective(objective, gp.GRB.MINIMIZE)

    # Constraints for absolute values in L1
    for m in range(num_features):
        model.addLConstr(abs_beta[m] >= beta[m])
        model.addLConstr(abs_beta[m] >= -beta[m])

    # Optimize model
    model.optimize()
    if model.status != GRB.Status.OPTIMAL:
        raise ValueError("The model is infeasible or unbounded.")

    # Retrieve coefficients and calculate predictions
    opt_beta = np.array([beta[m].x for m in range(num_features)])
    y_pred = np.dot(x_test, opt_beta)

    # Calculate evaluation metrics
    y_test_np = y_test.iloc[:, 0].to_numpy()
    MAE_closed = np.mean(np.abs(y_pred - y_test_np))
    RMSE_closed = np.sqrt(np.mean((y_pred - y_test_np) ** 2))
    R2_closed = r2_score(y_test_np, y_pred)

    if _print:
        print("\nLinear regression with Lasso regularization (Model 2)")
        print("Beta =", opt_beta)
        print("MAE for closed form with L1:", MAE_closed)
        print("RMSE for closed form with L1:", RMSE_closed)
        print("R-squared:", R2_closed)

    return (opt_beta, y_pred, MAE_closed, RMSE_closed, R2_closed)



# Linear regression with Ridge (L2) regularization
# Linear regression with Ridge (L2) regularization
def prediction_closed_form_L2_model2(x_train, x_test, y_train, y_test, _lambda=10, _print=True):
    Id = np.eye(x_train.shape[1])

    # Ridge regression closed-form solution
    a = x_train.T @ x_train + _lambda * Id
    b = x_train.T @ y_train
    beta = np.linalg.inv(a) @ b

    # Calculate predictions
    y_pred = x_test @ beta
    y_test_np = y_test.to_numpy()

    # Calculate evaluation metrics
    MAE_closed = np.mean(np.abs(y_pred - y_test_np))
    RMSE_closed = np.sqrt(np.mean((y_pred - y_test_np) ** 2))
    r2_closed = r2_score(y_test_np, y_pred)

    if _print:
        print("\nLinear regression with Ridge regularization (Model 2)")
        print("Beta =", beta)
        print("MAE for closed form with L2:", MAE_closed)
        print("RMSE for closed form with L2:", RMSE_closed)
        print("R-squared:", r2_closed)

    return (beta, y_pred, MAE_closed, RMSE_closed, r2_closed)



# Optimize lambda for both L1 and L2 regularization
def optimize_lambda_model2(X_train, X_test, y_train, y_test, model="model2", plot=True):
    if model == 'model1':
        lambda_range_L1 = np.arange(5000, 7000, 50)
        lambda_range_L2 = np.arange(0, 30, 1)
    else:
        lambda_range_L1 = np.arange(0, 10000, 500)
        lambda_range_L2 = np.arange(0, 1000, 10)
    
    # L1 Regularization
    rmse_values_L1 = [prediction_closed_form_L1_model2(X_train, X_test, y_train, y_test, _lambda=l, verbose=False)[3] for l in lambda_range_L1]
    opt_lambda_L1 = lambda_range_L1[np.argmin(rmse_values_L1)]

    # L2 Regularization
    rmse_values_L2 = [prediction_closed_form_L2_model2(X_train, X_test, y_train, y_test, _lambda=l, verbose=False)[3] for l in lambda_range_L2]
    opt_lambda_L2 = lambda_range_L2[np.argmin(rmse_values_L2)]

    print(f"\nOptimal Lambda for L1 (Lasso): {opt_lambda_L1} | RMSE: {min(rmse_values_L1)}")
    print(f"Optimal Lambda for L2 (Ridge): {opt_lambda_L2} | RMSE: {min(rmse_values_L2)}")

    # Plot RMSE values vs Lambda
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(lambda_range_L1, rmse_values_L1, marker='o', label='Lasso (L1)')
        plt.plot(lambda_range_L2, rmse_values_L2, marker='o', label='Ridge (L2)')
        plt.xlabel("Lambda")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title("RMSE as a function of Lambda for L1 and L2 regularization")
        plt.show()

    return opt_lambda_L1, opt_lambda_L2
def regularization_model2(data, model, features, target, data_path, plot=True):
    '''
    Computes the assignment step on regularization methods for Model 2.

    Args:
        data: Dataframe containing all relevant data for our model, features & targets.
        model (string): Either 'model1' or 'model2'.
        plot: Allows to print the plots. Default is set to True.

    Returns:
        (beta_linear, y_pred_linear, beta_L1, y_pred_L1, beta_L2, y_pred_L2)
    '''
    print('\n Regularization Model 2 results:')
    
    # Get training and testing set for our model
    X_train, X_test, y_train, y_test = split_data(df=data, features=features, targets=target)
    X_train.insert(0, 'Bias', 1)  # Add bias term
    X_test.insert(0, 'Bias', 1)    # Add bias term

    # Compute the linear regression of this model
    beta_linear = closed_form(X_train, y_train)
    y_pred_linear = X_test.dot(beta_linear)

    # Calculate metrics for linear regression
    MAE_lin = mean_absolute_error(y_test, y_pred_linear)
    RMSE_lin = np.sqrt(np.mean((y_pred_linear - y_test) ** 2))
    R2_lin = r2_score(y_test, y_pred_linear)

    # Optimize lambda values for L1 and L2 regularization
    opt_lambda_L1, opt_lambda_L2 = optimize_lambda_model2(X_train, X_test, y_train, y_test, model=model, plot=plot)
    
    # Lasso Regression (L1 Regularization)
    beta_L1, y_pred_L1 = prediction_closed_form_L1_model2(X_train, X_test, y_train, y_test, _lambda=opt_lambda_L1)
    MAE_L1 = mean_absolute_error(y_test, y_pred_L1)
    RMSE_L1 = np.sqrt(np.mean((y_pred_L1 - y_test) ** 2))
    R2_L1 = r2_score(y_test, y_pred_L1)

    # Ridge Regression (L2 Regularization)
    beta_L2, y_pred_L2 = prediction_closed_form_L2_model2(X_train, X_test, y_train, y_test, _lambda=opt_lambda_L2)
    MAE_L2 = mean_absolute_error(y_test, y_pred_L2)
    RMSE_L2 = np.sqrt(np.mean((y_pred_L2 - y_test) ** 2))
    R2_L2 = r2_score(y_test, y_pred_L2)

    if plot:
        # Print results for each regression method
        print("Linear Regression Results:")
        print("MAE:", MAE_lin)
        print("RMSE:", RMSE_lin)
        print("R²:", R2_lin)

        print("\nLasso Regression Results:")
        print("MAE:", MAE_L1)
        print("RMSE:", RMSE_L1)
        print("R²:", R2_L1)

        print("\nRidge Regression Results:")
        print("MAE:", MAE_L2)
        print("RMSE:", RMSE_L2)
        print("R²:", R2_L2)

    return (beta_linear, y_pred_linear, beta_L1, y_pred_L1, beta_L2, y_pred_L2)
