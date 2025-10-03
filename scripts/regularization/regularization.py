import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.data_collection.data_generator import split_data, data_collection
from scripts.linear_regression.linear_regression import closed_form, evaluate_model
from scripts.nonlinear_regression.nonlinear_regression import nonlinear_regression
from sklearn.metrics import mean_absolute_error, r2_score


# Lasso regularization
def prediction_closed_form_L1(x_train, x_test, y_train, y_test, _lambda = 1000, _print = True):
    '''
    Linear regression with Lasso regularization

    Args:
        (x_train, x_test): features training and testing datasets
        (y_train, y_test): targets training and testing datasets
        _lambda: regularization parameter
        _print: set to True as default. Enables to print the solution or not.

    Returns:
        (Optimal beta, y prediction from x_test, MAE, RMSE)
    '''
    #   n+1, m+1
    num_data, num_features = x_train.shape #Output: (number_of_rows, number_of_columns)

    if not (isinstance(x_train, np.ndarray)):
        x_train = x_train.to_numpy() #Of shape (num_data, num_features)
    y_train = y_train.iloc[:, 0].to_numpy() #Of shape (num_data)

    # Create the Gurobi model
    model = gp.Model(name="L1 regularization")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 100 

    # Add variables to the Gurobi model
    beta = {
        m: model.addVar(
            lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY
        )
        for m in range(num_features)
    }

    aux_beta = {
        m: model.addVar(
            lb=0, ub=gp.GRB.INFINITY
        )
        for m in range(num_features)
    }


    # Set objective function 
    objective = gp.quicksum( 
            (y_train[i] - gp.quicksum( beta[j] * x_train[i,j] for j in range(num_features))) **2
        for i in range(num_data)
        ) + _lambda * gp.quicksum( 
            aux_beta[j] 
        for j in range(num_features) 
        )

    model.setObjective(objective, gp.GRB.MINIMIZE)

    # Add constraints to the Gurobi model 
    beta_value = {
        m: model.addLConstr(
            aux_beta[m],
            gp.GRB.GREATER_EQUAL,
            beta[m]
        )
        for m in range(num_features)
    }

    beta_value_bis = {
        m: model.addLConstr(
            aux_beta[m],
            gp.GRB.GREATER_EQUAL,
            - beta[m]
        )
        for m in range(num_features)
    }

    # Optimize the problem
    model.optimize()

    # Check the model status
    if model.status == gp.GRB.Status.OPTIMAL:
        # If the model is optimal, retrieve the solution
        opt_beta = [beta[m].x for m in range(num_features)]
        opt_objective = model.ObjVal
        
    elif model.status == gp.GRB.Status.INFEASIBLE:
        # Handle infeasibility
        try:
            model.computeIIS()
            model.write("infeasible_model.ilp")  # Save an IIS file to examine constraints
            raise ValueError("The model is infeasible. IIS written to infeasible_model.ilp.")
        except gp.GurobiError as e:
            print(f"Error during IIS computation: {e}")

    elif model.status == gp.GRB.Status.UNBOUNDED:
        # Handle unbounded case
        raise ValueError("The model is unbounded.")
    
    elif model.status == gp.GRB.Status.SUBOPTIMAL:
        # Handle suboptimal solutions
        print("The model is suboptimal; an approximate solution was found.")
        opt_beta = [beta[m].x for m in range(num_features)]
        opt_objective = model.ObjVal

    else:
        # Handle other statuses if needed
        raise ValueError(f"Optimization was stopped with status {model.status}")


    # Predict y values using the Lasso regression model
    y_test_predict = np.dot(x_test, opt_beta)

    y_test = y_test.iloc[:, 0].to_numpy() # Adjust to use the single target directly
    MAE_closed = (1 / len(y_test)) * np.sum(np.abs(y_test_predict - y_test))
    mae_module = mean_absolute_error(y_test, y_test_predict)
    RMSE_closed = np.sqrt((1 / len(y_test)) * np.sum(np.square(y_test_predict - y_test)))
    r2_closed = r2_score(y_test, y_test_predict)

    if _print:
        print("\nLinear regression with Lasso regularization")
        print("Beta =", np.array(opt_beta))
        print("MAE for closed form with L1:", MAE_closed)
        if not MAE_closed==mae_module:
            print("MAE:", mae_module)
        print("RMSE for closed form with L1:", RMSE_closed)
        print("R-squared:", r2_closed)

    return (opt_beta, y_test_predict, MAE_closed, RMSE_closed, r2_closed)


# Ridge regression
def prediction_closed_form_L2(x_train, x_test, y_train, y_test, _lambda = 10, _print = True):
    '''
    Linear regression with Ridge regularization

    Args:
        x_train, x_test, y_train, y_test = split_data()
        x_train, x_test: features training and testing datasets
        y_train, y_test: targets training and testing datasets
        _lambda: regularization parameter
        _print: set to True as default. Enables to print the solution or not.

    Returns:
        Optimal beta, y prediction from x_test, MAE, RMSE
    '''
    Id = np.eye(x_train.shape[1])

    # Compute beta from compact formulation of Ridge model
    a = np.dot(x_train.T, x_train) + _lambda * Id
    b = np.dot(x_train.T, y_train)

    beta = np.dot(np.linalg.inv(a), b)

    # Predict y values using the Ridge regression model
    y_test_predict = x_test.dot(beta)

    if not (isinstance(y_test_predict, np.ndarray)):
        y_test_predict = y_test_predict.to_numpy()
    y_test = y_test.to_numpy()

    MAE_closed = (1 / len(y_test)) * np.sum(np.abs(y_test_predict - y_test))
    mae_module = mean_absolute_error(y_test, y_test_predict)
    RMSE_closed = np.sqrt((1 / len(y_test)) * np.sum(np.square(y_test_predict - y_test)))
    r2_closed = r2_score(y_test, y_test_predict)

    if _print:
        print("\nLinear regression with Ridge regularization")
        print("Beta =", beta)
        print("MAE for closed form with L2:", MAE_closed)
        if not MAE_closed==mae_module:
            print("MAE:", mae_module)
        print("RMSE for closed form with L2:", RMSE_closed)
        print("R-squared:", r2_closed)

    return (beta, y_test_predict, MAE_closed, RMSE_closed, r2_closed)


def lambda_optimization(method, model, X_train, X_test, y_train, y_test, plots=True):
    '''
    Experiment with different values for the regularization parameter
    to find the optimal level of regularization. 

    Args:
        method (string): Either 'linear' or 'nonlinear' regression model to regularize.
        model (string): Either 'model1' or 'model2' to work on.
        (X_train, X_test, y_train, y_test): Data of features and targets, obtained with split_data().
        plots: Set to True as default. If set to False, plots will not be printed.

    Returns:
        (opt_lambda_L1, opt_lambda_L2): Optimal values of regularization parameter for L1 and L2 regularization respectively.
    '''
    # L1 regularization optimization of lambda
    if model == 'model1':
        if method == 'linear':
            _lambda_values_L1 = np.arange(0,2,0.1)
        else:
            _lambda_values_L1 = np.arange(0,0.5,0.01)
    else:
        if method == 'linear':
            _lambda_values_L1 = np.arange(0,0.1,0.002)
        else:
            _lambda_values_L1 = np.arange(33150,33250,5)

    rmse_values_L1 = []

    for _lambda in _lambda_values_L1:
        RMSE = prediction_closed_form_L1(X_train,X_test,y_train,y_test, _lambda = _lambda, _print = False)[3]
        rmse_values_L1.append(RMSE)

    opt_RMSE_L1 = min(rmse_values_L1)
    opt_lambda_L1 = _lambda_values_L1[rmse_values_L1.index(opt_RMSE_L1)]

    print('\nL1 regularization optimization of lambda parameter')
    print(f'The optimal RMSE = {opt_RMSE_L1} for lambda = {opt_lambda_L1}')

    # L2 regularization optimization of lambda
    if model == 'model1':
        if method == 'linear':
            _lambda_values_L2 = np.arange(540,610,1)
        else:
            _lambda_values_L2 = np.arange(0,30,1)
    else:
        if method == 'linear':
            _lambda_values_L2 = np.arange(0,0.1,0.002)
        else:
            _lambda_values_L2 = np.arange(0,30,1)
    
    rmse_values_L2 = []
    for _lambda in _lambda_values_L2:
        RMSE = prediction_closed_form_L2(X_train,X_test,y_train,y_test, _lambda = _lambda, _print = False)[3]
        rmse_values_L2.append(RMSE)

    opt_RMSE_L2 = min(rmse_values_L2)
    opt_lambda_L2 = _lambda_values_L2[rmse_values_L2.index(opt_RMSE_L2)]

    print('\nL2 regularization optimization of lambda parameter')
    print(f'The optimal RMSE = {opt_RMSE_L2} for lambda = {opt_lambda_L2}')

    if plots:
        plt.figure(figsize=(8, 5))

        plt.subplot(2,1,1)
        plt.plot(_lambda_values_L1, rmse_values_L1, marker = 'o')
        plt.title('RMSE as a function of Lambda for L1 regularization')
        plt.ylabel('Root mean squared error (RMSE)')

        plt.subplot(2,1,2)                
        plt.plot(_lambda_values_L2, rmse_values_L2, marker = 'o')
        plt.title('RMSE as a function of Lambda for L2 regularization')
        plt.xlabel('Lambda (regularization parameter)')
        plt.ylabel('Root mean squared error (RMSE)')
        
        plt.tight_layout()
        plt.show()

    return opt_lambda_L1, opt_lambda_L2


def regularization(data, model, features, target, plots = True, _lambda_opt = False):
    '''
    Computes the assignment step on regularization methods.

    Args:
        model (string): Either 'model1' or 'model2'.
        data: Dataframe containing all relevant data for our model, features & targets.
        plots: Allows to print the plots. Default is set to True. 

    Returns:
        (y_predict_lin, y_predict_lin_L1, y_predict_lin_L2, y_pred_nl, y_predict_nonlin_L1, y_predict_nonlin_L2):
    '''
    print('\nRegularization results:')
    
    # Get training and testing set for our linear model
    X_train, X_test, y_train, y_test = split_data(
        df=data, 
        features=features, 
        targets=target
        )
    X_train.insert(0, '1', 1)
    X_test.insert(0, '1', 1)

    # Compute the linear regression of this model
    print("\nLinear regression solution")

    beta = closed_form(X_train, y_train)
    y_predict_lin = X_test.dot(beta)
    y_test_predict_np = y_predict_lin.to_numpy()
    y_test_np = y_test.to_numpy()

    MAE_closed = (1 / len(y_test_np)) * np.sum(np.abs(y_test_predict_np - y_test_np))
    mae_module = mean_absolute_error(y_test_np, y_test_predict_np)
    RMSE_closed = np.sqrt((1 / len(y_test_np)) * np.sum(np.square(y_test_predict_np - y_test_np)))
    r2_closed = r2_score(y_test_np, y_test_predict_np)

    if False:
        print("\nLinear regression closed form solution")
        print("Beta =", beta)
        print("MAE:", MAE_closed)
        if not MAE_closed==mae_module:
            print("MAE:", mae_module)
        print("RMSE:", RMSE_closed)
        print("R-squared:", r2_closed)

    # Optimize the level of regularization
    if _lambda_opt:
        opt_lambda_L1, opt_lambda_L2 = lambda_optimization('linear', model, X_train, X_test, y_train, y_test, plots=plots)
    else:
        # The optimization has given the results: 
        if model == 'model1':
            opt_lambda_L1, opt_lambda_L2 = 1.9, 572
        else:
            opt_lambda_L1, opt_lambda_L2 = 1, 1 # Small values to regularize still, the optimal was 0
        
        print(f'\nL1 regularization optimization of lambda parameter: {opt_lambda_L1}')
        print(f'\nL2 regularization optimization of lambda parameter: {opt_lambda_L2}')

    # Evaluate both regularization methods on the testing dataset
    beta_L1, y_predict_lin_L1, MAE_closed_L1, RMSE_closed_L1, r2_closed_L1 = prediction_closed_form_L1(X_train, X_test, y_train, y_test, _lambda = opt_lambda_L1, _print = False)
    beta_L2, y_predict_lin_L2, MAE_closed_L2, RMSE_closed_L2, r2_closed_L2 = prediction_closed_form_L2(X_train, X_test, y_train, y_test, _lambda = opt_lambda_L2, _print = False)

    if plots: 
        model_metrics = pd.DataFrame(data=[MAE_closed, RMSE_closed, r2_closed], index=['MAE', 'RMSE', 'R-squared'] , columns=['Linear'])
        model_metrics['L1 regularized'] = [MAE_closed_L1, RMSE_closed_L1, r2_closed_L1]
        model_metrics['L2 regularized'] = [MAE_closed_L2, RMSE_closed_L2, r2_closed_L2]
        print('\nEvaluation metrics for linear regression:\n', model_metrics)
    
        feature_weights = pd.DataFrame(beta, columns=['Weights'])
        feature_weights['L1 regularized'] = beta_L1
        feature_weights['L2 regularized'] = beta_L2.flatten()
        print('\nFeature weights for linear regression:\n', feature_weights)


    # ----------------------------------------------------------------------------------------
    # Get training and testing set for our non-linear model
    X_train, X_test, y_train, y_test = split_data(
        df=data, 
        features=features, 
        targets=target
    )

    # Compute the non-linear regression of this model
    print("\nNon-linear regression solution")
    poly_features_train, poly_features_test, y_pred_nl, feature_weights_nl = nonlinear_regression(data, features=features, targets=target, degrees=[1, 3], n_samples=len(data), plots=False)
    X_train, X_test = poly_features_train, poly_features_test
    y_test_np = y_test.to_numpy()

    MAE_closed = (1 / len(y_test_np)) * np.sum(np.abs(y_pred_nl - y_test_np))
    mae_module = mean_absolute_error(y_test_np, y_pred_nl)
    RMSE_closed = np.sqrt((1 / len(y_test_np)) * np.sum(np.square(y_pred_nl - y_test_np)))
    r2_closed = r2_score(y_test_np, y_pred_nl)

    if False:
        print("MAE:", MAE_closed)
        if not MAE_closed==mae_module:
            print("MAE:", mae_module)
        print("RMSE:", RMSE_closed)
        print("R-squared:", r2_closed)

    # Optimize the level of regularization (lambda)
    if _lambda_opt:
        opt_lambda_L1_nl, opt_lambda_L2_nl = lambda_optimization('nonlinear', model, X_train, X_test, y_train, y_test, plots=plots)
    else:
        # The optimization has given the results: 
        if model == 'model1':
            opt_lambda_L1_nl, opt_lambda_L2_nl = 0.01, 13
        else:
            opt_lambda_L1_nl, opt_lambda_L2_nl = 33185, 6

        print(f'\nL1 regularization optimization of lambda parameter: {opt_lambda_L1_nl}')
        print(f'\nL2 regularization optimization of lambda parameter: {opt_lambda_L2_nl}')

    # Evaluate both regularization methods on the testing dataset
    beta_L1_nl, y_predict_nonlin_L1, MAE_closed_L1_nl, RMSE_closed_L1_nl, r2_closed_L1_nl = prediction_closed_form_L1(X_train, X_test, y_train, y_test, _lambda = opt_lambda_L1_nl, _print = False)
    beta_L2_nl, y_predict_nonlin_L2, MAE_closed_L2_nl, RMSE_closed_L2_nl, r2_closed_L2_nl = prediction_closed_form_L2(X_train, X_test, y_train, y_test, _lambda = opt_lambda_L2_nl, _print = False)

    if plots: 
        model_metrics_nl = pd.DataFrame(data=[MAE_closed, RMSE_closed, r2_closed], index=['MAE', 'RMSE', 'R-squared'] , columns=['Non-Linear'])
        model_metrics_nl['L1 regularized'] = [MAE_closed_L1_nl, RMSE_closed_L1_nl, r2_closed_L1_nl]
        model_metrics_nl['L2 regularized'] = [MAE_closed_L2_nl, RMSE_closed_L2_nl, r2_closed_L2_nl]
        print('\nEvaluation metrics for non-linear regression:\n', model_metrics_nl)
    
        feature_weights_nl['L1 regularized'] = beta_L1_nl
        feature_weights_nl['L2 regularized'] = beta_L2_nl.flatten()
        print('\nFeature weights for polynomial degree 3:\n', feature_weights_nl)

    return y_predict_lin, y_predict_lin_L1, y_predict_lin_L2, y_pred_nl, y_predict_nonlin_L1, y_predict_nonlin_L2