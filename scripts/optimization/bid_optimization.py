import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



def bid_optimization(data, dates, y, wind_capacity=6000, plots=True):
    """
    Compute the optimal power to bid in the day-ahead market for each hour of the day.

    Args:
        data (panda.dataframe): dataframe of the problem data
        dates (panda.dataframe): 1D dataframe of selected dates for which to optimize our problem
        y (panda.dataframe): 1D dataframe of power predictions for the selected dates 
        wind_capacity (int): capacity of the wind farm
        plots (bool): plots are not printed if set to False 

    Returns:
        (opt_power_DA, opt_revenue) (list, float):    List of hourly optimal bidding power 
            for the day-ahead market. Expected revenue based on power predictions.
    """
    data = data.reset_index()  # 'ts' will now be a column if it was the index
    data = data.loc[data['ts'].isin(dates)]
    price_DA, price_up, price_down = list(data['SpotPriceDKK']), list(data['BalancingPowerPriceUpDKK']), list(data['BalancingPowerPriceDownDKK'])
    
    # Get the power predictions
    if not (isinstance(y, np.ndarray)): # it would be a single-column panda.dataframe otherwise
        power_predict = y.iloc[:, 0].to_numpy()
    elif y.ndim > 1:
        power_predict = y.flatten()
    else: 
        power_predict = y

    TIME = range(len(dates))

    model = gp.Model(name="Trading optimization problem")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 100 

    # Add variables to the Gurobi model
    power_DA = {
        t: model.addVar(
            lb=0, ub=wind_capacity, name=f"day-ahead bidding power at timestep {t}"
        )
        for t in TIME
    }

    delta = {
        t: model.addVar(
            lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=f"delta at timestep {t}"
        )
        for t in TIME
    }

    delta_up = {
        t: model.addVar(
            lb=0, ub=wind_capacity, name=f"delta_up at timestep {t}"
        )
        for t in TIME
    }

    delta_down = {
        t: model.addVar(
            lb=0, ub=wind_capacity, name=f"delta_down at timestep {t}"
        )
        for t in TIME
    }


    # Set objective function 
    objective = gp.quicksum(
            price_DA[t] * power_DA[t]
                + price_up[t] * delta_up[t]
                - price_down[t] * delta_down[t]
            for t in TIME
            )

    model.setObjective(objective, gp.GRB.MAXIMIZE)  # maximize revenue


    # Add constraints to the Gurobi model 
    delta_value = {
        t: model.addLConstr(
                delta[t],
                gp.GRB.EQUAL,
                power_predict[t] - power_DA[t],
                name=f"delta value at time {t}",
            )
        for t in TIME
    }

    delta_value_aux = {
        t: model.addLConstr(
                delta[t],
                gp.GRB.EQUAL,
                delta_up[t] - delta_down[t],
                name=f"auxiliary definition of delta at time {t}",
            )
        for t in TIME
    }

    
    # Optimize problem
    model.optimize()
    if model.status != GRB.Status.OPTIMAL:
        try:
            model.computeIIS()
        except gp.GurobiError:
            raise ValueError("The model is unbounded.")
        raise (ValueError("The model is infeasible"))
    
    opt_power_DA = [power_DA[t].x for t in TIME]
    opt_revenue = model.ObjVal

    if not plots:
        return opt_power_DA, opt_revenue
    
    # Plots ---------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot prices on the left y-axis (EUR)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Price (EUR)")
    ax1.plot(TIME, price_DA, label="Day-ahead Spot Price", color='blue', linestyle='-', marker='o')
    ax1.plot(TIME, price_up, label="Balancing Up Price", color='green', linestyle='--', marker='x')
    ax1.plot(TIME, price_down, label="Balancing Down Price", color='red', linestyle='--', marker='x')
    ax1.tick_params(axis='y')
    margin = 0.05 * np.max(price_DA)
    ax1.set_ylim(bottom = 0 - margin)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    # Create a second y-axis for power (MW)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Power (MW)")
    ax2.plot(TIME, power_predict, label="Power Prediction", color='orange', linestyle='-')
    ax2.plot(TIME, opt_power_DA, label="Optimized Bid", color='purple', linestyle='-')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0 - 0.05*wind_capacity, wind_capacity + 0.05*wind_capacity)

    # Add titles and grid
    plt.title("Trading Optimization: Prices (EUR) vs Power (MW) per Hour")
    fig.tight_layout()

    # Show legends for both y-axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show the plot
    plt.grid(True)
    plt.show()

    return opt_power_DA, opt_revenue




