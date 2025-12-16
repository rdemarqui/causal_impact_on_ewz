import pandas as pd
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


def synthetic_control_weights(Y1_pre, Y0_pre):
    """
    Compute donor weights for Synthetic Control following Abadie's restrictions.
    
    Parameters
    ----------
    Y1_pre : array-like, shape (T,)
        Pre-intervention outcome series for the treated unit.
        
    Y0_pre : array-like, shape (T, K)
        Pre-intervention outcome series for the donor pool (K donors).
    
    Returns
    -------
    weights : ndarray, shape (K,)
        Optimized donor weights that minimize the pre-treatment MSE.
    """
    
    k = Y0_pre.shape[1]  # number of donors

    # Objective function: minimize MSE
    def objective(w):
        Y_synth = w @ Y0_pre.T
        return np.sum((Y1_pre - Y_synth) ** 2)

    # Constraint: sum of weights must be 1
    cons = ({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    })

    # Bounds: weights >= 0
    bounds = [(0, 1) for _ in range(k)]

    # Initial guess: uniform distribution
    w0 = np.ones(k) / k

    # Optimization
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons
    )

    return res.x



def plot_synthetic_weights(weights_dict, title='Synthetic Control Weights', figsize=(10,4)):
    """
    Plot the weights of the selected donor units in a synthetic control.

    Parameters
    ----------
    weights_dict : dict
        Dictionary with donor names as keys and weights as values.
    title : str, default 'Synthetic Control Weights'
        Title of the plot.
    figsize : tuple, default (10,4)
        Figure size.
    """

    df_weights = (
        pd.DataFrame(list(weights_dict.items()), columns=['ETF', 'Weight'])
          .sort_values('Weight', ascending=False)
    )

    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=df_weights,
        x='ETF',
        y='Weight',
        s=80,
        color='lightgrey',
        edgecolor='black'
    )

    plt.ylabel('Weight')
    plt.xlabel('')
    plt.title(title)
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()



def placebo_test(donors, treatment_date, weight_calculus="canonical"):
    """
    Create a dictionary with synthetic value for each donor
    """
    placebo = {}

    for donor in donors.columns:
        # Separate treated and donors
        treated_d = donors[donor]
        donors_d = donors.drop(columns=[donor])

        # Separate pre treatment dataset
        Y1_pre_d = treated_d[treated_d.index < treatment_date].values
        Y0_pre_d = donors_d[donors_d.index < treatment_date].values

        if weight_calculus == 'ols':
            # OLS without intercept
            ols_model = sm.OLS(Y1_pre_d, Y0_pre_d).fit()
            weights_d = ols_model.params
        else:
            # Compute donor weights for Synthetic Control following Abadie's restrictions
            weights_d = synthetic_control_weights(Y1_pre_d, Y0_pre_d)

        # Apply weights
        synth_abadie_d = weights_d @ donors_d.T

        placebo[donor] = treated_d - synth_abadie_d

    return placebo



def mspe(y, y_synth):
    return float(np.mean((y - y_synth)**2))



def plot_mspe_ratio(mspe_plot, treated_label='EWZ', title="", figsize=(8,5)):
    """
    Plot MSPE ratios for placebo tests, highlighting the treated unit.

    Parameters
    ----------
    mspe_plot : pd.DataFrame
        DataFrame indexed by unit (e.g. ETF tickers) with a column 'mspe_ratio'.
        The DataFrame should already be sorted in ascending order.
    treated_label : str, default 'EWZ'
        Label of the treated unit to be highlighted.
    figsize : tuple, default (8,5)
        Figure size.
    """

    # Flag treated unit
    mspe_plot = mspe_plot.copy()
    mspe_plot['is_treated'] = mspe_plot.index == treated_label

    plt.figure(figsize=figsize)

    sns.scatterplot(
        data=mspe_plot,
        x=mspe_plot.index,
        y='mspe_ratio',
        hue='is_treated',
        palette={False: 'lightgrey', True: 'red'},
        s=80,
        edgecolor='black',
        legend=False
    )

    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('MSPE ratio')
    plt.title(title)

    plt.tight_layout()
    plt.show()