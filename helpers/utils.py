"""!@file yz870/helpers/utils.py
@brief Utility functions for the lighthouse problem

@details This file contains the utility functions for the lighthouse problem, 
including
- a function to compare the maximum likelihood estimates of alpha and mean of the data
- a function to plot the 2D posterior distribution
- a function to plot the histogram of the samples from the MCMC algorithm

@author Created by Yichi Zhang (yz870) on 18/03/2024
"""

# Import the necessary packages
import typing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def compare_mle(data : np.array, nll : typing.Callable) -> None:
    """@brief Compare the maximum likelihood estimates of alpha and mean of the data

        @details This function compares the maximum likelihood estimates of alpha and the mean of the data.
        It prints the MLEs of alpha and the mean and medium of the data.

        @param data: np.array, the data used to estimate the parameters
        @param likelihood: function, the likelihood function
    """

    # initial guess
    params0 = [0, 1]

    # minimize the negative log likelihood
    res = minimize(nll, params0, args=(data,))
    alpha_mle, _ = res.x

    # estimate the mean, medium of the data
    mean = np.mean(data)
    medium = np.median(data)

    # print the estimates and the difference between them
    print(f'MLE of alpha: {alpha_mle}, mean of the data: {mean}, medium of the data: {medium}')
    return None

def plot_posterior_2d(data : np.array, posterior : typing.Callable, alpha_range : np.array, beta_range : np.array) -> plt.figure:
    """@brief Plot the 2D posterior distribution

        @details This function plots the 2D posterior distribution of the parameters alpha and beta.
        The range being plotted is specified by alpha_range and beta_range.

        @param data: np.array, the data used to estimate the parameters
        @param posterior: function, the posterior distribution
        @param alpha_range: np.array, the range of alpha values
        @param beta_range: np.array, the range of beta values

        @return fig: plt.figure, the figure object
    """

    # evaluate the posterior distribution
    alpha = np.linspace(1.6-4*5, 1.6+4*5, 100)
    beta = np.linspace(0.01, 10, 100)
    alpha, beta = np.meshgrid(alpha, beta)
    posterior_vals = posterior(data, alpha, beta)

    fig = plt.figure()
    plt.contour(alpha, beta, posterior_vals, colors='black', linewidths=1)
    plt.contourf(alpha, beta, posterior_vals, 100, cmap='GnBu')
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.xlim(alpha_range[0], alpha_range[1])
    plt.ylim(beta_range[0], beta_range[1])
    plt.colorbar()
    plt.tight_layout()

    return fig

def plot_histogram(samples : np.array, labels : list[str], contour = True) -> plt.figure:
    """@brief Plot the histogram of the samples

        @details This function plots the histogram of the samples.

        @param samples: np.array, the samples
        @param labels: list[str], the labels for the histogram
        @param kde: bool, whether to plot the kernel density estimatE

        @return fig: plt.figure, the figure object
    """
    df = pd.DataFrame({label: samples[:,i] for i, label in enumerate(labels)})
    fig  = sns.pairplot(df, kind='hist', corner=True, plot_kws={'color': 'lightblue'}, diag_kws={'color': 'lightblue'})
    if contour:
        fig.map_lower(sns.kdeplot, levels=[0.1,0.3,0.5,0.7,0.9], colors = 'darkblue')
    plt.tight_layout()
    return fig