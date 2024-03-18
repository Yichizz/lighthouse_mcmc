"""!@file yz870/helpers/diagnostics.py
@brief This file contains useful diagonistics for the markov chains generated by the MCMC algorithm

@details This file contains functions to informally and formally diagnose the convergence of the MCMC algorithm
to the target distribution. 
The functions include:
- chain_plotter
- trace_plotter
- geweke_test
- gelman_rubin_test
- effective_sample_size

@author Created by Yichi Zhang (yz870) on 18/03/2024
"""

import typing
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import yule_walker

def chain_plotter(chain : np.array, labels : typing.List[str]) -> plt.figure:
    """@brief Plot the chains

        @details This function plots the chains generated by the MCMC algorithm.
        Different components of the chain are plotted in different subplots.

        @param chain: np.array, the generated chain
        @param labels: list, the labels of the parameters

        @return fig: plt.figure, the figure object
    """

    nsteps, nchains, ndim = chain.shape
    fig, ax = plt.subplots(ndim, 1, figsize=(10, 2*ndim))
    for i in range(ndim):
        ax[i].plot(chain[:,0,i])
        ax[i].set_ylabel(labels[i])
    ax[-1].set_xlabel('Step')
    plt.tight_layout()
    return fig

def trace_plotter(chain : np.array, posterior : typing.Callable, labels : typing.List[str], 
                  steps : int = 100, data : np.array = None) -> plt.figure:
    """@brief Plot the traces

        @details This function plots the traces of the chains generated by the MCMC algorithm.
        The traces are plotted upon the posterior distribution of the parameters.

        @param chain: np.array, the generated chain
        @param posterior: function, the posterior distribution
        @param labels: list, the labels of the parameters
        @param steps: int, the number of steps to plot
        @param data: np.array, the data used to estimate the parameters

        @return fig: plt.figure, the figure object
    """

    nsteps, nchains, ndim = chain.shape
    fig = plt.figure()
    
    alpha = np.linspace(1.6-4*5, 1.6+4*5, 100)
    beta = np.linspace(0.01, 10, 100)
    alpha, beta = np.meshgrid(alpha, beta)
    posterior_vals = posterior(data, alpha, beta)

    plt.contour(alpha, beta, posterior_vals, colors='black', linewidths=1)
    plt.contourf(alpha, beta, posterior_vals, 100, cmap='GnBu')
    
    # add first steps of the chain
    plt.plot(chain[:steps,0,0], chain[:steps,0,1], 'r-')
    plt.scatter(chain[:steps,0,0], chain[:steps,0,1], c='r')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend([f'first {steps} steps'])
    plt.xlim(-2,2)
    plt.ylim(0,4)
    plt.colorbar()
    plt.tight_layout()

    return fig

def spec(x : np.array) -> float:
    """@brief Spectral density

        @details This function returns the spectral density of the input time series.

        @param x: np.array, the input time series

        @return float, the spectral density
    """

    rho, sigma = yule_walker(x, 1)
    return sigma**2 / (1 - rho**2)

def geweke_test(chain : np.array, first : float = 0.1, last : float = 0.5, intervals : int = 20) -> typing.List[typing.Tuple[int, float]]:
    """@brief Geweke test

        @details This function runs the Geweke test to informally diagnose the convergence of the MCMC algorithm.
        The Geweke test compares the mean of the first % of the series with the mean of the last % of the series.
        If the series is converged, this score should oscillate between -1 and 1.

        @param chain: np.array, the generated chain
        @param first: float, the fraction of series at the beginning of the trace
        @param last: float, the fraction of series at the end to be compared with the section at the beginning
        @param intervals: int, the number of segments

        @return z_scored: np.array, the z-scores
    """

    if np.ndim(chain) > 1:
        return [geweke_test(y, first, last, intervals) for y in np.transpose(chain)]

    # Filter out invalid intervals
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first, last))

    # Initialize list of z-scores
    zscores = [None] * intervals

    # Starting points for calculations
    starts = np.linspace(0, int(len(chain)*(1.-last)), intervals).astype(int)

    # Loop over start indices
    for i,s in enumerate(starts):

        # Size of remaining array
        chain_trunc = chain[s:]
        n = len(chain_trunc)

        # Calculate slices
        first_slice = chain_trunc[:int(first * n)]
        last_slice = chain_trunc[int(last * n):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(spec(first_slice)/len(first_slice) +
                     spec(last_slice)/len(last_slice))
        zscores[i] = len(chain) - n, z

    return zscores
