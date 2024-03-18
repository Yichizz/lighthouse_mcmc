"""!@mainpage Lighthouse Problem
@brief This is the main page for the solutions for lighthouse problem

@details This contains all the codes you need to reproduce all the figures and results in the report.
@author Created by Yichi Zhang (yz870) on 18/03/2024
"""

# Import the necessary packages
from helpers.mcmc_runners import metropolis_hasting
from helpers.utils import compare_mle, plot_posterior_2d
from helpers.diagonistics import chain_plotter, trace_plotter
import os
import time
import typing 
import numpy as np
from scipy.stats import norm, loguniform

# load locations and intensities of the lighthouse flashes
# if there's a data file 'lighthouse_flash_data.txt', load the data from the file

try:
    data = np.loadtxt('lighthouse_flash_data.txt')
    print('Data loaded from file')
except:
    raise FileNotFoundError('Data file not found')

locations = data[:,0]
intensities = data[:,1]

# define our model for the lighthouse problem
@np.vectorize
def cauchy_pdf(x : float, alpha : float, beta : float) -> float:
    """@brief Cauchy distribution pdf

        @details This function returns the probability density function of the Cauchy distribution at a given point x.

        @param x: float, the point at which to evaluate the pdf
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
    """
    return beta / (np.pi * (beta**2 + (x - alpha)**2))

def likelihood(data : np.array, alpha : float, beta : float) -> float:
    """@brief Likelihood function

        @details This function returns the likelihood of the data given the parameters alpha and beta.

        @param data: np.array, the locations of the lighthouse flashes
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
    """
    return np.prod([cauchy_pdf(xk, alpha, beta) for xk in data], axis=0)

def posterior(x : np.array, alpha : float, beta : float) -> float:
    """@brief Posterior distribution

        @details This function returns the posterior distribution of the parameters alpha and beta given the data.
        We assume a log-uniform prior for beta and a normal prior for alpha.

        @param x: np.array, the locations of the lighthouse flashes
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
    """
    return likelihood(x, alpha, beta) * loguniform.pdf(beta, 0.01, 5) * norm.pdf(alpha, 1.6,4)

# part iii) numerically we compare the mle of alpha and mean and medium of the data
def nll(params : np.array, data : np.array) -> float:
    """@brief Negative log likelihood

        @details This function returns the negative log likelihood of the data given the parameters alpha and beta.

        @param params: np.array, the parameters alpha and beta
        @param data: np.array, the locations of the lighthouse flashes
    """
    alpha, beta = params
    return -np.sum(np.log(cauchy_pdf(data, alpha, beta)))
print('Part iii) Compare the MLE of alpha and the mean of the data')
compare_mle(locations, nll)

# part v) run the Metropolis-Hasting algorithm to draw samples from the posterior distribution
# we first investigate how the posterior distribution looks like
if not os.path.exists('figures'):
    os.makedirs('figures')

fig1 = plot_posterior_2d(locations, posterior, alpha_range = np.array([-2,2]), beta_range = np.array([0,4]))
fig1.savefig('figures/posterior_distribution_2d.png')

# define log of posterior distribution
def log_posterior(params : tuple, data : np.array) -> float:
    """@brief Log of the posterior distribution

        @details This function returns the log of the posterior distribution of the parameters alpha and beta given the data.

        @param params: tuple, the parameters alpha and beta
        @param data: np.array, the locations of the lighthouse flashes
    """
    alpha, beta = params
    if beta < 0.01 or beta > 5:
        return -np.inf 
    else:
        post = posterior(data, alpha, beta)
    return np.log(post)

# now, we run the mh algorithm 10000 steps
np.random.seed(0)
begin_time = time.time()
print("Part v) running the Metropolis-Hasting algorithm")
chain, num_accept = metropolis_hasting(10000, log_posterior, 1, 2, np.array([0, 1]), locations)
end_time = time.time()
time_taken = end_time - begin_time
print(f'running time: {time_taken} seconds')
print(f'acceptance rate: {num_accept/10000}')
print(f'time per step: {time_taken/10000} seconds')

# convergence diagonistics: informal
# plot the chain and trace
fig2 = chain_plotter(chain, ['alpha', 'beta'])
fig2.savefig('figures/chain_plot.png')
fig3 = trace_plotter(chain, posterior, ['alpha', 'beta'], steps = 150, data = locations)
fig3.savefig('figures/trace_plot.png')
# plot autocorrelation
