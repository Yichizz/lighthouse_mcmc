"""!@file main_partv.py
@brief This is the main page for the solutions for lighthouse problem from part iii) to part v) in the report.

@details This contains all the codes you need to reproduce all the figures and results in the report.
Usage: python main_partv.py --nsteps 10000
@author Created by Yichi Zhang (yz870) on 18/03/2024
"""

# Import the necessary packages
from helpers.mcmc_runners import metropolis_hasting
from helpers.utils import compare_mle, plot_posterior_2d, plot_histogram
from helpers.diagonistics import chain_plotter, trace_plotter, geweke_test
import os
import time
import numpy as np
import arviz as az
from emcee.autocorr import integrated_time
from scipy.stats import norm, loguniform

# get number of steps from the command line
import argparse
parser = argparse.ArgumentParser(description='Run the MCMC algorithm for the lighthouse problem')
parser.add_argument('--nsteps', type=int, help='number of steps for the MCMC algorithm')
if parser.parse_args().nsteps is not None:
    nsteps = parser.parse_args().nsteps
else:
    nsteps = 10000

# load locations and intensities of the lighthouse flashes
# if there's a data file 'lighthouse_flash_data.txt', load the data from the file
try:
    data = np.loadtxt('lighthouse_flash_data.txt')
    print('Data loaded from file')
except:
    raise FileNotFoundError('Data file not found')

locations = data[:,0]

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
    return likelihood(x, alpha, beta) * loguniform.pdf(beta, 0.01, 5) * norm.pdf(alpha, 0,4)

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

print('Part iv) We use a normal prior for alpha and a log-uniform prior for beta')
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
chain, num_accept = metropolis_hasting(nsteps, log_posterior, 1, 2, np.array([0, 1]), locations)
end_time = time.time()
time_taken = end_time - begin_time
print(f'running time: {time_taken} seconds')
print(f'acceptance rate: {num_accept/nsteps*100}%')
print(f'time per step: {time_taken/nsteps} seconds')

# convergence diagonistics: informal
# plot the chain and trace
fig2 = chain_plotter(chain, ['alpha', 'beta'])
fig2.savefig('figures/chain_plot.png')
fig3 = trace_plotter(chain, posterior, ['alpha', 'beta'], steps = 150, data = locations)
fig3.savefig('figures/trace_plot.png')

# convergence diagonistics: formal
# first, we detect burn-in using geweke test
z_scores_alpha = geweke_test(chain[:,0,0], intervals = 20)
z_scores_beta = geweke_test(chain[:,0,1], intervals = 20)
# burn-in is the smallest number of steps such that the z-scores are without significant trend
for n, z in z_scores_alpha:
    if abs(z) > 1.96:
        burn_in_alpha = n
        break
        
for n, z in z_scores_beta:
    if abs(z) > 1.96:
        burn_in_beta = n
        break

burn_in = max(burn_in_alpha, burn_in_beta)
print(f'burn-in detected period from the geweke test: {burn_in}')

# second, we detect convergence using gelman-rubin test
# we run 5 independent chains for different starting points
np.random.seed(0)
chains = []
print('running 5 independent chains')
for i in range(5):
    chain_i, _ = metropolis_hasting(nsteps, log_posterior, 1, 2, np.random.rand(2), locations)
    chain_i = chain_i.reshape(2, nsteps,1)
    # discard the burn-in
    chain_i = chain_i[:,burn_in:,:] 
    chains.append(chain_i) 

# convert the chains to np.array shape (5, nsteps-burn_in, 2)
chains = np.concatenate(chains, axis=2).transpose(2,1,0)
chains= az.convert_to_dataset(chains, group='posterior')
# calculate the Gelman-Rubin statistic
r_hat = az.rhat(chains)
print(f'Gelman-Rubin statistic: {r_hat['x'].values}')
# if the r_hat is close to 1, the chains are converged

# compute the autocorrelation time
tau = np.max([integrated_time(chain[:,0,k])for k in [0,1]])
print(f'autocorrelation time: {tau}')
# compute the effective sample size
iid_samples = chain[burn_in::2*int(tau), 0, :]
num_iid_samples = iid_samples.shape[0]
print(f'effective sample size: {num_iid_samples}')
print(f'percentage of effective sample size: {num_iid_samples/nsteps*100}%')
print(f'time per effective sample: {time_taken/num_iid_samples} seconds')

# histogram of the joint and marginal samples
fig = plot_histogram(iid_samples, ['alpha', 'beta'], contour=True)
fig.savefig('figures/histogram.png')
print(f'joint and marginal samples saved to figures/histogram.png')

# show estimated mean and standard deviation
print('---------Final Results---------')
print(f'alpha {np.mean(iid_samples[:,0])} +/- {np.std(iid_samples[:,0])}')
print(f'beta {np.mean(iid_samples[:,1])} +/- {np.std(iid_samples[:,1])}')