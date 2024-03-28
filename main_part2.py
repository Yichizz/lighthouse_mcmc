"""!@file main_partvii.py
@brief This is the main page for the solutions for lighthouse problem from part vi) to part viii) in the report.

@details This contains all the codes you need to reproduce all the figures and results in the report.
Usage: python main_part2.py --nsteps 10000
@author Created by Yichi Zhang (yz870) on 19/03/2024
"""

import os
import time
import numpy as np
import arviz as az
import emcee
from emcee.autocorr import integrated_time
from scipy.stats import norm, loguniform
from helpers.utils import plot_histogram
from helpers.diagonistics import geweke_test
import matplotlib.pyplot as plt

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
intensity = data[:,1]

# define our model for the lighthouse problem
print('Part vi) We use a lognormal prior for the intensity parameter I_0')
@np.vectorize
def cauchy_pdf(x : float, alpha : float, beta : float) -> float:
    """@brief Cauchy distribution pdf

        @details This function returns the probability density function of the Cauchy distribution at a given point x.

        @param x: float, the point at which to evaluate the pdf
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
    """
    return beta / (np.pi * (beta**2 + (x - alpha)**2))

@np.vectorize
def log_normal_pdf(x : float, mu : float, sigma : float) -> float:
    """@brief Log-normal distribution pdf

        @details This function returns the probability density function of the log-normal distribution at a given point x.

        @param x: float, the point at which to evaluate the pdf
        @param mu: float, the location parameter
        @param sigma: float, the scale parameter
    """
    if x <= 0:
        return 0
    else:
        return 1/(x*sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(x)-mu)/sigma)**2)
    
def prior(I_0 : float, alpha : float, beta : float, I_0_min : float) -> float:
    """@brief Prior distribution

        @details This function returns the prior distribution of the parameters I_0, alpha and beta.
        We assume a log-normal prior for I_0, a normal prior for alpha and a log-uniform prior for beta.

        @param I_0: float, the intensity parameter
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
        @param I_0_min: float, the minimum value of I_0
    """
    if I_0 < I_0_min:
        return 0
    else:
        I_0_shifted = I_0 - I_0_min
        return log_normal_pdf(I_0_shifted, 2, 1) * loguniform.pdf(beta, 0.01, 5) * norm.pdf(alpha, 0,4)
    
def likelihood(positions : np.array, intensity : np.array, I_0 : float, alpha : float, beta : float) -> float:
    """@brief Likelihood function

        @details This function returns the likelihood of the data given the parameters I_0, alpha and beta.

        @param positions: np.array, the locations of the lighthouse flashes
        @param intensities: np.array, the intensities of the lighthouse flashes
        @param I_0: float, the intensity parameter
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
    """
    likelihood_locations = np.prod([cauchy_pdf(xk, alpha, beta) for xk in positions], axis=0)

    distance = beta**2 + (positions - alpha)**2
    mu = np.log(I_0) - np.log(distance)
    likelihood_intensities = np.prod([log_normal_pdf(intensity[i], mu[i], 0.1) for i in range(len(intensity))], axis=0)
    return likelihood_locations * likelihood_intensities

def posterior(positions : np.array, intensities : np.array, I_0 : float, alpha : float, beta : float) -> float:
    """@brief Posterior distribution

        @details This function returns the posterior distribution of the parameters I_0, alpha and beta given the data.

        @param positions: np.array, the locations of the lighthouse flashes
        @param intensities: np.array, the intensities of the lighthouse flashes
        @param I_0: float, the intensity parameter
        @param alpha: float, the location parameter
        @param beta: float, the scale parameter
        @param I_0_min: float, the minimum value of I_0
    """
    # absolute intensity of the lighthouse should be at least the highest intensity observed
    I_0_min = np.max(intensities) 
    return likelihood(positions, intensities, I_0, alpha, beta) * prior(I_0, alpha, beta, I_0_min)


# part vii) we use emcee to sample from the posterior distribution
def log_posterior(params : np.array, locations : np.array, intensities : np.array) -> float:
    """@brief Log posterior distribution

        @details This function returns the log posterior distribution of the parameters I_0, alpha and beta given the data.

        @param params: np.array, the parameters I_0, alpha and beta
        @param locations: np.array, the locations of the lighthouse flashes
        @param intensities: np.array, the intensities of the lighthouse flashes
    """
    I_0, alpha, beta = params
    if I_0 < np.max(intensities) or beta < 0.01 or beta > 5:
        return -np.inf
    else:
        I_0_shifted = I_0 - np.max(intensities)
        log_prior = np.log(log_normal_pdf(I_0_shifted, 2, 1)) + np.log(norm.pdf(alpha, 0, 4)) + np.log(loguniform.pdf(beta, 0.01, 5))
        log_likelihood_x = np.sum([np.log(cauchy_pdf(xk, alpha, beta)) for xk in locations])
        distance = beta**2 + (locations - alpha)**2
        mu = np.log(I_0) - np.log(distance)
        log_likelihood_I = np.sum([np.log(log_normal_pdf(intensity[i], mu[i], 1)) for i in range(len(intensity))])
        return log_prior + log_likelihood_x + log_likelihood_I
         
print('Part vii) We use emcee to sample from the posterior distribution')
nwalkers, ndim = 10, 3
print(f'running emcee with {nwalkers} walkers and {nsteps} steps')
np.random.seed(0)
start_i0 = np.random.uniform(8,16, nwalkers)
start_alpha = np.random.uniform(0,2, nwalkers)
start_beta = np.random.uniform(1,5, nwalkers)
start = np.column_stack([start_i0, start_alpha, start_beta])
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[locations, intensity])
t_start = time.time()
sampler.run_mcmc(start, nsteps)
t_end = time.time()
print(f"Time taken = {t_end - t_start} seconds")
print(f"Time per iteration = {(t_end - t_start)/nsteps} seconds")
print(f"Acceptance fraction = {np.mean(sampler.acceptance_fraction)}")

# convergence diagonistics: informal
# plot the chain 
# generated by the github copilot
fig, axs = plt.subplots(3, 1, figsize=(20, 6))

for i in range(nwalkers):
    axs[0].plot(sampler.chain[i,:,0])
    axs[1].plot(sampler.chain[i,:,1])
    axs[2].plot(sampler.chain[i,:,2])

axs[0].set_title('I0 chain')
axs[1].set_title('alpha chain')
axs[2].set_title('beta chain')
plt.tight_layout()
fig.savefig('figures/chains.png')

# convergence diagonistics: formal
# first, we detect burn-in using geweke test
burn_in = []
for c, chain in enumerate(sampler.chain):
    num_steps, num_dim = chain.shape
    for i in range(num_dim):
        z_scores = geweke_test(chain[:,i], intervals=20)
        for n, z in z_scores:
            if np.abs(z) > 1.96 and n>0:
                burn_in.append(n)
                break

print(f'average burn-in detected from geweke test: {int(np.ceil(np.mean(burn_in)))}')
discard = int(np.ceil(np.mean(burn_in)))
        
# second, we detect convergence using gelman-rubin test
chains = sampler.get_chain(discard=discard).transpose(1,0,2)
chains= az.convert_to_dataset(chains, group='posterior')
# calculate the Gelman-Rubin statistic
r_hat = az.rhat(chains)
# if the r_hat is close to 1, the chains are converged
print(f'Gelman-Rubin statistic: {r_hat["x"].values}')

# compute the autocorrelation time
taus = sampler.get_autocorr_time(tol=2, discard=discard)
tau = np.max(taus)
print(f'autocorrelation time: {tau}')

# compute the effective sample size
iid_samples = sampler.get_chain(discard=discard, thin=int(tau), flat=True)
num_iid_samples = iid_samples.shape[0]

time_taken = t_end - t_start
print(f'effective sample size: {num_iid_samples}')
print(f'percentage of effective sample size: {num_iid_samples/nsteps*100}%')
print(f'time per effective sample: {time_taken/num_iid_samples} seconds')

# histogram of the joint and marginal samples
fig = plot_histogram(iid_samples, ['I0', 'alpha', 'beta'], contour=True)
fig.savefig('figures/histogram_3d.png')
print(f'joint and marginal samples saved to figures/histogram.png')

# show estimated mean and standard deviation
print('---------Final Results---------')
print(f'I_0 {np.mean(iid_samples[:,0])} +/- {np.std(iid_samples[:,0])}')
print(f'alpha {np.mean(iid_samples[:,1])} +/- {np.std(iid_samples[:,1])}')
print(f'beta {np.mean(iid_samples[:,2])} +/- {np.std(iid_samples[:,2])}')