"""!@file yz870/helpers/mcmc_runners.py
@brief This file contains the classes and functions to run MCMC simulations

@details run markov chain monte carlo simulations to estimate the parameters of the lighthouse problem
@author Created by Yichi Zhang (yz870) on 18/03/2024
"""

# Import the necessary packages
import tqdm
import typing   
import numpy as np
from scipy.stats import multivariate_normal

def metropolis_hasting(nsteps : int, log_target : typing.Callable, delta : float,
                    ndim : int, start : np.array, data : np.array) -> typing.Tuple[np.array, int]:
    
    """!@brief Metropolis-Hasting algorithm

        @details This function runs the Metropolis-Hasting algorithm to generate stochastic samples from a given distribution.
        The proposal distribution is a multivariate normal distribution with mean at the current position and covariance matrix delta*I.

        @param nsteps: int, number of steps to run the algorithm
        @param log_target: function, the log of the target distribution
        @param delta: float, the variance of the multivariate normal proposal distribution
        @param ndim: int, the dimension of the target distribution
        @param start: np.array, the starting position of the chain
        @param data: np.array, the data used to evaluate the target distribution

        @return chain: np.array, the generated chain
        @return num_accept: int, the number of proposals accepted
    """

    assert start.shape[0] == ndim, 'The dimension of the starting position does not match the dimension of the target distribution'

    cov = delta * np.eye(2)
    chain = np.zeros((nsteps, 1, ndim))
    chain[0,0] = start
    num_accept = 0
    for i in tqdm.tqdm(range(nsteps-1)):
        x_current = chain[i,0]                     # current position
        Q = multivariate_normal(x_current, cov)    # poposal distribution
        x_proposed = Q.rvs() 
        log_a = log_target(x_proposed, data = data) - \
                    log_target(x_current, data = data)            # acceptance ratio
        u = np.random.uniform()                    # uniform random variable
        if np.log(u)<log_a:
            x_new = x_proposed                     # ACCEPT
            num_accept += 1                        # count how many proposals are accepted
        else:
            x_new = x_current                      # REJECT
        chain[i+1,0] = x_new                       # store new position in chain
    return chain, num_accept                     # store new position in chain

