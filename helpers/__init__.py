"""
helpers
=======

Provides:
    1. mcmc_runners.py: Run Metropolis-Hasting algorithm
    2. diagonistics.py: Diagnostics functions for samples from the MCMC algorithm
    3. utils.py: Utility functions for the lighthouse problem

Dependencies:
-------------
- numpy
    The fundamental package for scientific computing with Python.
    - Website: https://numpy.org/
- typing
    Support for type hints.
    - Website: https://docs.python.org/3/library/typing.html
- scipy
    A scientific computing library for Python.
    - Website: https://www.scipy.org/
- seaborn
    A Python visualization library based on matplotlib.
    - Website: https://seaborn.pydata.org/
- matplotlib
    A comprehensive library for creating static, animated, and interactive visualizations in Python.
    - Website: https://matplotlib.org/
- pandas
    A fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation library built on top of the Python programming language.
    - Website: https://pandas.pydata.org/
- arviz
    A Python package for exploratory analysis of Bayesian models.
    - Website: https://arviz-devs.github.io/arviz/
- emcee
    The Python ensemble sampling toolkit for affine-invariant MCMC.
    - Website: https://emcee.readthedocs.io/en/stable/

Examples:
---------
Usages:
    Import the necessary submodules from the helpers package
    >>> from helpers.mcmc_runners import metropolis_hasting
    >>> from helpers.diagonistics import geweke_test
    >>> from helpers.utils import plot_histogram

    Run the Metropolis-Hasting algorithm
    >>> chain, num_accept = metropolis_hasting(nsteps, log_target, delta, ndim, start, data)

    Run the Geweke test
    >>> zscores = geweke_test(chain, intervals)

    Plot the histogram of the samples from the MCMC algorithm
    >>> plot_histogram(chain, nsteps)

Notes:
------
- The helpers package is designed to provide utility functions for the lighthouse problem.
- For help on the functions, use the help() function in Python.
"""