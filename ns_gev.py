# -*- coding: utf-8 -*-
"""Non-stationary General Extreme Value Distribution Parameter Estimation.

Notes
-----
* https://github.com/AusClimateService/unseen
* LMM method for guessing initial values not finished/tested.


References
----------
[1] Hosking, J.R.M. (1990), L-Moments: Analysis and Estimation of Distributions
    Using Linear Combinations of Order Statistics. Journal  of the Royal
    Statistical Society: Series B (Methodological),
    52: 105-124. https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
[2] Méndez, F. J., Menéndez, M., Luceño, A., & Losada, I. J. (2007). Analyzing Monthly
    Extreme Sea Levels with a Time-Dependent GEV Model. Journal of Atmospheric and O
    ceanic Technology (Vol. 24, Issue 5, pp. 894–911). American Meteorological Society.
    https://doi.org/10.1175/jtech2009.1
[3] Robin, Y., & Ribes, A. (2020). Nonstationary extreme value analysis for event
    attribution combining climate models and observations. Advances in Statistical
    Climatology, Meteorology and Oceanography (Vol. 6, Issue 2, pp. 205–221).
    Copernicus GmbH. https://doi.org/10.5194/ascmo-6-205-2020
[4] https://xiaoganghe.github.io/python-climate-visuals/chapters/data-analytics/scipy-basic.html


@author: Annette Stellema
@email: stellema@github.com
@created: Thu Sep 21 14:29:51 2023
"""
import dask
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import genextreme as genex
from statsmodels.regression.quantile_regression import QuantReg
import xarray as xr

import time
from functools import wraps


def timeit(func):
    """Print the execution time for the decorated function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('{}: {}s'.format(func.__name__, round(end - start, 2)))
        return result
    return wrapper


def linear_moments(x):
    """Calculate the first three sample L-moments.

    Based on [4]_.

    Parameters
    ----------
    x : array_like
        A 1D array of real values.

    Returns
    -------
    l1, l2, l3 : float
        The first three L-moments of x.
    """
    n = x.size

    # Probability weighted moments.
    b0 = np.mean(x)
    b1 = np.sum([(n - j - 1) * x[j] / n / (n - 1) for j in range(n)])
    b2 = np.sum([(n - j - 1) * (n - j - 2) * x[j] / n / (n - 1) / (n - 2)
                 for j in range(n - 1)])

    # Sample L-moments.
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * (b2 - b1) + b0

    return l1, l2, l3


def gev_params_fsolve(lm, t3):
    """Estimate GEV distribution parameters based on the sample L-moments.

    Based on [4]_.

    Parameters
    ----------
    lm : (M,) ndarray
        The first M L-moments.
    t3 : float
        The L-skewness (i.e., l3 / l2).

    Returns
    -------
    shape, loc, scale : float
       Estimated shape, location and scale parameters (shape, loc0, scale0)
    """
    f = lambda x, t: 2 * (1 - 3**(-x)) / (1 - 2**(-x)) - 3 - t
    shape = scipy.optimize.fsolve(f, 0.1, t3)[0]
    GAM = scipy.special.gamma(1 + shape)
    scale = lm[1] * shape / (GAM * (1 - 2**-shape))
    loc = lm[0] - scale * (1 - GAM) / shape
    return shape, loc, scale


class NonStationaryGEV(scipy.stats.rv_continuous):
    """Wrapper for non-stationary GEV (NS-GEV) distributions."""

    def _fitstart_genextremes(self, data, times):
        """Initialize NS-GEV parameters."""
        c, loc0, scale0 = genex.fit(data)
        return -c, loc0, 0, scale0, 0

    def _fitstart_guess(self, data, times):
        """Initiaze NS-GEV parameters using the mean, std and guesses.

        Parameters
        ----------
        data, times : array_like
            1D array of data (and times for consistency).
        const : float, optional
            Guess of the shape and trend parameters.

        Returns
        -------
        shape, loc0, loc1, scale0, scale1 : float
            The estimated shape, location and scale parameters.
        """
        return 0.1, np.mean(data), 0.1, np.std(data), 0.1

    def _fitstart_lmm(self, data, times):
        """Initialize NS-GEV parameters.

        Based on the Method of L-moments (LMM) (Hosking 1990; Robin and Ribes 2020).

        Parameters
        ----------
        data, times : ndarray
            1D arrays of data points and time indexes.

        Notes
        -----
        * Uses the L-moments and distribution param relationship (Hosking 1990).
        * Compute L-moments for the residuals based on predicted values from the
          quantile regression (Robin and Ribes 2020).
        """
        if isinstance(data, xr.DataArray):
            data = data.values  # !!! Must be ndarray for QuantReg

        # Step 1: Quantile regression
        quantiles = np.arange(0.05, 0.96, 0.01)
        models = [QuantReg(data, times).fit(q=q) for q in quantiles]
        residuals = np.array([data - model.predict(times) for model in models]).T

        # Step 2: Calculate L-moments using the results of the quantile regression
        # First three l-moments for each data timestep (i.e., first L-moment is lm[0])
        lm = np.array([linear_moments(residuals[i]) for i in range(times.size)]).T
        t3 = lm[2] / lm[1]  # L-skewness (l3 / l2)

        # Step 3: Linear regression of the L-moments
        params_values = np.array([gev_params_fsolve(lm[:, i], t3[i]) for i in range(times.size)]).T

        shape = np.mean(params_values[0])

        times_matrix = np.vstack([times, np.ones_like(times)]).T  # Coefficent matrix
        loc0, loc1 = np.linalg.lstsq(times_matrix, params_values[1], rcond=None)[0]
        scale0, scale1 = np.linalg.lstsq(times_matrix, params_values[2], rcond=None)[0]
        return shape, loc0, loc1, scale0, scale1


    def nllf(self, params, data, times):
        """NS-GEV penalliezd negative log-likelihood function.

        Parameters
        ----------
        params : tuple of floats
            Shape, location and scale parameters (shape, loc0, loc1, scale0, scale1).
        data : xarray.DataArray
            Data time series.
        times : numpy.array
            Indexes of covariate.

        Returns
        -------
        f_sum : float
            The negative likelihood function.

        Notes
        -----
        * We use a log-link transformation to ensure the scale parameter is positive.
            * Update: Using bounds in optimise because but this was causing weird results?
        * The NLLFs follow the equations in Méndez et al. (2007):
            - partially integrate the GEV CDF to get the PDF.
            - nllf = -sum(log(pdf))
        * Scipy: 'A large, finite penalty (rather than infinite negative log-likelihood)
        is applied for observations beyond the support of the distribution.''
        """
        if len(params) == 5:
            # Non-stationary GEV parameters.
            shape, loc0, loc1, scale0, scale1 = params
            loc = loc0 + loc1 * times
            scale = scale0 + scale1 * times

        else:
            # Stationary GEV parameters.
            shape, loc, scale = params

        # Negative Log-likelihood of GEV probability density function.
        s = (data - loc) / scale

        if shape != 0:
            Z = 1 + shape * s
            f = np.log(scale) + (1 + 1 / shape) * np.ma.log(Z) + np.ma.power(Z, (-1 / shape))
            f = np.where((scale > 0) & (Z > 0), f, np.inf)

        elif shape == 0:
            f = np.log(scale) + s + np.exp(-s)

        total, n_bad = scipy.stats._distn_infrastructure._sum_finite(f)
        return total + n_bad * scipy.stats._constants._LOGXMAX * 100

    @timeit
    def fit(self, data, times, start_method='genextremes', stationary=False):
        """Fit NS-GEV params by minimising the negative log-likelihood function.

        Parameters
        ----------
        data, times : array_like
            1D arrays of the data and covariate.
        start_method : ['genextremes', 'lmm'], optional
            Method to initialize parameters. The default is 'genextremes'.
        stationary : bool, optional
            Fit as a stationary GEV (no loc1, scale1). The default is False.

        Returns
        -------
        res : dict
            The optimization result (instance of scipy.optimize.OptimizeResult). Includes:
            x the optimal values, x0 the initial values, x_name the parameter names,
            success/status a Boolean flag, nfev number of function evaluations and fun the
            values of the objective function.

        Notes
        -----
        * Need to add test for invalid parameters before returning.
        """
        # Initial values of distribution parameters.
        if start_method == 'genextremes':
            params_i = self._fitstart_genextremes(data, times)
        elif start_method == 'lmm':
            params_i = self._fitstart_lmm(data, times)
        elif start_method == 'guess':
            params_i = self._fitstart_guess(data, times)

        bounds = [(None, None), (None, None), (None, None), (0, None), (None, None)]

        if stationary:
            # Don't optimise loc1 and scale1.
            params_i = [params_i[i] for i in [0, 1, 3]]
            bounds = [bounds[i] for i in [0, 1, 3]]

        # Minimise the negative log-likelihood function.
        res = scipy.optimize.minimize(self.nllf, params_i, args=(data, times),
                                      method='Nelder-Mead', bounds=bounds)

        res['x_name'] = ['c', 'μ0', 'μ1', 'σ0', 'σ1'] if not stationary else ['c', 'μ', 'σ']
        res['x0'] = np.array(params_i)  # Add initial guesses to result output.
        return res

    def plot_fit(self, params, data, t):
        """Plot a timeseries and the generated pdf (with the histogram and genextreme pdf).

        Parameters
        ----------
        params : tuple of floats
            Shape, location and scale parameters (shape, loc0, loc1, scale0, scale1).
        data : xarray.DataArray
            Data timeseries.
        t : int
            Index of the covariate.
        """
        fig, ax = plt.subplots(1, 2, figsize=[12, 6], squeeze=True)

        # Subplot 1: Data timeseries.
        data.plot(ax=ax[0])
        ax[0].set_title('a) Data timeseries', loc='left')

        # Subplot 2: histogram and PDFs.
        # Histogram.
        _, x, _ = data.plot.hist(bins=40, density=True, alpha=0.5, label='Histogram', ax=ax[1])
        ax[1].set_title('b) Data histogram and PDF', loc='left')

        if len(params) == 5:
            # Non-stationary GEV parameters.
            shape, loc0, loc1, scale0, scale1 = params
        else:
            # Stationary GEV parameters.
            shape, loc0, scale0 = params
            scale1, loc1 = 0, 0

        loc = loc0 + loc1 * t
        scale = scale0 + scale1 * t

        # Distribution parameters from genextremes using default args
        shape_genext, loc_genext, scale_genext = genex.fit(data)

        # Plot genextreme GEV fit
        pdf_genext = genex.pdf(x, shape_genext, loc=loc_genext, scale=scale_genext)
        ax[1].plot(x, pdf_genext, c='r', ls=':', lw=3, label='GEV fit using genextreme')

        # Plot NonStationaryGEV fit
        pdf_nsgev = genex.pdf(x, -shape, loc=loc, scale=scale)
        ax[1].plot(x, pdf_nsgev, c='k', lw=3, label='GEV fit using NonStationaryGEV', zorder=0)

        ax[1].set_ylabel('Probability')
        ax[1].legend()
        plt.tight_layout()
        plt.show()
        return
