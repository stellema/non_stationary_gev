# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:29:51 2023

@author: stellema


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

"""

import dask
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import genextreme as gev
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

    References
    ----------
    * https://xiaoganghe.github.io/python-climate-visuals/chapters/data-analytics/scipy-basic.html
    """
    f = lambda x, t: 2 * (1 - 3**(-x)) / (1 - 2**(-x)) - 3 - t
    shape = scipy.optimize.fsolve(f, 0.1, t3)[0]
    GAM = scipy.special.gamma(1 + shape)
    scale = lm[1] * shape / (GAM * (1 - 2**-shape))
    loc = lm[0] - scale * (1 - GAM) / shape
    return shape, loc, scale


class NonStationaryGEV(scipy.stats.rv_continuous):
    """Wrapper for non-stationary GEV (NS-GEV) distributions."""

    def _fitstart_ns(self, data, times, const=0.1):
        """Initiaze NS-GEV parameters (shape, loc0, loc1, scale0, scale1).

        Parameters
        ----------
        data, times : array_like
            1D arrays of data points and time indexes (added for consistency).
        const : float, optional
            Guess of the shape and trend parameters.

        Returns
        -------
        shape, loc0, loc1, scale0, scale1 : float
            The estimated shape, location and scale parameters.
        """
        return const, np.mean(data), const, np.std(data), const

    def _fitstart_alt(self, data, times):
        """Initialize NS-GEV parameters."""
        loc0, scale0 = self._fit_loc_scale_support(data)
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

    @dask.delayed
    def nllf(self, params, data, t):
        """Calculate the NS-GEV negative log-likelihood function at a time point.

        Parameters
        ----------
        params : tuple of floats
            Shape, location and scale parameters (shape, loc0, loc1, scale0, scale1)
        data : xarray.DataArray
            A 1D array of data points
        t : int
            The time index in which to calculate the negative log-likelihood function.

        Returns
        -------
        f : float

        Notes
        -----
        * ? Use the "mixed L moments: maximum likelihood" (El Adlouni et al. 2007).
        * Are we assuming that the distribution is normal?
        * We use a log-link transformation to ensure the scale parameter is positive.
        * To get log-likelihood function: partially integrate the GEV CDF to get the PDF.
        * The NLLFs follow the equations in Méndez et al. (2007).
        """
        shape, loc0, loc1, scale0, scale1 = params

        # Location
        loc = loc0 + loc1 * t

        # Scale
        scale = np.exp(scale0 + scale1 * t)

        # Subset datarray at time t
        x = data.isel(time=t)

        # Negative Log-likelihood of GEV probability density function.
        s = (x - loc) / scale
        Z = 1 + shape * s

        if Z > 0:
            if shape != 0:
                f = (1 + 1 / shape) * np.log(Z) + Z**(-1 / shape) + np.log(scale)  # GEV NLLF
            elif shape == 0:
                f = np.log(scale) + s + np.exp(-s)  # Gumbel Equation NLLF
        else:
            f = 0  # ???
        return f  # ???

    def sum_nllf(self, params, data, times):
        """Sum NS-GEV negative log-likelihood function over the time series.

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
        """
        f_list = []
        for t in times:  # Check and drop any NaN times?
            # f_list.append(nsgev.fit(params, data, t))
            f_list.append(self.nllf(params, data, t))
        f_sum = np.sum(f_list)
        f_sum = dask.compute(f_sum)
        return f_sum

    @timeit
    def fit(self, data, times):
        """Fit NS-GEV params by minimising the negative log-likelihood function."""
        # Initial values of distribution parameters.
        params_i = self._fitstart_lmm(data.values, times)
        # params_i = self._fitstart_ns(data.values, times, const=0)
        print('Initial guesses:', params_i)  # !!!

        # Minimise the negative log-likelihood function.
        result = scipy.optimize.minimize(self.sum_nllf, params_i, args=(data, times),
                                         method='BF')  # Downhill simplex method
        # result = scipy.optimize.fmin(nllf, params_i, args=(data, times))
        print(', '.join(['{}={:.04f}'.format(k, v) for k, v in zip(['c', 'μ0', 'μ1', 'σ0', 'σ1'],
                                                                   result.x)]))
        return result


def plot_timeseries_hist(x):
    """Plot dataarray time series and histogram."""
    fig, ax = plt.subplots(1, 2, figsize=[12, 5], squeeze=True)
    x.plot(ax=ax[0])
    n, bins, patches = x.plot.hist(bins='fd', density=True, alpha=0.5,
                                   label='Histogram', ax=ax[1])


def get_test_arrays(n, rvs=scipy.stats.genextreme.rvs):
    """Create untrended and trended timeseries or testing.

    Parameters
    ----------
    n : int
        Number of timesteps.
    rvs : method, optional
        Distribution type. The default is scipy.stats.genextreme.rvs.

    Returns
    -------
    times, da_stable, da : array_like
        Test arrays.
    """
    # # ACCESS-S2 rainfall time series at Hobart.
    # var = 'pr'
    # files = glob.glob('/g/data/ux62/access-s2/hindcast/raw_model/atmos/{}/daily/e01/*.nc'.format(var))
    # files.sort()

    # def subset_loc(ds):
    #     return ds.sel(lon=147, lat=-42, method='nearest', drop=True)

    # ds = xr.open_mfdataset(files[0:1], preprocess=subset_loc, chunks='auto',
    #                        compat='override', coords='minimal', parallel=True)

    # Time indexes (can change dataset time to datetime format, but leave this as is).
    times = np.arange(n)

    # Noisy line (stable)
    #da_stable = xr.DataArray(scipy.stats.gumbel_r.rvs(loc=0.5, scale=2, size=times.size), coords={'time': times})
    da_stable = xr.DataArray(rvs(1, loc=1, scale=0.5, size=times.size), coords={'time': times})
    da_stable = da_stable.where(da_stable > 0, 0)
    da_stable = da_stable.chunk()

    # Noisy line (increasing)
    da = xr.DataArray(da_stable.values + times * 1e-3, coords={'time': times})
    da = da.where(da > 0, 0)
    da = da.chunk()

    # # Oscillating line (increasing)
    # prd, mx = 10, 5
    # da_osc = xr.DataArray(np.sin(np.pi * times / 10) * mx / 2 + mx / 2 * times / 100,
    #                       coords={'time': times})
    # da_osc = da_osc.where(da_osc > 0, 0)
    return times, da_stable, da


if __name__ == '__main__':

    times, da_stable, da = get_test_arrays(n=200, rvs=scipy.stats.genextreme.rvs)

    plot_timeseries_hist(da_stable)
    plot_timeseries_hist(da)
    # plot_timeseries_hist(da_osc)

    # Calculate parameters.
    nsgev = NonStationaryGEV()
    res = nsgev.fit(da, times)

    # Calculate parameters.
    nsgev = NonStationaryGEV()
    res_stable = nsgev.fit(da_stable, times)
    shape_nsgev, loc0_nsgev, loc1_nsgev, scale0_nsgev, scale1_nsgev = res_stable.x

    # Test genextremes on da_stable using the same initial guesses (can't specify optimisation method or bounds)
    params_i = NonStationaryGEV()._fitstart_lmm(da_stable.values, times)
    shape_genext, loc0_genext, scale0_genext = gev.fit(da_stable, loc=params_i[1], scale=params_i[3], optimizer=scipy.optimize.fmin)
    print(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], [shape_genext, loc0_genext, scale0_genext])])

    # Test genextremes on da_stable using results of nsgev
    shape_genext, loc0_genext, scale0_genext = gev.fit(da_stable, shape_nsgev, loc=loc0_nsgev, scale=scale0_nsgev)
    print(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], [shape_genext, loc0_genext, scale0_genext])])

    # Test genextremes on da_stable using mean and std
    shape_genext, loc0_genext, scale0_genext = gev.fit(da_stable, loc=np.mean(da_stable), scale=np.std(da_stable))
    print(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], [shape_genext, loc0_genext, scale0_genext])])

    # Test genextremes on da_stable using default args
    shape_genext, loc0_genext, scale0_genext = gev.fit(da_stable)
    print(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], [shape_genext, loc0_genext, scale0_genext])])


    print(shape_nsgev, loc0_nsgev, scale0_nsgev)
    print(shape_genext, loc0_genext, scale0_genext)

    fig, ax = plt.subplots(figsize=[10, 6])
    xvals = np.arange(0, 5)

    # Plot data
    da_stable.plot.hist(bins=40, density=True, alpha=0.5, label='Histogram', ax=ax)

    # Plot genextreme GEV fit
    pdf_genext = gev.pdf(xvals, shape_genext, loc0_genext, scale0_genext)
    ax.plot(xvals, pdf_genext, label='GEV fit using genextreme')

    # Plot NonStationaryGEV fit
    pdf_nsgev = gev.pdf(xvals, -shape_nsgev, loc0_nsgev, scale0_nsgev)
    ax.plot(xvals, pdf_nsgev, label='GEV fit using NonStationaryGEV')

    ax.set_ylabel('Probability')
    ax.legend()

    plt.show()
