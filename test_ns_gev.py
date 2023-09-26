# -*- coding: utf-8 -*-
"""Draft tests of non-stationary GEV functions for https://github.com/AusClimateService/unseen


Notes
-----
* Check:
    RuntimeWarning: invalid value encountered in log f = np.log(scale) + (1 + 1 / shape) * np.ma.log(Z) + np.ma.power(Z, (-1 / shape))
    RuntimeWarning: divide by zero encountered in divide return func(*(_execute_task(a, cache) for a in args))

@author: Annette Stellema
@email: stellema@github.com
@created: Fri Sep 22 13:35:14 2023
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
import scipy
from scipy.stats import genextreme as genex
import xarray as xr

from ns_gev import NonStationaryGEV


# @todo
logger = logging.getLogger('test_ns_gev')
if len(logger.handlers) == 0:
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(Path.cwd() / 'test_ns_gev.log')
    f_handler.setLevel(logging.DEBUG)
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(logging.Formatter('{message}', style='{'))
    f_handler.setFormatter(logging.Formatter('{asctime}: {message}', '%Y-%m-%d %H:%M:%S', style='{'))
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.setLevel(logging.DEBUG)

def plot_timeseries_hist(x):
    """Plot dataarray time series and histogram."""
    fig, ax = plt.subplots(1, 2, figsize=[12, 5], squeeze=True)
    x.plot(ax=ax[0])
    _, _, _ = x.plot.hist(bins='auto', density=True, alpha=0.5, label='Histogram', ax=ax[1])


def test_genextremes_fit_initial_guesses(data, shape_nsgev, loc0_nsgev, scale0_nsgev):
    """Print genextremes fit using different initial guesses."""
    # Test genextremes data fit using the same initial guess as NS-GEV.
    params_i = NonStationaryGEV()._fitstart_lmm(data.values, np.arange(data.size))
    params = genex.fit(data, loc=params_i[1], scale=params_i[3])
    logger.info(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], params)])

    # Test genextremes data fit using results of nsgev as initial guess.
    params = genex.fit(data, shape_nsgev, loc=loc0_nsgev, scale=scale0_nsgev)
    logger.info(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], params)])

    # Test genextremes data fit using mean and std as initial guess.
    params = genex.fit(data, loc=np.mean(data), scale=np.std(data))
    logger.info(*['{}={:.03f}'.format(k, v) for k, v in zip(['c', 'μ0', 'σ0'], params)])


def get_test_arrays(n, rvs=scipy.stats.genextreme.rvs, **kwargs):
    """Create untrended and trended timeseries or testing.

    Parameters
    ----------
    n : int
        Number of timesteps.
    rvs : method, optional
        Distribution type. The default is scipy.stats.genextreme.rvs.
    **kwargs : dict
        Optional shape, loc, scale and trend/gradient parameters.

    Returns
    -------
    times, da, da_trend : array_like
        Test arrays.

    Notes
    -----
    * Need to add non-linear trends.
    """
    trend = kwargs.pop('trend', 1e-3)

    # Time indexes (can change dataset time to datetime format, but leave this as is).
    times = np.arange(n)

    # Noisy line (stable)
    da = xr.DataArray(rvs, coords={'time': times})
    da = da.chunk()

    # Noisy line (increasing)
    da_trend = xr.DataArray(da.values + trend, coords={'time': times})
    da_trend = da_trend.chunk()

    # # Oscillating line (increasing).
    # prd, mx = 10, 5
    # da_osc = xr.DataArray(np.sin(np.pi * times / 10) * mx / 2 + mx / 2 * times / 100,
    #                       coords={'time': times})
    return times, da, da_trend


def test_plot_ns_gev_params(array, trend):
    """Calculate parameters from NS-GEV using trended and non-trended data."""
    times, da, da_trend = get_test_arrays(n=array.size, rvs=array, trend=trend)
    nsgev = NonStationaryGEV()

    # Calculate parameters.
    for data in [da, da_trend]:
        # Calculate parameters.
        res = nsgev.fit(data, times, stationary=0, start_method='genextremes')
        logger.info('Initial: {}'.format(', '.join(['{}={:.05f}'.format(k, v) for k, v in zip(res.x_name, res.x0)])))
        logger.info('Optimal: {} (n={})'.format(', '.join(['{}={:.05f}'.format(k, v) for k, v in zip(res.x_name, res.x)]), array.size))
        shape_genext, loc_genext, scale_genext = genex.fit(data)
        logger.info('Genext fit: c={: .4f}, μ={:.4f}, σ={:.4f}'.format(shape_genext, loc_genext, scale_genext))

        # Plot timeseries & PDFs.
        # Expect bad fit for trended, but same shape,loc0,scale0 as untrended.
        nsgev.plot_fit(res.x, data, t=0)

    # Plot the distribution using a time subset (to see NS-GEV fit).
    data_slices = [[k, k+100] for k in np.arange(0, data.size - 99, 100)]
    for j in [data_slices[0], data_slices[-1]]:
        nsgev.plot_fit(res.x, data.isel(time=slice(*j)), t=np.mean(j))
    return


if __name__ == '__main__':
    n = 20000  # Number of data points/time steps
    times = np.arange(n)
    test_arrays = []
    trend_list = []

    # GEV distribution with positive linear trend.
    shape, loc, scale, trend = 0.8, 3.2, 0.5, 5e-4
    test_arrays.append(scipy.stats.genextreme.rvs(0.8, loc=3.2, scale=0.5, size=n, random_state=0))
    trend_list.append(times * 5e-4)

    # GEV distribution with negative exp trend.
    test_arrays.append(scipy.stats.genextreme.rvs(0.8, loc=-3.3, scale=0.5, size=n, random_state=0))
    trend_list.append((times * -5e-4)**2)

    # TGEV distribution with weak positive linear trend.
    test_arrays.append(scipy.stats.genextreme.rvs(0.1, loc=0.1, scale=0.1, size=n, random_state=0))
    trend_list.append(times * 5e-6)

    test_arrays.append(scipy.stats.genextreme.rvs(0.2, loc=-8, scale=0.9, size=n, random_state=0))
    trend_list.append(times * -5e-2)

    # Gumbel_r distribution with positive linear trend).
    test_arrays.append(scipy.stats.gumbel_r.rvs(loc=120, scale=3, size=n, random_state=0))
    trend_list.append(times * 5e-4)

    # Gumbel_l distribution with small positive linear trend.
    test_arrays.append(scipy.stats.gumbel_l.rvs(loc=-3.2, scale=0.5, size=n, random_state=0))
    trend_list.append(times * 5e-3)

    for array, trend in zip(test_arrays, trend_list):
        test_plot_ns_gev_params(array, trend)
