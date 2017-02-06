"""
Routines related to the dark current model and guide / acq success prediction.
"""

from six.moves import zip
import numpy as np
from numpy import exp, log, arange
import warnings

import Ska.Numpy
from Chandra.Time import DateTime

# Define a common fixed binning of dark current distribution
from . import darkbins

# Some constants and globals.  Done this way to support sherpa fitting.
# Needs to be re-worked to be nicer.

# Fixed gaussian for smoothing the broken power law
dx = 0.1
sigma = 0.30                            # Gaussian sigma in log space
xg = arange(-2.5 * sigma, 2.5 * sigma, dx, dtype=float)
yg = exp(-0.5 * (xg / sigma) ** 2)
yg /= np.sum(yg)

NPIX = 1024 ** 2

# Fixed
xbins = darkbins.bins
xall = darkbins.bin_centers
imin = 0
imax = len(xall)

# scale and offset fit of polynomial to acq failures in log space
acq_fit = {
    'scale': (-0.491, 0.990, 0.185),
    'offset': (0.280, 0.999, -1.489),
    }

warm_threshold = 100.

DARK_SCALE_4C = 1.0 / 0.70  # Increase in dark current per 4 degC increase in T_ccd

def dark_temp_scale(t_ccd, t_ccd_ref=-19.0, scale_4c=None):
    """Return the multiplicative scale factor to convert a CCD dark map
    or dark current value from temperature ``t_ccd`` to temperature
    ``t_ccd_ref``::

      scale = scale_4c ** ((t_ccd_ref - t_ccd) / 4.0)

    In other words, if you have a dark current value that corresponds to ``t_ccd``
    and need the value at a different temperature ``t_ccd_ref`` then use the
    the following.  Do not be misled by the misleading parameter names.

      >>> from chandra_aca.dark_scale import dark_temp_scale
      >>> scale = dark_temp_scale(t_ccd, t_ccd_ref, scale_4c)
      >>> dark_curr_at_t_ccd_ref = scale * dark_curr_at_t_ccd

    The default value for ``scale_4c`` is 1.0 / 0.7.  It is written this way
    because the equation was previously expressed using 1 / scale_4c with a
    value of 0.7. This value is based on best global fit for dark current model
    in `plot_predicted_warmpix.py`.  This represents the multiplicative change
    in dark current for each 4 degC increase::

      >>> dark_temp_scale(t_ccd=-18, t_ccd_ref=-10, scale_4c=2.0)
      4.0

    :param t_ccd: actual temperature (degC)
    :param t_ccd_ref: reference temperature (degC, default=-19.0)
    :param scale_4c: increase in dark current per 4 degC increase (default=1.0 / 0.7)

    :returns: scale factor
    """
    if scale_4c is None:
        scale_4c = DARK_SCALE_4C

    return scale_4c ** ((t_ccd_ref - t_ccd) / 4.0)


def get_dark_model(date, t_ccd):
    """
    Return the dark current model corresponding to ``date`` and ``t_ccd``.

    :param date: date in any DateTime format
    :param t_ccd: CCD temperature (deg C)

    :returns: TBD
    """
    raise NotImplementedError()


def get_dark_hist(date, t_ccd):
    """
    Return the dark current histogram corresponding to ``date`` and ``t_ccd``.

    :param date: date in any DateTime format
    :param t_ccd: CCD temperature (deg C)

    :returns: bin_centers, bins, darkhist
    """
    pars = get_sbp_pars(date)
    x = darkbins.bin_centers
    y = smooth_broken_pow(pars, x)

    scale = dark_temp_scale(-19, t_ccd)
    xbins = darkbins.bins * scale
    x = x * scale
    return x, xbins, y


def smooth_broken_pow(pars, x):
    """Smoothed broken power-law.  Pars are same as bpl1d (NOT + gaussian sigma):
    1: gamma1
    2: gamma2
    3: x_b (break point)
    4: x_r (normalization reference point)
    5: ampl1
    #   NOT 6: sigma (bins)"""
    (gamma1, gamma2, x_b, x_r, ampl1) = pars
    ampl2 = ampl1 * (x_b / x_r) ** (gamma2 - gamma1)
    ok = xall > x_b
    y = ampl1 * (xall / x_r) ** (-gamma1)
    y[ok] = ampl2 * (xall[ok] / x_r) ** (-gamma2)
    imin = np.searchsorted(xall, x[0] - 1e-3)
    imax = np.searchsorted(xall, x[-1] + 1e-3)
    return np.convolve(y, yg, mode='same')[imin:imax]


def temp_scalefac(T_ccd):
    """Return the multiplicative scale factor to convert a CCD dark map from
    the nominal -19C temperature to the temperature T.  Based on best global fit for
    dark current model in plot_predicted_warmpix.py.  Previous value was 0.62 instead
    of 0.70.

    If attempting to reproduce previous analysis, be aware that this is now calling
    chandra_aca.dark_model.dark_temp_scale and the value will be determined using the
    module DARK_SCALE_4C value which may differ from previous values of 1.0/0.70 or 1.0/0.62.
    """
    warnings.warn("temp_scalefac is deprecated.  See chandra_aca.dark_model.dark_temp_scale.")
    return dark_temp_scale(-19, T_ccd)


def as_array(vals):
    if np.array(vals).ndim == 0:
        is_scalar = True
        vals = np.array([vals])
    else:
        is_scalar = False

    vals = np.array(vals)
    return vals, is_scalar


def get_sbp_pars(dates):
    """
    Return smooth broken powerlaw parameters at ``date``.  This is based on the
    sbp fits for the darkhist_peaknorm histograms, with parameters derived from
    by-hand inspection of fit trending.  See NOTES.
    """
    dates, is_scalar = as_array(dates)
    n_dates = len(dates)

    years = DateTime(dates).frac_year

    ones = np.ones(n_dates)
    g1 = 0.05 * ones
    g2 = 3.15 * ones
    x_r = 50.0 * ones

    ampl = (years - 2000.0) * 1390.2 + 1666.4

    bp_years = np.array([1999.0, 2000.9, 2003.5, 2007.0, 2007.01, 2011.5, 2011.51, 2013.7])
    bp_vals = np.array([125.000, 125.00, 110.00, 117.80, 111.50, 125.00, 108.80, 115.00])
    bp = Ska.Numpy.interpolate(bp_vals, bp_years, years, method='linear')

    if is_scalar:
        g1 = g1[0]
        g2 = g2[0]
        ampl = ampl[0]
        bp = bp[0]
        x_r = x_r[0]

    return g1, g2, bp, x_r, ampl


def get_warm_fracs(warm_threshold, date='2013:001', T_ccd=-19.0):
    x, xbins, y = get_dark_hist(date, T_ccd)
    # First get the full bins to right of warm_threshold
    ii = np.searchsorted(xbins, warm_threshold)
    warmpix = np.sum(y[ii:])
    lx = np.log(warm_threshold)
    lx0 = np.log(xbins[ii - 1])
    lx1 = np.log(xbins[ii])
    ly0 = np.log(y[ii - 1])
    ly1 = np.log(y[ii])
    m = (ly1 - ly0) / (lx1 - lx0)
    partial_bin = y[ii] * (lx1 ** m - lx ** m) / (lx1 ** m - lx0 ** m)
    # print ii, x[ii], xbins[ii - 1], xbins[ii], y[ii], partial_bin
    warmpix += partial_bin

    return warmpix / (1024.0 ** 2)
