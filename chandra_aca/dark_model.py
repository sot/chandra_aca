# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Routines related to the canonical Chandra ACA dark current model.

The model is based on smoothed twice-broken power-law fits of
dark current histograms from Jan-2007 though Aug-2017.  This analysis
was done entirely with dark current maps scaled to -14 C.

See: /proj/sot/ska/analysis/dark_current_model/dark_model.ipynb
and other files in that directory.

Alternatively:
http://nbviewer.ipython.org/url/asc.harvard.edu/mta/ASPECT/analysis/dark_current_model/dark_model.ipynb
"""

import numpy as np
import warnings

from Chandra.Time import DateTime

# Define a common fixed binning of dark current distribution
from . import darkbins

# Global cache (e.g. for initial dark current in synthetic_dark_image
CACHE = {}

# Some constants and globals.  Done this way to support sherpa fitting.
# Needs to be re-worked to be nicer.

# Fixed gaussian for smoothing the broken power law
dx = 0.1
sigma = 0.30                            # Gaussian sigma in log space
xg = np.arange(-2.5 * sigma, 2.5 * sigma, dx, dtype=float)
yg = np.exp(-0.5 * (xg / sigma) ** 2)
yg /= np.sum(yg)

NPIX = 1024 ** 2

# Fixed
xbins = darkbins.bins
xall = darkbins.bin_centers
imin = 0
imax = len(xall)

# Warm threshold used in fitting acq prob model.  This constant is
# not used in any configured code, but leave here just in case.
warm_threshold = 100.

# Increase in dark current per 4 degC increase in T_ccd
DARK_SCALE_4C = 1.0 / 0.70


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


def get_dark_hist(date, t_ccd):
    """
    Return the dark current histogram corresponding to ``date`` and ``t_ccd``.

    :param date: date in any DateTime format
    :param t_ccd: CCD temperature (deg C)

    :returns: bin_centers, bins, darkhist
    """
    pars = get_sbp_pars(date)
    x = darkbins.bin_centers
    y = smooth_twice_broken_pow(pars, x)

    # Model params are calibrated using reference temp. -14 C
    scale = dark_temp_scale(-14, t_ccd)
    xbins = darkbins.bins * scale
    x = x * scale
    return x, xbins, y


def smooth_broken_pow(pars, x):
    """
    Smoothed broken power-law.  Pars are same as bpl1d (NOT + gaussian sigma):
    1: gamma1
    2: gamma2
    3: x_b (break point)
    4: x_r (normalization reference point)
    5: ampl1
    """
    (gamma1, gamma2, x_b, x_r, ampl1) = pars
    ampl2 = ampl1 * (x_b / x_r) ** (gamma2 - gamma1)
    ok = xall > x_b
    y = ampl1 * (xall / x_r) ** (-gamma1)
    y[ok] = ampl2 * (xall[ok] / x_r) ** (-gamma2)
    imin = np.searchsorted(xall, x[0] - 1e-3)
    imax = np.searchsorted(xall, x[-1] + 1e-3)
    return np.convolve(y, yg, mode='same')[imin:imax]


def smooth_twice_broken_pow(pars, x):
    """
    Smoothed broken power-law.  Pars are same as bpl1d (NOT + gaussian sigma):
    1: gamma1
    2: gamma2
    3: gamma3
    4: x_b (break point)
    5: ampl1
    """
    x_b2 = 1000
    x_r = 50
    (gamma1, gamma2, gamma3, x_b, ampl1) = pars

    y = ampl1 * (xall / x_r) ** (-gamma1)

    i0, i1 = np.searchsorted(xall, [x_b, x_b2])
    ampl2 = ampl1 * (x_b / x_r) ** (gamma2 - gamma1)
    y[i0:i1] = ampl2 * (xall[i0:i1] / x_r) ** (-gamma2)

    i1 = np.searchsorted(xall, x_b2)
    ampl3 = ampl2 * (x_b2 / x_r) ** (gamma3 - gamma2)
    y[i1:] = ampl3 * (xall[i1:] / x_r) ** (-gamma3)

    imin = np.searchsorted(xall, x[0] - 1e-3)
    imax = np.searchsorted(xall, x[-1] + 1e-3)
    return np.convolve(y, yg, mode='same')[imin:imax]


def temp_scalefac(T_ccd):
    """
    Return the multiplicative scale factor to convert a CCD dark map from
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
    Return smooth broken powerlaw parameters set(s) at ``dates``.

    This is based on the sbp fits for the darkhist_zodi_m14 histograms in
    /proj/sot/ska/analysis/dark_current_model/dark_model.ipynb.

    The actual bi-linear fits (as a function of year) to the g1, g2, g3, x_b, and ampl
    parameters are derived from fits and by-hand inspection of fit trending.

    This is only accurate for dates > 2007.0.

    :param dates: one or a list of date(s) in DateTime compatible format
    :returns: one or a list of parameter lists [g1, g2, g3, x_b, ampl]
    """
    dates, is_scalar = as_array(dates)

    mid_year = 2012.0  # Fixed in dark_model.ipynb notebook
    years = DateTime(dates).frac_year
    dyears = years - mid_year

    # Poly fit parameter for pre-2012 and post-2012.  Vals here are:
    # y_mid, slope_pre, slope_post
    par_fits = ((0.075, -0.00692, -0.0207),  # g1
                (3.32, 0.0203, 0 * 0.0047),  # g2
                (2.40, 0.061, 0.061),  # g3
                (192, 0.1, 0.1),  # x_b
                (18400, 1.45e3, 742),  # ampl
                )

    pars_list = []
    for dyear in dyears:
        pars = []
        for y_mid, slope_pre, slope_post in par_fits:
            slope = slope_pre if dyear < 0 else slope_post
            pars.append(y_mid + slope * dyear)
        pars_list.append(pars)

    if is_scalar:
        pars_list = pars_list[0]

    return pars_list


def get_warm_fracs(warm_threshold, date='2013:001:12:00:00', T_ccd=-19.0):
    """
    Calculate fraction of pixels in modeled dark current distribution
    above warm threshold(s).

    :param warm_threshold: scalar or list of threshold(s) in e-/sec
    :param date: date to use for modeled dark current distribution/histogram
    :param T_ccd: temperature (C) of modeled dark current distribution
    :returns: list or scalar of warm fractions (depends on warm_threshold type)
    """

    x, xbins, y = get_dark_hist(date, T_ccd)
    warm_thresholds, is_scalar = as_array(warm_threshold)

    warmpixes = []
    for warm_threshold in warm_thresholds:
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
        warmpix += partial_bin
        warmpixes.append(warmpix)

    if is_scalar:
        out = warmpixes[0]
    else:
        out = np.array(warmpixes)

    return out / (1024.0 ** 2)


def synthetic_dark_image(date, t_ccd_ref=None):
    """
    Generate a synthetic dark current image corresponding to the specified
    ``date`` and ``t_ccd``.

    :param date: (DateTime compatible)
    :param t_ccd_ref: ACA CCD temperature
    """

    from mica.archive.aca_dark import get_dark_cal_image

    if 'dark_1999223' not in CACHE:
        dark = get_dark_cal_image('1999:223:12:00:00', select='nearest', t_ccd_ref=-14).ravel()
        CACHE['dark_1999223'] = dark.copy()
    else:
        dark = CACHE['dark_1999223'].copy()

    # Fill any pixels above 40 e-/sec with a random sampling from a cool
    # pixel below 40 e-/sec
    warm = dark > 40
    warm_idx = np.flatnonzero(warm)
    not_warm_idx = np.flatnonzero(~warm)
    fill_idx = np.random.randint(0, len(not_warm_idx), len(warm_idx))
    dark[warm_idx] = dark[fill_idx]

    darkmodel = smooth_twice_broken_pow(get_sbp_pars(date), xall)

    darkran = np.random.poisson(darkmodel)

    nn = 0
    for ii, npix in enumerate(darkran):
        # Generate n log-uniform variates within bin
        if npix > 0:
            logdark = np.random.uniform(np.log(xbins[ii]), np.log(xbins[ii + 1]), npix)
            dark[nn:nn + npix] += np.exp(logdark)
            nn += npix

    np.random.shuffle(dark)
    dark.shape = (1024, 1024)

    if t_ccd_ref is not None:
        dark *= dark_temp_scale(-14, t_ccd_ref)

    return dark
