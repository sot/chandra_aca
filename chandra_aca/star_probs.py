# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions related to probabilities for star acquisition and guide tracking.
"""

from __future__ import print_function, division

from numba import jit
from six.moves import zip

from scipy.optimize import brentq
import numpy as np
from Chandra.Time import DateTime

# Local import in acq_success_prob():
# from .dark_model import get_warm_fracs

#
# NOTE: the "WITH_MS" are historical and no longer used in flight
#       and do not reflect ACA behavior beyond 2016-Feb-08.
#
# Scale and offset fit of polynomial to acq failures in log space.
# Derived in the fit_sota_model_probit.ipynb IPython notebook for data
# covering 2007-Jan-01 - 2015-July-01.  This is in the aca_stats repo.
# (Previously in state_of_aca but moved 2016-Feb-2).
#
# scale = scl2 * m10**2 + scl1 * m10 + scl0, where m10 = mag - 10,
# and likewise for offset.

WARM_THRESHOLD = 100  # Value (N100) used for fitting

SOTA_FIT_NO_1P5_WITH_MS = [9.6887121605441173,  # scl0
                           9.1613040261776177,  # scl1
                           -0.41919343599067715,  # scl2
                           -2.3829996965532048,  # off0
                           0.54998934814773903,  # off1
                           0.47839260691599156]  # off2

SOTA_FIT_ONLY_1P5_WITH_MS = [8.541709287866361,
                             0.44482688155644085,
                             -3.5137852251178465,
                             -1.3505424393223699,
                             1.5278061271148755,
                             0.30973569068842272]

#
# FLIGHT model coefficients.
#
# Multiple stars flag disabled (starting operationally with FEB0816).  Fit
# with fit_flight_acq_prob_model.ipynb in the aca_stats repo.

# Coefficents for dependence of probability on search box size (halfwidth).  From:
# https://github.com/sot/skanb/blob/master/pea-test-set/fit_box_size_acq_prob.ipynb

HALFWIDTH_B1 = 0.96
HALFWIDTH_B2 = -0.30

# Min and max star acquisition probabilities, regardless of model predictions
MIN_ACQ_PROB = 1e-6
MAX_ACQ_PROB = 0.985

SOTA_FIT_NO_1P5_NO_MS = [4.38145,  # scl0
                         6.22480,  # scl1
                         2.20862,  # scl2
                         -2.24494,  # off0
                         0.32180,  # off1
                         0.08306,  # off2
                         ]

SOTA_FIT_ONLY_1P5_NO_MS = [4.73283,  # scl0
                           7.63540,  # scl1
                           4.56612,  # scl2
                           -1.49046,  # off0
                           0.53391,  # off1
                           -0.37074,  # off2
                           ]

# Default global values using NO_MS settings.  Kinda ugly.
def set_acq_model_ms_filter(ms_enabled=False):
    """
    Choose one of two sets of acquisition model fit parameters based
    on ``ms_enabled``.  The default is ``False``:

    - True: MS filtering enabled (prior to FEB0816 loads), where stars would
      be rejected if MS flag was set
    - False: MS filtering disabled (including and after FEB0816 loads)

    The selected fit parameters are global/module-wide.

    """
    global SOTA_FIT_NO_1P5
    global SOTA_FIT_ONLY_1P5
    if ms_enabled:
        SOTA_FIT_NO_1P5 = SOTA_FIT_NO_1P5_WITH_MS
        SOTA_FIT_ONLY_1P5 = SOTA_FIT_ONLY_1P5_WITH_MS
    else:
        SOTA_FIT_NO_1P5 = SOTA_FIT_NO_1P5_NO_MS
        SOTA_FIT_ONLY_1P5 = SOTA_FIT_ONLY_1P5_NO_MS

# Use the *_NO_MS parameters by default.
set_acq_model_ms_filter(ms_enabled=False)


def t_ccd_warm_limit(mags, date=None, colors=0, min_n_acq=5.0,
                     cold_t_ccd=-21, warm_t_ccd=-5):
    """
    Find the warmest CCD temperature which meets the ``min_n_acq`` acquisition stars
    criterion.  This returns a value between ``cold_t_ccd`` and ``warm_t_ccd``.  At the
    cold end the result may be below ``min_n_acq``, in which case the star catalog
    may be rejected.

    The ``min_n_acq`` argument can be specified in one of two ways:

     - Scalar float value: expected number of acquired stars must exceed threshold.
         Expected number is the sum of the individual probabilities.
     - Tuple (n, prob): computed probability of acquiring ``n`` or fewer stars
         must not exceed ``prob``.

    :param mags: list of star ACA mags
    :param date: observation date (any Chandra.Time valid format)
    :param colors: list of star B-V colors (optional, default=0.0)
    :param min_n_acq: float or tuple (see above)
    :param cold_t_ccd: coldest CCD temperature to consider (default=-21 C)
    :param warm_t_ccd: warmest CCD temperature to consider (default=-5 C)

    :returns: (t_ccd, n_acq | prob_n_or_fewer) tuple with CCD temperature upper limit and:
              - number of expected ACQ stars at that temperature (scalar min_n_acq)
              - probability of acquiring ``n`` or fewer stars (tuple min_n_acq)
    """
    if isinstance(min_n_acq, tuple):
        n_or_fewer, prob_n_or_fewer = min_n_acq

    def n_acq_above_min(t_ccd):
        """
        This will be positive if the expected number of stars is above the
        minimum number of stars.  Positive => more expected stars.
        """
        probs = acq_success_prob(date=date, t_ccd=t_ccd, mag=mags, color=colors)
        return np.sum(probs) - min_n_acq

    def prob_n_or_fewer_below_max(t_ccd):
        """
        This will be positive if the computed probability of acquiring n_or_fewer
        stars is less than the threshold.  Positive => lower prob. of safing action.
        """
        probs = acq_success_prob(date=date, t_ccd=t_ccd, mag=mags, color=colors)
        n_acq_probs, n_or_fewer_probs = prob_n_acq(probs)
        return prob_n_or_fewer - n_or_fewer_probs[n_or_fewer]

    merit_func = (prob_n_or_fewer_below_max if isinstance(min_n_acq, tuple)
                  else n_acq_above_min)

    if merit_func(warm_t_ccd) >= 0:
        # If there are enough ACQ stars at the warmest reasonable CCD temperature
        # then use that temperature.
        t_ccd = warm_t_ccd

    elif merit_func(cold_t_ccd) <= 0:
        # If there are not enough ACQ stars at the coldest CCD temperature then stop there
        # as well.  The ACA thermal model will never predict a temperature below this
        # value so this catalog will fail thermal check.
        t_ccd = cold_t_ccd

    else:
        # At this point there must be a zero in the range [cold_t_ccd, warm_t_ccd]
        t_ccd = brentq(merit_func, cold_t_ccd, warm_t_ccd, xtol=1e-4, rtol=1e-4)

    out = merit_func(t_ccd)
    if isinstance(min_n_acq, tuple):
        out = prob_n_or_fewer - out
    else:
        out = out + min_n_acq

    return t_ccd, out


@jit(nopython=True)
def _prob_n_acq(star_probs, n_stars, n_acq_probs):
    """
    Jit-able portion of prob_n_acq
    """
    for cfg in range(1 << n_stars):
        prob = 1.0
        n_acq = 0
        for slot in range(n_stars):
            success = cfg & (1 << slot)
            if success > 0:
                prob = prob * star_probs[slot]
                n_acq += 1
            else:
                prob = prob * (1 - star_probs[slot])
        n_acq_probs[n_acq] += prob


def prob_n_acq(star_probs):
    """
    Given an input array of star acquisition probabilities ``star_probs``,
    return the probabilities of acquiring exactly n_acq stars, where n_acq
    is evaluated at values 0 to n_stars.  This is returned as an array
    of length n_stars.  In addition the cumulative sum, which represents
    the probability of acquiring n_acq or fewer stars, is returned.

    :param star_probs: array of star acq probabilities (list or ndarray)

    :returns n_acq_probs, cum_n_acq_probs: tuple of ndarray, ndarray
    """
    star_probs = np.array(star_probs, dtype=np.float64)
    n_stars = len(star_probs)
    n_acq_probs = np.zeros(n_stars + 1, dtype=np.float64)

    _prob_n_acq(star_probs, n_stars, n_acq_probs)

    return n_acq_probs, np.cumsum(n_acq_probs)


def acq_success_prob(date=None, t_ccd=-19.0, mag=10.0, color=0.6, spoiler=False, halfwidth=120, bgd=0):
    """
    Return probability of acquisition success for given date, temperature and mag.

    Any of the inputs can be scalars or arrays, with the output being the result of
    the broadcasted dimension of the inputs.

    This is based on the dark model and acquisition success model presented
    in the State of the ACA 2013, and subsequently updated to use a Probit
    transform and separately fit B-V=1.5 stars.  This is available in the
    state_of_aca repo as fit_sota_model_probit.ipynb.

    :param date: Date(s) (scalar or np.ndarray, default=NOW)
    :param t_ccd: CD temperature(s) (degC, scalar or np.ndarray, default=-19C)
    :param mag: Star magnitude(s) (scalar or np.ndarray, default=10.0)
    :param color: Star color(s) (scalar or np.ndarray, default=0.6)
    :param spoil: Star spoiled (boolean or np.ndarray, default=False)
    :param halfwidth: Search box halfwidth (arcsec, default=120)
    :param bgd: Expected background level (e.g. zodiacal light) in e-/sec (default=0)

    :returns: Acquisition success probability(s)
    """
    from .dark_model import get_warm_fracs

    date = DateTime(date).secs

    is_scalar, dates, t_ccds, mags, colors, spoilers, halfwidths = broadcast_arrays(
        date, t_ccd, mag, color, spoiler, halfwidth)

    spoilers = spoilers.astype(bool)

    warm_fracs = []
    for date, t_ccd in zip(dates.ravel(), t_ccds.ravel()):
        warm_frac = get_warm_fracs(WARM_THRESHOLD - bgd, date=date, T_ccd=t_ccd)
        warm_fracs.append(warm_frac)
    warm_frac = np.array(warm_fracs).reshape(dates.shape)

    probs = model_acq_success_prob(mags, warm_fracs, colors, halfwidths)

    p_0p7color = .4294  # probability multiplier for a B-V = 0.700 star (REF?)
    p_spoiler = .9241  # probability multiplier for a search-spoiled star (REF?)

    # If the star is brighter than 8.5 or has a calculated probability
    # higher than the max_star_prob, clip it at that value
    probs[mags < 8.5] = MAX_ACQ_PROB
    probs[colors == 0.7] *= p_0p7color
    probs[spoilers] *= p_spoiler

    probs = probs.clip(MIN_ACQ_PROB, MAX_ACQ_PROB)

    # Return probabilities.  The [()] getitem at the end will flatten a
    # scalar array down to a pure scalar.
    return probs[0] if is_scalar else probs


def model_acq_success_prob(mag, warm_frac, color=0, halfwidth=120):
    """
    Calculate raw model probability of acquisition success for a star with ``mag``
    magnitude and a CCD warm fraction ``warm_frac``.  This is not typically used directly
    since it does not account for star properties like spoiled or color=0.7.

    Uses the empirical relation::

       P_fail_probit = offset(mag) + scale(mag) * warm_frac + box_delta(halfwidth)
       P_acq_fail = Normal_CDF()
       P_acq_success = 1 - P_acq_fail

    This is based on the dark model and acquisition success model presented in the State
    of the ACA 2013, and subsequently updated to use a Probit transform and separately fit
    B-V=1.5 stars.  It was updated in 2017 to include a fitted dependence on the search
    box halfwidth.  See:

    https://github.com/sot/skanb/blob/master/pea-test-set/fit_box_size_acq_prob.ipynb
    https://github.com/sot/aca_stats/blob/master/fit_flight_acq_prob_model.ipynb

    :param mag: ACA magnitude (float or np.ndarray)
    :param warm_frac: N100 warm fraction (float or np.ndarray)
    :param color: B-V color to check for B-V=1.5 => red star (float or np.ndarray)
    :param halfwidth: search box halfwidth (arcsec, default=120)
    """
    from scipy.stats import norm

    is_scalar, mag, warm_frac, color = broadcast_arrays(mag, warm_frac, color)

    m10 = mag - 10.0
    box120 = (halfwidth - 120) / 120  # Normalized halfwidth, 0.0 for halfwidth=120

    p_fail = np.zeros_like(mag)
    for mask, fit_pars in ((color == 1.5, SOTA_FIT_ONLY_1P5),
                           (color != 1.5, SOTA_FIT_NO_1P5)):
        if np.any(mask):
            scale = np.polyval(fit_pars[0:3][::-1], m10)
            offset = np.polyval(fit_pars[3:6][::-1], m10)
            box_delta = HALFWIDTH_B1 * box120 + HALFWIDTH_B2 * box120 ** 2

            p_fail[mask] = (offset + scale * warm_frac + box_delta)[mask]

    p_fail = norm.cdf(p_fail)  # probit transform
    p_fail[mag < 8.5] = 0.015  # actual best fit is ~0.006, but put in some conservatism
    p_success = (1 - p_fail)

    # Clip values to reasonable range regardless of model prediction
    p_success = p_success.clip(MIN_ACQ_PROB, MAX_ACQ_PROB)

    return p_success[0] if is_scalar else p_success  # Return scalar if ndim=0


def broadcast_arrays(*args):
    is_scalar = all(np.array(arg).ndim == 0 for arg in args)
    args = np.atleast_1d(*args)
    outs = [is_scalar] + np.broadcast_arrays(*args)
    return outs


def mag_for_p_acq(p_acq, date=None, t_ccd=-19.0, halfwidth=120):
    """
    For a given ``date`` and ``t_ccd``, find the star magnitude that has an
    acquisition probability of ``p_acq``.  Star magnitude is defined/limited
    to the range 5.0 - 12.0 mag.

    :param p_acq: acquisition probability (0 to 1.0)
    :param date: observation date (any Chandra.Time valid format)
    :param t_ccd: ACA CCD temperature (deg C)
    :param halfwidth: search box halfwidth (arcsec, default=120)
    :returns mag: star magnitude
    """

    def prob_minus_p_acq(mag):
        """Function that gets zeroed in brentq call later"""
        prob = acq_success_prob(date=date, t_ccd=t_ccd, mag=mag, halfwidth=halfwidth)
        return prob - p_acq

    # prob_minus_p_acq is monotonically decreasing from the (minimum)
    # bright mag to the (maximum) faint_mag.
    bright_mag = 5.0
    faint_mag = 12.0
    if prob_minus_p_acq(bright_mag) <= 0:
        # Below zero already at bright mag limit so return the bright limit.
        mag = bright_mag

    elif prob_minus_p_acq(faint_mag) >= 0:
        # Above zero still at the faint mag limit so return the faint limit.
        mag = faint_mag

    else:
        # At this point there must be a zero in the range [bright_mag, faint_mag]
        mag = brentq(prob_minus_p_acq, bright_mag, faint_mag, xtol=1e-4, rtol=1e-4)
        mag = float(mag)

    return mag
