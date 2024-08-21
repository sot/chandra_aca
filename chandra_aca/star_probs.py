# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions related to probabilities for star acquisition and guide tracking.

Current default acquisition probability model: ``grid-*`` (latest grid model) This can
be changed by setting the module configuration value ``conf.default_model`` to an
available model or model glob in the ``chandra_models`` repository.

The grid-local-quadratic-2023-05 model definition and fit values based on:

- https://github.com/sot/aca_stats/blob/master/fit_acq_model-2023-05-local-quadratic.ipynb
- https://github.com/sot/aca_stats/blob/master/validate-2023-05-local-quadratic.ipynb
- SS&AWG review 2023-02-01

The grid-floor-2020-02 model definition and fit values based on:

- https://github.com/sot/aca_stats/blob/master/fit_acq_model-2020-02-binned-poly-binom-floor.ipynb
- SSAWG review: 2020-01-29
"""

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.stats
from astropy import config
from astropy.io import fits
from Chandra.Time import DateTime
from cxotime import CxoTimeLike
from numba import jit
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import bisect, brentq
from scipy.stats import beta
from ska_helpers import chandra_models

from chandra_aca.transform import (
    broadcast_arrays,
    broadcast_arrays_flatten,
    snr_mag_for_t_ccd,
)

# Cache of cubic spline functions.  Eval'd only on the first time.
SPLINE_FUNCS = {}

# Value (N100) used for fitting the `sota` model
WARM_THRESHOLD = 100

# Min and max star acquisition probabilities, regardless of model predictions
MIN_ACQ_PROB = 1e-6
MAX_ACQ_PROB = 0.985

MULT_STARS_ENABLED = False


class ConfigItem(config.ConfigItem):
    rootname = "chandra_aca.star_probs"


class Conf(config.ConfigNamespace):
    default_model = ConfigItem(
        defaultvalue="grid-*",
        description=(
            "Default acquisition probability model.  This can be a specific model name "
            "or a glob pattern (e.g. ``'grid-*'`` for latest grid model)."
        ),
    )


# Create a configuration instance for the user
conf = Conf()


def get_box_delta(halfwidth):
    """
    Transform from halfwidth (arcsec) to the box_delta value which gets added
    to failure probability (in probit space).

    Parameters
    ----------
    halfwidth
        scalar or ndarray of box sizes (halfwidth, arcsec)

    Returns
    -------
    box deltas
    """
    # Coefficents for dependence of probability on search box size (halfwidth).  From:
    # https://github.com/sot/skanb/blob/master/pea-test-set/fit_box_size_acq_prob.ipynb
    B1 = 0.96
    B2 = -0.30

    box120 = (
        halfwidth - 120
    ) / 120  # normalized version of box, equal to 0.0 at nominal default
    box_delta = B1 * box120 + B2 * box120**2

    return box_delta


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
    global MULT_STARS_ENABLED
    MULT_STARS_ENABLED = ms_enabled


def t_ccd_warm_limit(
    mags,
    date=None,
    colors=0,
    halfwidths=120,
    min_n_acq=5.0,
    cold_t_ccd=-16,
    warm_t_ccd=-5,
    model=None,
):
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

    Parameters
    ----------
    mags
        list of star ACA mags
    date
        observation date (any Chandra.Time valid format)
    colors
        list of star B-V colors (optional, default=0.0)
    halfwidths
        list of acq box halfwidths(optional, default=120)
    min_n_acq
        float or tuple (see above)
    cold_t_ccd
        coldest CCD temperature to consider (default=-16 C)
    warm_t_ccd
        warmest CCD temperature to consider (default=-5 C)
    model
        probability model (see acq_success_prob() for allowed values, default)

    Returns
    -------
    (t_ccd, n_acq | prob_n_or_fewer) tuple with CCD temperature upper limit:
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
        probs = acq_success_prob(
            date=date,
            t_ccd=t_ccd,
            mag=mags,
            color=colors,
            halfwidth=halfwidths,
            model=model,
        )
        return np.sum(probs) - min_n_acq

    def prob_n_or_fewer_below_max(t_ccd):
        """
        This will be positive if the computed probability of acquiring n_or_fewer
        stars is less than the threshold.  Positive => lower prob. of safing action.
        """
        probs = acq_success_prob(
            date=date,
            t_ccd=t_ccd,
            mag=mags,
            color=colors,
            halfwidth=halfwidths,
            model=model,
        )
        n_acq_probs, n_or_fewer_probs = prob_n_acq(probs)
        return prob_n_or_fewer - n_or_fewer_probs[n_or_fewer]

    merit_func = (
        prob_n_or_fewer_below_max if isinstance(min_n_acq, tuple) else n_acq_above_min
    )

    if merit_func(warm_t_ccd) >= 0:
        # If there are enough ACQ stars at the warmest reasonable CCD temperature
        # then use that temperature.
        t_ccd = warm_t_ccd

    elif merit_func(cold_t_ccd) <= 0:
        # If there are not enough ACQ stars at the coldest CCD temperature then stop
        # there as well.  The ACA thermal model will never predict a temperature below
        # this value so this catalog will fail thermal check.
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

    Parameters
    ----------
    star_probs : array of star acq probabilities (list or ndarray)
        :returns n_acq_probs, cum_n_acq_probs: tuple of ndarray, ndarray
    """
    star_probs = np.array(star_probs, dtype=np.float64)
    n_stars = len(star_probs)
    n_acq_probs = np.zeros(n_stars + 1, dtype=np.float64)

    _prob_n_acq(star_probs, n_stars, n_acq_probs)

    return n_acq_probs, np.cumsum(n_acq_probs)


def acq_success_prob(
    date: CxoTimeLike = None,
    t_ccd: float | np.ndarray[float] = -10.0,
    mag: float | np.ndarray[float] = 10.0,
    color: float | np.ndarray[float] = 0.6,
    spoiler: bool | np.ndarray[bool] = False,
    halfwidth: int | np.ndarray[int] = 120,
    model: Optional[str] = None,
) -> float | np.ndarray[float]:
    r"""
    Return probability of acquisition success for given date, temperature, star
    properties and search box size.

    Any of the inputs can be scalars or arrays, with the output being the result of
    the broadcasted dimension of the inputs.

    The default probability model is defined by ``conf.default_model`` in this module.
    Allowed values for the ``model`` name are 'sota', 'spline', or a grid model
    specified by 'grid-<name>-<date>' (e.g. 'grid-floor-2018-11').

    :param date: Date(s) (scalar or np.ndarray, default=NOW)
    :param t_ccd: CCD temperature(s) (degC, scalar or np.ndarray, default=-10C)
    :param mag: Star magnitude(s) (scalar or np.ndarray, default=10.0)
    :param color: Star color(s) (scalar or np.ndarray, default=0.6)
    :param spoiler: Star spoiled (boolean or np.ndarray, default=False)
    :param halfwidth: Search box halfwidth (arcsec, default=120)
    :param model: probability model name: 'sota' | 'spline' | 'grid-\*'

    :returns: Acquisition success probability(s)
    """
    if model is None:
        model = conf.default_model

    date = DateTime(date).secs
    is_scalar, dates, t_ccds, mags, colors, spoilers, halfwidths = broadcast_arrays(
        date, t_ccd, mag, color, spoiler, halfwidth
    )

    spoilers = spoilers.astype(bool)

    # Actually evaluate the model
    if model == "sota":
        from .dark_model import get_warm_fracs

        warm_fracs = []
        for date, t_ccd in zip(dates.ravel(), t_ccds.ravel()):
            warm_frac = get_warm_fracs(WARM_THRESHOLD, date=date, T_ccd=t_ccd)
            warm_fracs.append(warm_frac)
        warm_frac = np.array(warm_fracs).reshape(dates.shape)

        probs = sota_model_acq_prob(mags, warm_fracs, colors, halfwidths)
        probs[mags < 8.5] = MAX_ACQ_PROB

    elif model == "spline":
        probs = spline_model_acq_prob(mags, t_ccds, colors, halfwidths)

    elif model.startswith("grid-"):
        probs = grid_model_acq_prob(mags, t_ccds, colors, halfwidths, model=model)

    else:
        raise ValueError("`model` parameter must be 'sota' | 'spline' | 'grid-*'")

    # Deal with color=0.7 stars and/or spoiled stars.  The spoiling correction
    # is a relic that should never be used once proseco is promoted.
    p_0p7color = 0.4294  # probability multiplier for a B-V = 0.700 star (REF?)
    p_spoiler = 0.9241  # probability multiplier for a search-spoiled star (REF?)

    probs[np.isclose(colors, 0.7, atol=1e-6, rtol=0)] *= p_0p7color
    probs[spoilers] *= p_spoiler

    if model in ("sota", "spline"):
        # Clip probabilities for older models.  Newer models (grid-* included)
        # should do all this internally.
        probs = probs.clip(MIN_ACQ_PROB, MAX_ACQ_PROB)

    # Return probabilities.  The [()] getitem at the end will flatten a
    # scalar array down to a pure scalar.
    return probs[0] if is_scalar else probs


def get_default_acq_prob_model_info(verbose=True):
    """Get info about the default acquisition probability model.

    Example::

        >>> from chandra_aca import star_probs
        >>> star_probs.get_default_acq_prob_model_info()
        {'default_model': 'grid-*',
         'call_args': {'file_path': 'chandra_models/aca_acq_prob',
          'version': None,
          'repo_path': 'None',
          'require_latest_version': False,
          'timeout': 5,
          'read_func': '<function _read_grid_func_model at 0x11a443b50>',
          'read_func_kwargs': {'model_name': None}},
         'version': '3.48',
         'commit': '68a58099a9b51bef52ef14fbd0f1971f950e6ba3',
         'data_file_path': '/Users/aldcroft/ska/data/chandra_models/chandra_models/aca_acq_prob/grid-floor-2020-02.fits.gz',
         'repo_path': '/Users/aldcroft/ska/data/chandra_models',
         'md5': '3a47774392beeca2921b705e137338f4',
         'CHANDRA_MODELS_REPO_DIR': None,
         'CHANDRA_MODELS_DEFAULT_VERSION': None,
         'THERMAL_MODELS_DIR_FOR_MATLAB_TOOLS_SW': None}

    :param verbose: bool
        If False then return trimmed version with no call_args and None values removed.
    :returns: dict of model info
    """  # noqa: E501
    info = {"default_model": conf.default_model}
    if info["default_model"].startswith("grid-"):
        gfm = get_grid_func_model()
        info.update(gfm["info"])

    # Allow a trimmed down version (e.g. for proseco to avoid bloating the pickle)
    if not verbose:
        del info["call_args"]
        for key, val in list(info.items()):
            if val is None:
                del info[key]

    return info


def clip_and_warn(name, val, val_lo, val_hi, model, tol_lo=0.0, tol_hi=0.0):
    """
    Clip ``val`` to be in the range ``val_lo`` to ``val_hi`` and issue a
    warning if clipping occurs, subject to ``tol_lo`` and ``tol_hi`` expansions.
    The ``name`` and ``model`` are just used in the warning.

    Parameters
    ----------
    name
        Value name
    val
        Value
    val_lo
        Minimum
    val_hi
        Maximum
    model
        Model name
    tol_lo
        Tolerance below ``val_lo`` for issuing a warning (default=0.0)
    tol_hi
        Tolerance above ``val_hi`` for issuing a warning (default=0.0)

    Returns
    -------
    Clipped value
    """
    val = np.asarray(val)

    # Provide a tolerance for emitting a warning clipping
    if np.any((val > val_hi + tol_hi) | (val < val_lo - tol_lo)):
        warnings.warn(
            f"\nModel {model} computed between {val_lo} <= {name} <= {val_hi}, "
            f"clipping input {name}(s) outside that range."
        )

    # Now clip to the actual limits
    if np.any((val > val_hi) | (val < val_lo)):
        val = np.clip(val, val_lo, val_hi)

    return val


def get_grid_axis_values(hdr, axis):
    """Get grid model axis values from FITS header.

    This is an irregularly-spaced grid if ``hdr`` has ``{axis}_0`` .. ``{axis}_<N-1>``.
    Otherwise it is a regularly-spaced grid::

        linspace(hdr[f"{axis}_lo"], hdr[f"{axis}_hi"], n_vals)

    Parameters
    ----------
    hdr
        FITS header (dict-like)
    axis
        Axis name (e.g. "mag")
    """
    n_vals = hdr[f"{axis}_n"]
    if all(f"{axis}_{ii}" in hdr for ii in range(n_vals)):
        # New style grid model file
        vals = np.array([hdr[f"{axis}_{ii}"] for ii in range(n_vals)])
    else:
        # Old style grid model file
        vals = np.linspace(hdr[f"{axis}_lo"], hdr[f"{axis}_hi"], n_vals)

    return vals


def get_grid_func_model(
    model: Optional[str] = None,
    version: Optional[str] = None,
    repo_path: Optional[str | Path] = None,
):
    """Get grid model from the ``model`` name.

    This reads the model data from the FITS file in the ``chandra_models`` repository.
    The ``model`` name can be a glob pattern like ``grid-*``, which will match the
    grid model with the most recent date. If not provided the ``DEFAULT_MODEL`` is used.

    The return value is a dict with necessary data to use the model::

        "filename": filepath (Path),
        "func_no_1p5": RegularGridInterpolator for stars with color != 1.5 (common)
        "func_1p5": RegularGridInterpolator for stars with color = 1.5 (less common)
        "mag_lo": lower bound of mag axis
        "mag_hi": upper bound of mag axis
        "t_ccd_lo": lower bound of t_ccd axis
        "t_ccd_hi": upper bound of t_ccd axis
        "halfw_lo": lower bound of halfw axis
        "halfw_hi": upper bound of halfw axis
        "info": dict of provenance info for model file

    Parameters
    ----------
    model
        Model name (optional), defaults to ``conf.default_model``
    version
        Version / tag / branch of ``chandra_models`` repository (optional)
    repo_path
        Path to ``chandra_models`` repository (optional)

    Returns
    -------
    dict of model data
    """
    if model is None:
        model = conf.default_model

    return _get_grid_func_model(model, version, repo_path)


@chandra_models.chandra_models_cache
def _get_grid_func_model(model, version, repo_path):
    data, info = chandra_models.get_data(
        file_path="chandra_models/aca_acq_prob",
        read_func=_read_grid_func_model,
        read_func_kwargs={"model_name": model},
        version=version,
        repo_path=repo_path,
    )
    hdu0, probit_p_fail_no_1p5, probit_p_fail_1p5 = data

    hdr = hdu0.header
    grid_mags = get_grid_axis_values(hdr, "mag")
    grid_t_ccds = get_grid_axis_values(hdr, "t_ccd")
    grid_halfws = get_grid_axis_values(hdr, "halfw")

    # Sanity checks on model data
    assert probit_p_fail_no_1p5.shape == (
        len(grid_mags),
        len(grid_t_ccds),
        len(grid_halfws),
    )
    assert probit_p_fail_1p5.shape == probit_p_fail_no_1p5.shape

    # Generate the 3-d linear interpolation functions
    func_no_1p5 = RegularGridInterpolator(
        points=(grid_mags, grid_t_ccds, grid_halfws), values=probit_p_fail_no_1p5
    )
    func_1p5 = RegularGridInterpolator(
        points=(grid_mags, grid_t_ccds, grid_halfws), values=probit_p_fail_1p5
    )
    mag_lo = hdr["mag_lo"]
    mag_hi = hdr["mag_hi"]
    t_ccd_lo = hdr["t_ccd_lo"]
    t_ccd_hi = hdr["t_ccd_hi"]
    halfw_lo = hdr["halfw_lo"]
    halfw_hi = hdr["halfw_hi"]

    out = {
        "filename": info["data_file_path"],
        "func_no_1p5": func_no_1p5,
        "func_1p5": func_1p5,
        "mag_lo": mag_lo,
        "mag_hi": mag_hi,
        "t_ccd_lo": t_ccd_lo,
        "t_ccd_hi": t_ccd_hi,
        "halfw_lo": halfw_lo,
        "halfw_hi": halfw_hi,
        "info": info,
    }
    return out


def _read_grid_func_model(models_dir: Path, model_name: Optional[str] = None):
    """Read the model file and put into local vars"""
    if model_name is None:
        model_name = conf.default_model
    filepaths = models_dir.glob(model_name + ".fits.gz")
    filepaths = list(filepaths)
    filepaths = sorted(filepaths, key=_get_date_from_model_filename)
    if len(filepaths) == 0:
        raise IOError(f"no model files found for {model_name}")

    filepath = filepaths[-1]
    if not filepath.exists():
        raise IOError(f"model file {filepath} does not exist")

    with fits.open(filepath) as hdus:
        hdu0 = hdus[0]
        probit_p_fail_no_1p5 = hdus[1].data
        probit_p_fail_1p5 = hdus[2].data

    # Pack the output data as a tuple
    data = (hdu0, probit_p_fail_no_1p5, probit_p_fail_1p5)
    return data, filepath


def _get_date_from_model_filename(filepath: Path):
    match = re.search(r"(\d{4}-\d{2})\.fits\.gz$", filepath.name)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"could not parse date from {filepath}")


def grid_model_acq_prob(
    mag=10.0, t_ccd=-12.0, color=0.6, halfwidth=120, probit=False, model=None
):
    """Calculate a grid model probability of acquisition success for a star with
    specified mag, t_ccd, color, and search box halfwidth.

    This does a 3-d linear interpolation on mag, t_ccd, and halfwidth using a
    pre-computed gridded model that is stored in a FITS file.

    Parameters
    ----------
    mag
        ACA magnitude (float or np.ndarray)
    t_ccd
        CCD temperature (degC, float or ndarray)
    color
        B-V color to check for B-V=1.5 => red star (float or np.ndarray)
    halfwidth
        search box halfwidth (arcsec, default=120, float or ndarray)
    probit
        if True then return Probit(p_success). Default=False
    model
        Model name, e.g. 'grid-floor-2018-11'

    Returns
    -------
    Acquisition success probability(s)
    """
    # Get the grid model function and model parameters from a FITS file. This function
    # call is cached.
    gfm = get_grid_func_model(model)

    func_no_1p5 = gfm["func_no_1p5"]
    func_1p5 = gfm["func_1p5"]
    mag_lo = gfm["mag_lo"]
    mag_hi = gfm["mag_hi"]
    t_ccd_lo = gfm["t_ccd_lo"]
    t_ccd_hi = gfm["t_ccd_hi"]
    halfw_lo = gfm["halfw_lo"]
    halfw_hi = gfm["halfw_hi"]

    model_filename = Path(gfm["info"]["data_file_path"]).name

    # Make sure inputs are within range of gridded model
    # TODO: run additional test cases on ASVT, make a new model, remove tol_hi for mag.
    mag = clip_and_warn("mag", mag, mag_lo, mag_hi, model_filename, tol_hi=0.25)
    t_ccd = clip_and_warn("t_ccd", t_ccd, t_ccd_lo, t_ccd_hi, model_filename)
    halfwidth = clip_and_warn("halfw", halfwidth, halfw_lo, halfw_hi, model_filename)

    # Broadcast all inputs to a common shape.  If they are all scalars
    # then shape=().  The returns values are flattened, so the final output
    # needs to be reshape at the end.
    shape, t_ccds, mags, colors, halfwidths = broadcast_arrays_flatten(
        t_ccd, mag, color, halfwidth
    )

    if shape:
        # One or more inputs are arrays, output is array with shape
        is_1p5 = np.isclose(colors, 1.5)
        not_1p5 = ~is_1p5
        p_fail = np.zeros(len(mags), dtype=np.float64)
        points = np.vstack([mags, t_ccds, halfwidths]).transpose()
        if np.any(is_1p5):
            p_fail[is_1p5] = func_1p5(points[is_1p5])
        if np.any(not_1p5):
            p_fail[not_1p5] = func_no_1p5(points[not_1p5])
        p_fail.shape = shape

    else:
        # Scalar case
        func = func_1p5 if np.isclose(color, 1.5) else func_no_1p5
        p_fail = func([[mag, t_ccd, halfwidth]])

    # Convert p_fail to p_success (remembering at this point p_fail is probit)
    p_success = -p_fail
    if not probit:
        p_success = scipy.stats.norm.cdf(p_success)

    return p_success


def spline_model_acq_prob(
    mag=10.0, t_ccd=-12.0, color=0.6, halfwidth=120, probit=False
):
    """
    Calculate poly-spline-tccd model (aka 'spline' model) probability of acquisition
    success for a star with specified mag, t_ccd, color, and search box halfwidth.

    The model definition and fit values based on:

    - https://github.com/sot/aca_stats/blob/master/fit_acq_prob_model-2018-04-poly-spline-tccd.ipynb

    See also:

    - Description of the motivation and initial model development.
      https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/StarWorkingGroupMeeting2018x04x11
    - Final review and approval.
      https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/StarWorkingGroupMeeting2018x04x18

    :param mag: ACA magnitude (float or np.ndarray)
    :param t_ccd: CCD temperature (degC, float or ndarray)
    :param color: B-V color to check for B-V=1.5 => red star (float or np.ndarray)
    :param halfwidth: search box halfwidth (arcsec, default=120, float or ndarray)
    :param probit: if True then return Probit(p_success). Default=False

    :returns: Acquisition success probability(s)
    """  # noqa: E501
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        from .cubic_spline import CubicSpline

    is_scalar, t_ccds, mags, colors, halfwidths = broadcast_arrays(
        t_ccd, mag, color, halfwidth
    )

    if np.any(t_ccds < -16.0):
        warnings.warn(
            "\nSpline model is not calibrated below -16 C, "
            "so take results with skepticism!\n"
            "For cold temperatures use the SOTA model."
        )

    # Cubic spline functions are computed on the first call and cached
    if len(SPLINE_FUNCS) == 0:
        fit_no_1p5 = np.array(
            [
                -2.69826,
                -1.96063,
                -1.20245,
                -0.01713,
                1.23724,  # P0 values
                0.07135,
                0.12711,
                0.14508,
                0.59646,
                0.64262,  # P1 values
                0.02341,
                0.0,
                0.00704,
                0.06926,
                0.05629,
            ]
        )  # P2 values
        fit_1p5 = np.array(
            [
                -2.56169,
                -1.65157,
                -0.26794,
                1.00488,
                3.52181,  # P0 values
                0.0,
                0.09193,
                0.23026,
                0.61243,
                0.94157,  # P1 values
                0.00471,
                0.00637,
                0.01118,
                0.07461,
                0.09556,
            ]
        )  # P2 values
        spline_mags = np.array([8.5, 9.25, 10.0, 10.4, 10.7])

        for vals, label in ((fit_no_1p5, "no_1p5"), (fit_1p5, "1p5")):
            SPLINE_FUNCS[0, label] = CubicSpline(
                spline_mags, vals[0:5], bc_type=((1, 0.0), (2, 0.0))
            )
            SPLINE_FUNCS[1, label] = CubicSpline(
                spline_mags, vals[5:10], bc_type=((1, 0.0), (2, 0.0))
            )
            SPLINE_FUNCS[2, label] = CubicSpline(
                spline_mags, vals[10:15], bc_type=((1, 0.0), (2, 0.0))
            )

    # Model is calibrated using t_ccd - (-12) for numerical stability.
    tc12 = t_ccds - (-12)
    box_deltas = get_box_delta(halfwidths)

    probit_p_fail = np.zeros(shape=t_ccds.shape, dtype=np.float64)
    is_1p5 = np.isclose(colors, 1.5)

    # Process the color != 1.5 stars, then the color == 1.5 stars
    for label in ("no_1p5", "1p5"):
        mask = is_1p5 if label == "1p5" else ~is_1p5

        # If no stars in this category then continue
        if not np.any(mask):
            continue

        magm = mags[mask]
        tcm = tc12[mask]
        boxm = box_deltas[mask]

        # The model is only calibrated betweeen 8.5 and 10.7.  First, clip mags going
        # into the spline to be larger than 8.5.  (Extrapolating slightly above 10.7 is
        # OK). Second, subtract a linearly varying term from probit_p_fail from stars
        # brighter than 8.5 mag ==> from 0.0 for mag=8.5 to 0.25 for mag=6.0.  This is
        # to allow star selection to favor a 6.0 mag star over 8.5 mag.
        magmc = magm.clip(8.5, None)
        bright = (8.5 - magm.clip(None, 8.5)) / 10.0

        p0 = SPLINE_FUNCS[0, label](magmc)
        p1 = SPLINE_FUNCS[1, label](magmc)
        p2 = SPLINE_FUNCS[2, label](magmc)

        probit_p_fail[mask] = p0 + p1 * tcm + p2 * tcm**2 + boxm - bright

    # Return probability of success (not failure, as in the raw model)
    p_out = -probit_p_fail

    # Return raw probit value?
    if not probit:
        p_out = scipy.stats.norm.cdf(
            p_out
        )  # transform from probit to linear probability

    return p_out


def model_acq_success_prob(mag, warm_frac, color=0, halfwidth=120):
    """
    Call sota_model_acq_prob() with same params.  This is retained purely
    for back-compatibility but use is deprecated.
    """
    return sota_model_acq_prob(mag, warm_frac, color, halfwidth)


def sota_model_acq_prob(mag, warm_frac, color=0, halfwidth=120):
    """
    Calculate raw SOTA model probability of acquisition success.

    This is for a star with ``mag`` magnitude and a CCD warm fraction ``warm_frac``.
    This is not typically used directly since it does not account for star properties
    like spoiled or color=0.7.

    Uses the empirical relation::

       P_fail_probit = offset(mag) + scale(mag) * warm_frac + box_delta(halfwidth)
       P_acq_fail = Normal_CDF()
       P_acq_success = 1 - P_acq_fail

    This is based on the dark model and acquisition success model presented in the State
    of the ACA 2013, and subsequently updated to use a Probit transform and separately
    fit B-V=1.5 stars.  It was updated in 2017 to include a fitted dependence on the
    search box halfwidth.  See:

    https://github.com/sot/skanb/blob/master/pea-test-set/fit_box_size_acq_prob.ipynb
    https://github.com/sot/aca_stats/blob/master/fit_acq_prob_model-2017-07-sota.ipynb

    Parameters
    ----------
    mag
        ACA magnitude (float or np.ndarray)
    warm_frac
        N100 warm fraction (float or np.ndarray)
    color
        B-V color to check for B-V=1.5 => red star (float or np.ndarray)
    halfwidth
        search box halfwidth (arcsec, default=120)

    Returns
    -------
    Acquisition success probability(s)
    """
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

    SOTA_FIT_NO_1P5_WITH_MS = [
        9.6887121605441173,  # scl0
        9.1613040261776177,  # scl1
        -0.41919343599067715,  # scl2
        -2.3829996965532048,  # off0
        0.54998934814773903,  # off1
        0.47839260691599156,
    ]  # off2

    SOTA_FIT_ONLY_1P5_WITH_MS = [
        8.541709287866361,
        0.44482688155644085,
        -3.5137852251178465,
        -1.3505424393223699,
        1.5278061271148755,
        0.30973569068842272,
    ]

    #
    # FLIGHT model coefficients.
    #
    # Multiple stars flag disabled (starting operationally with FEB0816).  Fit
    # with fit_flight_acq_prob_model.ipynb in the aca_stats repo.

    SOTA_FIT_NO_1P5_NO_MS = [
        4.38145,  # scl0
        6.22480,  # scl1
        2.20862,  # scl2
        -2.24494,  # off0
        0.32180,  # off1
        0.08306,  # off2
    ]

    SOTA_FIT_ONLY_1P5_NO_MS = [
        4.73283,  # scl0
        7.63540,  # scl1
        4.56612,  # scl2
        -1.49046,  # off0
        0.53391,  # off1
        -0.37074,  # off2
    ]

    if MULT_STARS_ENABLED:
        SOTA_FIT_NO_1P5 = SOTA_FIT_NO_1P5_WITH_MS
        SOTA_FIT_ONLY_1P5 = SOTA_FIT_ONLY_1P5_WITH_MS
    else:
        SOTA_FIT_NO_1P5 = SOTA_FIT_NO_1P5_NO_MS
        SOTA_FIT_ONLY_1P5 = SOTA_FIT_ONLY_1P5_NO_MS

    is_scalar, mag, warm_frac, color = broadcast_arrays(mag, warm_frac, color)

    m10 = mag - 10.0

    p_fail = np.zeros_like(mag)
    color1p5 = np.isclose(color, 1.5, atol=1e-6, rtol=0)
    for mask, fit_pars in ((color1p5, SOTA_FIT_ONLY_1P5), (~color1p5, SOTA_FIT_NO_1P5)):
        if np.any(mask):
            scale = np.polyval(fit_pars[0:3][::-1], m10)
            offset = np.polyval(fit_pars[3:6][::-1], m10)
            box_delta = get_box_delta(halfwidth)

            p_fail[mask] = (offset + scale * warm_frac + box_delta)[mask]

    p_fail = scipy.stats.norm.cdf(p_fail)  # probit transform
    p_fail[mag < 8.5] = 0.015  # actual best fit is ~0.006, but put in some conservatism
    p_success = 1 - p_fail

    # Clip values to reasonable range regardless of model prediction
    p_success = p_success.clip(MIN_ACQ_PROB, MAX_ACQ_PROB)

    return p_success[0] if is_scalar else p_success  # Return scalar if ndim=0


def mag_for_p_acq(p_acq, date=None, t_ccd=-10.0, halfwidth=120, model=None):
    """
    For a given ``date`` and ``t_ccd``, find the star magnitude that has an
    acquisition probability of ``p_acq``.  Star magnitude is defined/limited
    to the range 5.0 - 12.0 mag.

    Parameters
    ----------
    p_acq
        acquisition probability (0 to 1.0)
    date
        observation date (any Chandra.Time valid format)
    t_ccd
        ACA CCD temperature (deg C)
    halfwidth
        search box halfwidth (arcsec, default=120)
    model : probability model (see acq_success_prob() for allowed values, default)
        :returns mag: star magnitude
    """

    def prob_minus_p_acq(mag):
        """Function that gets zeroed in brentq call later"""
        prob = acq_success_prob(
            date=date, t_ccd=t_ccd, mag=mag, halfwidth=halfwidth, model=model
        )
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


def guide_count(mags, t_ccd, count_9th=False):
    """Calculate a guide star fractional count/metric using signal-to-noise scaled
    mag thresholds.

    This uses a modification of the guide star fractional counts that were
    suggested at the 7-Mar-2018 SSAWG and agreed upon at the 21-Mar-2018
    SSAWG.  The implementation here does a piecewise linear interpolation
    between the reference mag - fractional count points instead of the
    original "threshold interpolation" (nearest neighbor mag <= reference
    mag).  Approved at 16-Jan-2019 SSAWG.

    One feature is the slight incline in the guide_count curve from 1.0005 at
    mag=6.0 to 1.0 at mag=10.0.  This does not show up in standard outputs
    of guide_counts to two decimal places (``8 * 0.0005 = 0.004``), but helps with
    minimization.

    Parameters
    ----------
    mags : float, array
        Star magnitude(s)
    t_ccds : float, array
        CCD temperature(s)
    count_9th : bool
        Return fractional count of 9th mag or brighter stars

    Returns
    -------
    float, fractional count
    """
    mags = np.atleast_1d(mags)
    mags, t_ccds = np.broadcast_arrays(mags, t_ccd)

    # Generate interpolation curve for each specified input ``t_ccd``
    ref_t_ccd = -10.9
    ref_mags0 = (9.0 if count_9th else 9.95) + np.array([0.0, 0.2, 0.3, 0.4])
    ref_mags = {}
    for t_ccd in np.unique(t_ccds):
        # The 5.25 and 5.35 limits are not temperature dependent, these reflect the
        # possibility that the star will be brighter than 5.2 mag and the OBC will
        # reject it.  Note that around 6th mag mean observed catalog error is
        # around 0.1 mag.
        ref_mags[t_ccd] = np.concatenate(
            [[5.25, 5.35], snr_mag_for_t_ccd(t_ccd, ref_mags0, ref_t_ccd)]
        )

    ref_counts = [0.0, 1.0005, 1.0, 0.75, 0.5, 0.0]

    # Do the interpolation, noting that np.interp will use the end ``counts``
    # values for any ``mag`` < ref_mags[0] or > ref_mags[-1].
    count = 0.0
    for mag, t_ccd in zip(mags, t_ccds):
        count += np.interp(mag, ref_mags[t_ccd], ref_counts)

    return count


def t_ccd_warm_limit_for_guide(
    mags, min_guide_count=4.0, warm_t_ccd=-5.0, cold_t_ccd=-16.0
):
    """
    Solve for the warmest temperature that still gets the min_guide_count.
    This returns a value between ``cold_t_ccd`` and ``warm_t_ccd``.  At the
    cold end the result may be below ``min_n_acq``, in which case the star catalog
    may be rejected.

    Parameters
    ----------
    mags
        list of star ACA mags
    min_guide_count
        float minimum fractional guide count
    warm_t_ccd
        warmest CCD temperature to consider (default=-5 C)
    cold_t_ccd
        coldest CCD temperature to consider (default=-16 C)

    Returns
    -------
    t_ccd
    """
    if guide_count(mags, warm_t_ccd) >= min_guide_count:
        return warm_t_ccd
    if guide_count(mags, cold_t_ccd) <= min_guide_count:
        # Note that this relies on a slight incline in the guide_count curve
        # from 1.0005 at mag=6.0 to 1.0 at mag=10.0.
        return cold_t_ccd

    def merit_func(t_ccd):
        count = guide_count(mags, t_ccd)
        return count - min_guide_count

    return bisect(
        merit_func, cold_t_ccd, warm_t_ccd, xtol=0.001, rtol=1e-15, full_output=False
    )


def binom_ppf(k, n, conf, n_sample=1000):
    """
    Compute percent point function (inverse of CDF) for binomial, where
    the percentage is with respect to the "p" (binomial probability) parameter
    not the "k" parameter.  The latter is what one gets with scipy.stats.binom.ppf().

    This function internally generates ``n_sample`` samples of the binomal PMF in
    order to compute the CDF and finally interpolate to get the PPF.

    The following example returns the 1-sigma (0.17 - 0.84) confidence interval
    on the true binomial probability for an experiment with 4 successes in 5 trials.

    Example::

      >>> binom_ppf(4, 5, [0.17, 0.84])
      array([ 0.55463945,  0.87748177])

    Parameters
    ----------
    k
        int, number of successes (0 < k <= n)
    n
        int, number of trials
    conf
        float or array of floats, percent point values
    n_sample
        number of PMF samples for interpolation

    Returns
    -------
    percent point function values corresponding to ``conf``
    """
    ps = np.linspace(0, 1, n_sample)  # Probability values
    pmfs = scipy.stats.binom.pmf(k=k, n=n, p=ps)
    cdf = np.cumsum(pmfs) / np.sum(pmfs)
    out = np.interp(conf, xp=cdf, fp=ps)

    return out


def binomial_confidence_interval(
    n_success: ArrayLike, n_trials: ArrayLike, coverage: float = 0.682689
) -> tuple:
    """Binomial error calculation using the Jeffreys prior.

    It returns a tuple with the ratio, the lower error, and the upper error.

    This is an equal-tailed Bayesian interval with a Jeffreys prior (Beta(1/2,1/2)).

        "Statistical Decision Theory and Bayesian Analysis" 2nd ed.
        Berger, J.O. (1985)
        Springer, New York

    This function is based on:

        "Confidence Intervals for a binomial proportion and asymptotic expansions",
        Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta
        Ann. Statist. Volume 30, Number 1 (2002), 160-201.
        http://projecteuclid.org/euclid.aos/1015362189

    Parameters
    ----------
    n_success : numpy array
        The number of 'successes'
    n_trials : numpy array
        The number of trials
    coverage : float
        The coverage of the confidence interval. The default corresponds to the coverage of
        '1-sigma' gaussian errors (0.682689).

    Returns
    -------
    ratio : tuple
        ratio of successes to trials, lower bound, upper bound
    """
    # keeping shape to make sure the output has the same shape as the input
    n_success, n_trials = np.broadcast_arrays(n_success, n_trials)
    shape = n_success.shape

    if np.any(n_trials < n_success):
        raise ValueError("n_trials must be greater than or equal to n_success")
    if np.any(n_trials < 0):
        raise ValueError("n_trials must be greater or equal to 0")
    if np.any(n_success < 0):
        raise ValueError("n_success must be greater or equal to 0")

    # normalize the input as numpy arrays
    n_trials = np.atleast_1d(n_trials)
    n_success = np.atleast_1d(n_success)
    # calculate the ratio
    ok = n_trials != 0
    ratio = np.ones_like(n_success) * np.nan
    ratio[ok] = n_success[ok] / n_trials[ok]

    # calculate the confidence intervals
    alpha = (1 - coverage) / 2
    low = np.zeros_like(ratio)
    up = np.ones_like(ratio)
    low[ratio > 0] = beta.isf(
        1 - alpha,
        n_success[ratio > 0] + 0.5,
        n_trials[ratio > 0] - n_success[ratio > 0] + 0.5,
    )
    up[ratio < 1] = beta.isf(
        alpha,
        n_success[ratio < 1] + 0.5,
        n_trials[ratio < 1] - n_success[ratio < 1] + 0.5,
    )
    return (ratio.reshape(shape), low.reshape(shape), up.reshape(shape))
