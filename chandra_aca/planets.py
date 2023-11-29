# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for planet position relative to Chandra, Earth, or Solar System
Barycenter.

Estimated accuracy of planet coordinates (RA, Dec) is as follows, where the JPL
Horizons positions are used as the "truth".

- `get_planet_chandra` errors:
    - Venus: < 4 arcsec with a peak around 3.5
    - Mars: < 3 arcsec with a peak around 2.0
    - Jupiter: < 0.8 arcsec
    - Saturn: < 0.5 arcsec

- `get_planet_eci` errors:
    - Venus: < 12 arcmin with peak around 2 arcmin
    - Mars: < 8 arcmin with peak around 1.5 arcmin
    - Jupiter: < 1 arcmin with peak around 0.5 arcmin
    - Saturn: < 0.5 arcmin with peak around 0.3 arcmin

See the ``validation/planet-accuracy.ipynb`` notebook for details.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import astropy.units as u
import jplephem.spk
import numba
import numpy as np
from astropy.io import ascii
from cxotime import CxoTime, CxoTimeLike
from ska_helpers.utils import LazyVal

from chandra_aca.transform import eci_to_radec

__all__ = (
    "get_planet_chandra",
    "get_planet_barycentric",
    "get_planet_eci",
    "get_planet_chandra_horizons",
    "get_planet_angular_sep",
    "NoEphemerisError",
    "GET_PLANET_ECI_ERRORS",
    "GET_PLANET_CHANDRA_ERRORS",
)

GET_PLANET_ECI_ERRORS = {
    "venus": 12 * u.arcmin,
    "mars": 8 * u.arcmin,
    "jupiter": 1 * u.arcmin,
    "saturn": 0.5 * u.arcmin,
}
GET_PLANET_CHANDRA_ERRORS = {
    "Venus": 4 * u.arcsec,
    "Mars": 3 * u.arcsec,
    "Jupiter": 0.8 * u.arcsec,
    "Saturn": 0.5 * u.arcsec,
}


class NoEphemerisError(Exception):
    """If there is no Chandra orbital ephemeris available"""

    pass


def load_kernel():
    kernel_path = Path(__file__).parent / "data" / "de432s.bsp"
    if not kernel_path.exists():
        raise FileNotFoundError(
            f"kernel data file {kernel_path} not found, "
            'run "python setup.py build" to install it locally'
        )
    kernel = jplephem.spk.SPK.open(kernel_path)
    import atexit

    atexit.register(_close_kernel)
    return kernel


def _close_kernel():
    KERNEL.val.close()


JPLEPHEM_T0 = jplephem.spk.T0
JPLEPHEM_S_PER_DAY = jplephem.spk.S_PER_DAY
KERNEL = LazyVal(load_kernel)
BODY_NAME_TO_KERNEL_SPEC = dict(
    [
        ("sun", [(0, 10)]),
        ("mercury", [(0, 1), (1, 199)]),
        ("venus", [(0, 2), (2, 299)]),
        ("earth-moon-barycenter", [(0, 3)]),
        ("earth", [(0, 3), (3, 399)]),
        ("moon", [(0, 3), (3, 301)]),
        ("mars", [(0, 4)]),
        ("jupiter", [(0, 5)]),
        ("saturn", [(0, 6)]),
        ("uranus", [(0, 7)]),
        ("neptune", [(0, 8)]),
        ("pluto", [(0, 9)]),
    ]
)
URL_HORIZONS = "https://ssd.jpl.nasa.gov/api/horizons.api?"

# NOTE: using TDB scale is important because the JPL ephemeris requires JD in TDB.
# Without that CxoTime(0.0).jd is the JD in UTC which is different by 63.184 s.
JD_CXCSEC0_TDB = CxoTime(0.0).tdb.jd


def convert_time_format_spk(time, fmt_out):
    """Fast version of convert_time_format for use with JPLEPHEM SPK.

    For use with JPLEPHEM SPK, which requires JD in TDB.

    This is much faster for float (secs) input and "secs" or "jd" output formats.

    For "jd" and formats other than "secs", the output is in TDB. For "secs" the output
    is in seconds since 1998.0 (TT), which is unaffected by the time scale.

    This function is suitable for using in planet position calculations since it is good
    to about 1 ms.

    Parameters
    ----------
    time : CxoTimeLike
        Time or times
    fmt_out : str
        Output format (any supported CxoTime format)

    Returns
    -------
    ndarray or numpy scalar
        Converted time or times
    """
    if fmt_out not in ("jd", "secs"):
        return getattr(CxoTime(time).tdb, fmt_out)

    # Check if input is in "secs" by determining if it is a float type scalar or array
    not_secs = True
    if isinstance(time, float):
        not_secs = False
    else:
        time = np.asarray(time)
        if time.dtype.kind == "f":
            not_secs = False

    if not_secs:
        out = getattr(CxoTime(time).tdb, fmt_out)
    elif fmt_out == "jd":
        out = JD_CXCSEC0_TDB + time / 86400.0
    else:
        # fmt_out == 'secs' and input time format is secs so no conversion is needed.
        out = time

    return out


def get_planet_angular_sep(
    body: str, ra: float, dec: float, time=None, observer_position: str = "earth"
) -> float:
    """Get angular separation between planet ``body`` and target ``ra``, ``dec``.

    Valid values for the ``observer_position`` argument are:

    - 'earth' (default, approximate, fastest)
    - 'chandra' (reasonably accurate fast, requires fetching ephemeris)
    - 'chandra-horizons' (most accurate, slow, requires internet access)

    Parameters
    ----------
    body : str
        Body name (lower case planet name)
    ra : float
        RA in degrees
    dec : float
        Dec in degrees
    time : CxoTime-compatible object
        Time or times of observation
    observer_position : str
        Observer position ('earth', 'chandra', or 'chandra-horizons')

    Returns
    -------
    angular separation (deg)
    """
    from agasc import sphere_dist

    time_secs = convert_time_format_spk(time, "secs")

    if observer_position == "earth":
        eci = get_planet_eci(body, time_secs)
        body_ra, body_dec = eci_to_radec(eci)
    elif observer_position == "chandra":
        eci = get_planet_chandra(body, time_secs)
        body_ra, body_dec = eci_to_radec(eci)
    elif observer_position == "chandra-horizons":
        time_secs = np.asarray(time_secs)

        if time_secs.shape == ():
            time_secs = [time_secs, time_secs + 1000]
            is_scalar = True
        else:
            is_scalar = False
        pos = get_planet_chandra_horizons(
            body, time_secs[0], time_secs[-1], n_times=len(time_secs)
        )
        body_ra = pos["ra"]
        body_dec = pos["dec"]
        if is_scalar:
            body_ra = body_ra[0]
            body_dec = body_dec[0]
    else:
        raise ValueError(
            f"{observer_position} is not an allowed value: "
            '("earth", "chandra", or "chandra-horizons")'
        )

    sep = sphere_dist(ra, dec, body_ra, body_dec)
    return sep


@numba.njit(cache=True)
def _spk_compute_scalar(tdb, init, intlen, coefficients):
    index_float, offset = np.divmod(
        (tdb - JPLEPHEM_T0) * JPLEPHEM_S_PER_DAY - init, intlen
    )
    index = int(index_float)
    return _compute_components(intlen, coefficients, offset, index)


@numba.njit(cache=True)
def _spk_compute_array(tdb, init, intlen, coefficients):
    index_float, offset = np.divmod(
        (tdb - JPLEPHEM_T0) * JPLEPHEM_S_PER_DAY - init, intlen
    )
    index = index_float.astype(np.int64)
    return _compute_components(intlen, coefficients, offset, index)

@numba.njit(cache=True)
def _compute_components(intlen, coefficients, offset, index):
    coefficients = coefficients[:, :, index]

    # Chebyshev polynomial.
    s = 2.0 * offset / intlen - 1.0
    s2 = 2.0 * s

    w0 = w1 = 0.0 * coefficients[0]
    for coefficient in coefficients[:-1]:
        w2 = w1
        w1 = w0
        w0 = coefficient + (s2 * w1 - w2)

    components = coefficients[-1] + (s * w0 - w1)
    return components


def spk_compute(segment, tdb):
    init, intlen, coefficients = segment._data

    # is_array = bool(getattr(tdb, "shape", ()))
    is_array = isinstance(tdb, np.ndarray)
    func = _spk_compute_array if is_array else _spk_compute_scalar
    out = func(tdb, init, intlen, coefficients)

    return out


def get_planet_barycentric(body: str, time: CxoTimeLike = None):
    """Get barycentric position for solar system ``body`` at ``time``.

    This uses the built-in JPL ephemeris file DE432s and jplephem.

    ``body`` must be one of "sun", "mercury", "venus", "earth-moon-barycenter", "earth",
    "moon", "mars", "jupiter", "saturn", "uranus", "neptune", or "pluto".

    Parameters
    ----------
    body
        Body name (lower case planet name)
    time
        Time or times for returned position (default=NOW)

    Returns
    -------
    barycentric position (km) as (x, y, z) or N x (x, y, z)
    """
    kernel = KERNEL.val
    if body not in BODY_NAME_TO_KERNEL_SPEC:
        raise ValueError(
            f"{body} is not an allowed value {tuple(BODY_NAME_TO_KERNEL_SPEC)}"
        )

    spk_pairs = BODY_NAME_TO_KERNEL_SPEC[body]
    time_jd = convert_time_format_spk(time, "jd")
    kernel_pairs = (kernel[spk_pair] for spk_pair in spk_pairs)
    pos_list = [spk_compute(kp, time_jd) for kp in kernel_pairs]
    pos = np.sum(pos_list, axis=0)

    return pos.transpose()  # SPK returns (3, N) but we need (N, 3)


def get_planet_eci(
    body: str, time: CxoTimeLike = None, pos_observer: Optional[str] = None
):
    """Get ECI apparent position for solar system ``body`` at ``time``.

    This uses the built-in JPL ephemeris file DE432s and jplephem. The position
    is computed at the supplied ``time`` minus the light-travel time from the
    observer to ``body`` to generate the apparent position on Earth at ``time``.

    ``body`` and ``pos_observer`` must be one of "sun", "mercury", "venus",
    "earth-moon-barycenter", "earth", "moon", "mars", "jupiter", "saturn", "uranus",
    "neptune", or "pluto".

    Estimated accuracy of planet coordinates (RA, Dec) is as follows, where the
    JPL Horizons positions are used as the "truth". This assumes the observer
    position is Earth (default).

    - Venus: < 12 arcmin with peak around 2 arcmin
    - Mars: < 8 arcmin with peak around 1.5 arcmin
    - Jupiter: < 1 arcmin with peak around 0.5 arcmin
    - Saturn: < 0.5 arcmin with peak around 0.3 arcmin

    Parameters
    ----------
    body
        Body name (lower case planet name)
    time
        Time or times for returned position (default=NOW)
    pos_observer
        Observer position (default=Earth)

    Returns
    -------
    ndarray
        Earth-Centered Inertial (ECI) position (km) as (x, y, z)
        or N x (x, y, z)
    """
    time_sec = convert_time_format_spk(time, "secs")

    pos_planet = get_planet_barycentric(body, time_sec)
    if pos_observer is None:
        pos_observer = get_planet_barycentric("earth", time_sec)

    dist = np.sqrt(np.sum((pos_planet - pos_observer) ** 2, axis=-1))  # km
    # Divide distance by the speed of light in km/s
    light_travel_time = dist / 299792.458  # s

    pos_planet = get_planet_barycentric(body, time_sec - light_travel_time)

    return pos_planet - pos_observer


def get_planet_chandra(body: str, time: CxoTimeLike = None):
    """Get position for solar system ``body`` at ``time`` relative to Chandra.

    This uses the built-in JPL ephemeris file DE432s and jplephem, along with
    the CXC predictive Chandra orbital ephemeris (from the OFLS). The position
    is computed at the supplied ``time`` minus the light-travel time from
    Chandra to ``body`` to generate the apparent position from Chandra at
    ``time``.

    Estimated accuracy of planet coordinates (RA, Dec) from Chandra is as
    follows, where the JPL Horizons positions are used as the "truth".

    - Venus: < 4 arcsec with a peak around 3.5
    - Mars: < 3 arcsec with a peak around 2.0
    - Jupiter: < 0.8 arcsec
    - Saturn: < 0.5 arcsec

    Parameters
    ----------
    body
        Body name
    time
        Time or times for returned position (default=NOW)

    Returns
    -------
    position relative to Chandra (km) as (x, y, z) or N x (x, y, z)
    """
    from cheta import fetch

    time = CxoTime(time)

    # Get position of Chandra relative to Earth
    try:
        dat = fetch.MSIDset(
            ["orbitephem0_x", "orbitephem0_y", "orbitephem0_z"],
            np.min(time) - 500 * u.s,
            np.max(time) + 500 * u.s,
        )
    except ValueError:
        raise NoEphemerisError("Chandra ephemeris not available")

    if len(dat["orbitephem0_x"].vals) == 0:
        raise NoEphemerisError("Chandra ephemeris not available")

    times = np.atleast_1d(time.secs)
    ephem = {
        key: np.interp(times, dat[key].times, dat[key].vals)
        for key in dat
    }

    pos_earth = get_planet_barycentric("earth", time)

    # Chandra position in km
    chandra_eci = np.zeros_like(pos_earth)
    chandra_eci[..., 0] = ephem["orbitephem0_x"].reshape(time.shape) / 1000
    chandra_eci[..., 1] = ephem["orbitephem0_y"].reshape(time.shape) / 1000
    chandra_eci[..., 2] = ephem["orbitephem0_z"].reshape(time.shape) / 1000
    planet_chandra = get_planet_eci(body, time, pos_observer=pos_earth + chandra_eci)

    return planet_chandra


def get_planet_chandra_horizons(
    body: Union[str, int],
    timestart: CxoTimeLike,
    timestop: CxoTimeLike,
    n_times: int = 10,
    timeout: float = 10,
):
    """Get body position and other info as seen from Chandra using JPL Horizons.

    In addition to the planet names, the ``body`` argument can be any identifier that
    Horizon supports, e.g. ``sun`` or ``geo`` (Earth geocenter).

    This function queries the JPL Horizons site using the web API interface
    (See https://ssd-api.jpl.nasa.gov/doc/horizons.html for docs).

    The return value is an astropy Table with columns: time, ra, dec, rate_ra,
    rate_dec, mag, surf_brt, ang_diam. The units are included in the table
    columns. The ``time`` column is a ``CxoTime`` object.

    The returned Table has a meta key value ``response_text`` with the full text
    of the Horizons response and a ``response_json`` key with the parsed JSON.

    Example::

      >>> from chandra_aca.planets import get_planet_chandra_horizons
      >>> dat = get_planet_chandra_horizons('jupiter', '2020:001', '2020:002', n_times=4)
      >>> dat
      <Table length=5>
               time             ra       dec     rate_ra    rate_dec    mag      surf_brt   ang_diam
                               deg       deg    arcsec / h arcsec / h   mag   mag / arcsec2  arcsec
              object         float64   float64   float64    float64   float64    float64    float64
      --------------------- --------- --------- ---------- ---------- ------- ------------- --------
      2020:001:00:00:00.000 276.96494 -23.20087      34.22       0.98  -1.839         5.408    31.75
      2020:001:06:00:00.000 277.02707 -23.19897      34.30       1.30  -1.839         5.408    31.76
      2020:001:12:00:00.000 277.08934 -23.19652      34.39       1.64  -1.839         5.408    31.76
      2020:001:18:00:00.000 277.15181 -23.19347      34.51       2.03  -1.839         5.408    31.76
      2020:002:00:00:00.000 277.21454 -23.18970      34.69       2.51  -1.839         5.408    31.76

    Parameters
    ----------
    body : one of 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
        'uranus', 'neptune', or any other body that Horizons supports.
    timestart
        start time (any CxoTime-compatible time)
    timestop
        stop time (any CxoTime-compatible time)
    n_times
        number of time samples (inclusive, default=10)
    timeout
        timeout for query to Horizons (secs)

    Returns
    -------
    Table of information
    """
    import requests

    timestart = CxoTime(timestart)
    timestop = CxoTime(timestop)
    planet_ids = {
        "mercury": "199",
        "venus": "299",
        "mars": "499",
        "jupiter": "599",
        "saturn": "699",
        "uranus": "799",
        "neptune": "899",
    }

    params = dict(
        COMMAND=planet_ids.get(body, str(body).lower()),
        MAKE_EPHEM="YES",
        CENTER="@-151",
        TABLE_TYPE="OBSERVER",
        ANG_FORMAT="DEG",
        START_TIME=timestart.datetime.strftime("%Y-%b-%d %H:%M"),
        STOP_TIME=timestop.datetime.strftime("%Y-%b-%d %H:%M"),
        STEP_SIZE=str(n_times - 1),
        QUANTITIES="1,3,9,13",
        CSV_FORMAT="YES",
    )

    # The HORIZONS web API seems to require all params to be quoted strings.
    # See: https://ssd-api.jpl.nasa.gov/doc/horizons.html
    for key, val in params.items():
        params[key] = repr(val)
    resp = requests.get(URL_HORIZONS, params=params, timeout=timeout)

    if resp.status_code != requests.codes["ok"]:
        raise ValueError(
            f"request {resp.url} failed: {resp.reason} ({resp.status_code})"
        )

    resp_json: dict = resp.json()
    result: str = resp_json["result"]
    lines = result.splitlines()

    if "$$SOE" not in lines:
        msg = "problem with Horizons query:\n" + "\n".join(lines)
        raise ValueError(msg)

    idx0 = lines.index("$$SOE") + 1
    idx1 = lines.index("$$EOE")
    lines = lines[idx0:idx1]
    dat = ascii.read(
        lines,
        format="no_header",
        delimiter=",",
        names=[
            "time",
            "null1",
            "null2",
            "ra",
            "dec",
            "rate_ra",
            "rate_dec",
            "mag",
            "surf_brt",
            "ang_diam",
            "null3",
        ],
        fill_values=[("n.a.", "0")],
    )

    times = [datetime.strptime(val[:20], "%Y-%b-%d %H:%M:%S") for val in dat["time"]]
    dat["time"] = CxoTime(times, format="datetime")
    dat["time"].format = "date"
    dat["ra"].info.unit = u.deg
    dat["dec"].info.unit = u.deg
    dat["rate_ra"].info.unit = u.arcsec / u.hr
    dat["rate_dec"].info.unit = u.arcsec / u.hr
    dat["mag"].info.unit = u.mag
    dat["surf_brt"].info.unit = u.mag / (u.arcsec**2)
    dat["ang_diam"].info.unit = u.arcsec

    dat["ra"].info.format = ".5f"
    dat["dec"].info.format = ".5f"
    dat["rate_ra"].info.format = ".2f"
    dat["rate_dec"].info.format = ".2f"
    dat["mag"].info.format = ".3f"
    dat["surf_brt"].info.format = ".3f"
    dat["ang_diam"].info.format = ".2f"

    dat.meta["response_text"] = resp.text
    dat.meta["response_json"] = resp_json

    del dat["null1"]
    del dat["null2"]
    del dat["null3"]

    return dat
