# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for planet position relative to Chandra, Earth, or Solar System Barycenter.
"""
from datetime import datetime
from pathlib import Path

import astropy.constants as const
import astropy.units as u
from astropy.io import ascii
import numpy as np
from cxotime import CxoTime
from ska_helpers.utils import LazyVal

__all__ = ('get_planet_chandra', 'get_planet_barycentric', 'get_planet_eci',
           'get_planet_chandra_horizons')


def load_kernel():
    from jplephem.spk import SPK
    kernel_path = Path(__file__).parent / 'data' / 'de432s.bsp'
    if not kernel_path.exists():
        raise FileNotFoundError(f'kernel data file {kernel_path} not found, '
                                'run "python setup.py build" to install it locally')
    kernel = SPK.open(kernel_path)
    return kernel


KERNEL = LazyVal(load_kernel)
BODY_NAME_TO_KERNEL_SPEC = dict([
    ('sun', [(0, 10)]),
    ('mercury', [(0, 1), (1, 199)]),
    ('venus', [(0, 2), (2, 299)]),
    ('earth-moon-barycenter', [(0, 3)]),
    ('earth', [(0, 3), (3, 399)]),
    ('moon', [(0, 3), (3, 301)]),
    ('mars', [(0, 4)]),
    ('jupiter', [(0, 5)]),
    ('saturn', [(0, 6)]),
    ('uranus', [(0, 7)]),
    ('neptune', [(0, 8)]),
    ('pluto', [(0, 9)])
])
URL_HORIZONS = 'https://ssd.jpl.nasa.gov/horizons_batch.cgi?'


def get_planet_barycentric(body, time=None):
    """Get barycentric position for solar system ``body`` at ``time``.

    This uses the built-in JPL ephemeris file DE432s and jplephem.

    :param body: Body name (lower case planet name)
    :param time: Time or times for returned position (default=NOW)
    :returns: barycentric position (km) as (x, y, z) or N x (x, y, z)
    """
    kernel = KERNEL.val
    if body not in BODY_NAME_TO_KERNEL_SPEC:
        raise ValueError(f'{body} is not an allowed value '
                         f'{tuple(BODY_NAME_TO_KERNEL_SPEC)}')

    spk_pairs = BODY_NAME_TO_KERNEL_SPEC[body]
    time = CxoTime(time)
    time_jd = time.jd
    pos = kernel[spk_pairs[0]].compute(time_jd)
    for spk_pair in spk_pairs[1:]:
        pos += kernel[spk_pair].compute(time_jd)

    return pos.transpose()  # SPK returns (3, N) but we need (N, 3)


def get_planet_eci(body, time=None):
    """Get ECI apparent position for solar system ``body`` at ``time``.

    This uses the built-in JPL ephemeris file DE432s and jplephem. The position
    is computed at the supplied ``time`` minus the light-travel time from Earth
    to ``body`` to generate the apparent position on Earth at ``time``.

    :param body: Body name (lower case planet name)
    :param time: Time or times for returned position (default=NOW)
    :returns: Earth-Centered Inertial (ECI) position (km) as (x, y, z)
        or N x (x, y, z)
    """
    time = CxoTime(time)

    pos_planet = get_planet_barycentric(body, time)
    pos_earth = get_planet_barycentric('earth', time)

    dist = np.sqrt(np.sum((pos_planet - pos_earth) ** 2, axis=-1)) * u.km
    light_travel_time = (dist / const.c).to(u.s)

    pos_planet = get_planet_barycentric(body, time - light_travel_time)

    return pos_planet - pos_earth


def get_planet_chandra(body, time=None):
    """Get position for solar system ``body`` at ``time`` relative to Chandra.

    This uses the built-in JPL ephemeris file DE432s and jplephem, along with
    the CXC predictive Chandra orbital ephemeris (from the OFLS). The position
    is computed at the supplied ``time`` minus the light-travel time from Earth
    to ``body`` to generate the apparent position on Earth at ``time``.

    :param body: Body name
    :param time: Time or times for returned position (default=NOW)
    :returns: position relative to Chandra (km) as (x, y, z) or N x (x, y, z)
    """
    from cheta import fetch

    time = CxoTime(time)

    planet_eci = get_planet_eci(body, time)

    # Get position of Chandra relative to Earth
    dat = fetch.MSIDset(['orbitephem0_x', 'orbitephem0_y', 'orbitephem0_z'],
                        np.min(time) - 500 * u.s, np.max(time) + 500 * u.s)
    times = np.atleast_1d(time.secs)
    dat.interpolate(times=times)

    # Chandra position in km
    x = dat['orbitephem0_x'].vals.reshape(time.shape) / 1000
    y = dat['orbitephem0_y'].vals.reshape(time.shape) / 1000
    z = dat['orbitephem0_z'].vals.reshape(time.shape) / 1000

    # Planet position relative to Chandra:
    #   Planet relative to Earth - Chandra relative to Earth
    planet_chandra = planet_eci
    planet_chandra[..., 0] -= x
    planet_chandra[..., 1] -= y
    planet_chandra[..., 2] -= z

    return planet_chandra


def get_planet_chandra_horizons(body, timestart, timestop, n_times=10, timeout=10):
    """Get body position, rate, mag, surface brightness, diameter from Horizons.

    This function queries the JPL Horizons site using the CGI batch interface
    (See https://ssd.jpl.nasa.gov/horizons_batch.cgi for docs).

    The return value is an astropy Table with columns: time, ra, dec, rate_ra,
    rate_dec, mag, surf_brt, ang_diam. The units are included in the table
    columns. The ``time`` column is a ``CxoTime`` object.

    The returned Table has a meta key value ``response_text`` with the full text
    of the Horizons response.

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

    :param body: one of 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
        'uranus', 'neptune'
    :param timestart: start time (any CxoTime-compatible time)
    :param timestop: stop time (any CxoTime-compatible time)
    :param n_times: number of time samples (inclusive, default=10)
    :param timeout: timeout for query to Horizons (secs)

    :returns: Table of information

    """
    import requests

    timestart = CxoTime(timestart)
    timestop = CxoTime(timestop)
    planet_ids = {'mercury': '199',
                  'venus': '299',
                  'mars': '499',
                  'jupiter': '599',
                  'saturn': '699',
                  'uranus': '799',
                  'neptune': '899'}
    if body not in planet_ids:
        raise ValueError(f'body must be one of {tuple(planet_ids)}')

    params = dict(
        COMMAND=planet_ids[body],
        MAKE_EPHEM='YES',
        CENTER='@-151',
        TABLE_TYPE='OBSERVER',
        ANG_FORMAT='DEG',
        START_TIME=timestart.datetime.strftime('%Y-%b-%d %H:%M'),
        STOP_TIME=timestop.datetime.strftime('%Y-%b-%d %H:%M'),
        STEP_SIZE=str(n_times),
        QUANTITIES='1,3,9,13',
        CSV_FORMAT='YES')

    # The HORIZONS web API seems to require all params to be quoted strings.
    # See: https://ssd.jpl.nasa.gov/horizons_batch.cgi
    for key, val in params.items():
        params[key] = repr(val)
    params['batch'] = 1
    resp = requests.get(URL_HORIZONS, params=params, timeout=timeout)

    if resp.status_code != requests.codes['ok']:
        raise ValueError('request {resp.url} failed: {resp.reason} ({resp.status_code})')

    lines = resp.text.splitlines()
    idx0 = lines.index('$$SOE') + 1
    idx1 = lines.index('$$EOE')
    lines = lines[idx0: idx1]
    dat = ascii.read(lines, format='no_header', delimiter=',',
                     names=['time', 'null1', 'null2', 'ra', 'dec', 'rate_ra', 'rate_dec',
                            'mag', 'surf_brt', 'ang_diam', 'null3']
                     )

    times = [datetime.strptime(val[:20], '%Y-%b-%d %H:%M:%S') for val in dat['time']]
    dat['time'] = CxoTime(times, format='datetime')
    dat['time'].format = 'date'
    dat['ra'].info.unit = u.deg
    dat['dec'].info.unit = u.deg
    dat['rate_ra'].info.unit = u.arcsec / u.hr
    dat['rate_dec'].info.unit = u.arcsec / u.hr
    dat['mag'].info.unit = u.mag
    dat['surf_brt'].info.unit = u.mag / (u.arcsec**2)
    dat['ang_diam'].info.unit = u.arcsec

    dat['ra'].info.format = '.5f'
    dat['dec'].info.format = '.5f'
    dat['rate_ra'].info.format = '.2f'
    dat['rate_dec'].info.format = '.2f'
    dat['mag'].info.format = '.3f'
    dat['surf_brt'].info.format = '.3f'
    dat['ang_diam'].info.format = '.2f'

    dat.meta['response_text'] = resp.text

    del dat['null1']
    del dat['null2']
    del dat['null3']

    return dat
