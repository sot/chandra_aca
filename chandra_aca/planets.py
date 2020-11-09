# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Functions for planet position relative to Chandra, Earth, or Solar System Barycenter.
"""
from pathlib import Path

import astropy.units as u
import numpy as np
from cxotime import CxoTime
from jplephem.spk import SPK
from ska_helpers.utils import LazyVal

__all__ = ['get_planet_chandra', 'get_planet_barycentric', 'get_planet_eci']


def load_kernel():
    kernel_path = Path(__file__).parent / 'data' / 'de432s.bsp'
    if not kernel_path.exists():
        raise FileNotFoundError(f'kernel data file {kernel_path} not found, '
                                'run "python setup.py --version" to install it locally')
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


def get_planet_barycentric(body, time=None):
    """Get barycentric position for solar system ``body`` at ``time``.

    :param body: Body name
    :param time: Time or times for returned position (default=NOW)
    :returns: barycentric position (km)
    """
    kernel = KERNEL.get()
    if body not in BODY_NAME_TO_KERNEL_SPEC:
        raise ValueError(f'{body} is not an allowed value of solar system body')

    spk_pairs = BODY_NAME_TO_KERNEL_SPEC[body]
    time = CxoTime(time)
    time_jd = time.jd
    pos = kernel[spk_pairs[0]].compute(time_jd)
    for spk_pair in spk_pairs[1:]:
        pos += kernel[spk_pair].compute(time_jd)

    return pos.transpose()  # SPK returns (3, N) but we need (N, 3)


def get_planet_eci(body, time=None):
    """Get ECI position for solar system ``body`` at ``time``.

    :param body: Body name
    :param time: Time or times for returned position (default=NOW)
    :returns: Earth-Centered Inertial (ECI) position (km)
    """
    time = CxoTime(time)

    pos_planet = get_planet_barycentric(body, time)
    pos_earth = get_planet_barycentric('earth', time)

    dist = np.sqrt(np.sum((pos_planet - pos_earth) ** 2, axis=-1)) * u.km
    import astropy.constants as const
    light_travel_time = (dist / const.c).to(u.s)

    pos_planet = get_planet_barycentric(body, time - light_travel_time)

    return pos_planet - pos_earth


def get_planet_chandra(body, time=None):
    """Get position for solar system ``body`` at ``time`` relative to Chandra.

    :param body: Body name
    :param time: Time or times for returned position (default=NOW)
    :returns: position relative to Chandra (km)
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
