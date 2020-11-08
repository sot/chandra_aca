# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

from chandra_aca.planets import (get_planet_chandra, get_planet_barycentric,
                                 get_planet_eci)
from chandra_aca.transform import eci_to_radec
from agasc import sphere_dist


def test_planet_positions():
    # Test basic functionality and include regression values (not an independent
    # functional test)
    bary = get_planet_barycentric('saturn', '2020:324:11:44:00')
    assert np.allclose(bary, [7.92469846e+08, -1.15786689e+09, -5.12388561e+08])

    eci = get_planet_eci('saturn', '2020:324:11:44:00')
    assert np.allclose(eci, [7.13565053e+08, -1.27291136e+09, -5.62273593e+08])

    eci = get_planet_chandra('saturn', '2020:324:11:44')
    assert np.allclose(eci, [7.13516810e+08, -1.27285190e+09, -5.62368753e+08])

    # Independent functional test to compare with JPL Horizons (Saturn from
    # Chandra Observatory -151)
    ra, dec = eci_to_radec(eci)
    ra2, dec2 = 299.27358333, -21.07644444

    assert sphere_dist(ra, dec, ra2, dec2) * 3600 < 1.0


def test_planet_positions_array():
    bary = get_planet_barycentric('saturn', ['2020:324:11:44:00', '2020:324:11:44:01'])
    assert bary.shape == (2, 3)

    eci = get_planet_eci('saturn', ['2020:324:11:44:00', '2020:324:11:44:01'])
    assert eci.shape == (2, 3)

    eci = get_planet_chandra('saturn', ['2020:324:11:44:00', '2020:324:11:44:01'])
    assert eci.shape == (2, 3)
    ra, dec = eci_to_radec(eci)

    # Value from JPL Horizons (Saturn from Chandra Observatory -151)
    ra2, dec2 = 299.27358333, -21.07644444

    assert np.all(sphere_dist(ra, dec, ra2, dec2) * 3600 < 1.0)
