# Licensed under a 3-clause BSD style license - see LICENSE.rst

from chandra_aca.planets import get_planet_chandra
from chandra_aca.transform import eci_to_radec
from agasc import sphere_dist


def test_planet_positions():
    eci = get_planet_chandra('saturn', '2020:324:11:44')
    ra, dec = eci_to_radec(eci)

    # Value from JPL Horizons (Saturn from Chandra Observatory -151)
    ra2, dec2 = 299.27358333, -21.07644444

    assert sphere_dist(ra, dec, ra2, dec2) * 3600 < 1.0
