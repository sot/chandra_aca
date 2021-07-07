# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
import pytest

from cxotime import CxoTime
from chandra_aca.planets import (get_planet_chandra, get_planet_barycentric,
                                 get_planet_chandra_horizons, get_planet_eci,
                                 get_planet_angular_sep)
from chandra_aca.transform import eci_to_radec, radec_to_yagzag
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


def test_venus_position1():
    """Obsid 18695 starcat at 2017:010:05:07:20.875, approx obs star 0510z"""

    # Output from JPL Horizons for Venus from Chandra
    date = CxoTime('2017-01-10T05:10')
    sc = SkyCoord('22:36:02.59', '-09:39:07.2', unit=(u.hr, u.deg))
    q_att = [-0.54137601, 0.17071483, -0.10344611, 0.81674192]

    eci = get_planet_chandra('venus', date)
    ra, dec = eci_to_radec(eci)
    yag, zag = radec_to_yagzag(ra, dec, q_att)
    # Confirm yag value is on "left side" of CCD, opposite all stars in 18695
    assert np.isclose(yag, 210.20, rtol=0, atol=0.01)
    assert np.isclose(zag, 69.45, rtol=0, atol=0.01)

    dist = sphere_dist(ra, dec, sc.ra.to_value(u.deg), sc.dec.to_value(u.deg)) * 3600
    assert np.all(dist < 4.0)


def test_venus_position2():
    # *******************************************************************************
    # Target body name: Venus (299)                     {source: CHANDRA_MERGED}
    # Center body name: Chandra Observatory (spacecraft) (-151) {source: CHANDRA_MERGED}
    # Center-site name: BODYCENTRIC
    # *******************************************************************************
    # Start time      : A.D. 2020-Jan-01 00:00:00.0000 UT
    # Stop  time      : A.D. 2020-Jun-01 00:00:00.0000 UT
    # Step-size       : 21600 minutes
    # *******************************************************************************

    txt = """
        date                   ra         dec
        2020-01-01T00:00     21:08:43.02 -18:22:41.8
        2020-01-16T00:00     22:19:56.31 -12:03:15.4
        2020-01-31T00:00     23:26:25.34 -04:40:18.3
        2020-02-15T00:00     00:29:55.07 +03:09:41.1
        2020-03-01T00:00     01:31:42.96 +10:46:02.6
        2020-03-16T00:00     02:32:52.02 +17:25:28.9
        2020-03-31T00:00     03:32:39.01 +22:40:58.7
        2020-04-15T00:00     04:26:52.03 +26:10:57.3
        2020-04-30T00:00     05:07:55.28 +27:38:30.8
        2020-05-15T00:00     05:22:08.73 +27:05:38.1
        2020-05-30T00:00     04:59:36.37 +24:14:26.9
    """
    dat = ascii.read(txt)
    date = CxoTime(dat['date'])
    sc = SkyCoord(dat['ra'], dat['dec'], unit=(u.hr, u.deg))

    eci = get_planet_chandra('venus', date)
    ra, dec = eci_to_radec(eci)

    dist = sphere_dist(ra, dec, sc.ra.to_value(u.deg), sc.dec.to_value(u.deg)) * 3600
    assert np.all(dist < 4.0)


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


def test_get_chandra_planet_horizons():
    dat = get_planet_chandra_horizons('jupiter', '2020:001', '2020:002', n_times=11)
    exp = ['         time             ra       dec     rate_ra    rate_dec   mag  '
           '    surf_brt   ang_diam',
           '                         deg       deg    arcsec / h arcsec / h  mag  '
           ' mag / arcsec2  arcsec ',
           '--------------------- --------- --------- ---------- ---------- ------'
           ' ------------- --------',
           '2020:001:00:00:00.000 276.96494 -23.20087      34.22       0.98 -1.839'
           '         5.408    31.75',
           '2020:001:02:24:00.000 276.98978 -23.20017      34.25       1.11 -1.839'
           '         5.408    31.75',
           '2020:001:04:48:00.000 277.01463 -23.19939      34.28       1.24 -1.839'
           '         5.408    31.76',
           '2020:001:07:12:00.000 277.03951 -23.19852      34.32       1.37 -1.839'
           '         5.408    31.76',
           '2020:001:09:36:00.000 277.06441 -23.19757      34.35       1.50 -1.839'
           '         5.408    31.76',
           '2020:001:12:00:00.000 277.08934 -23.19652      34.39       1.64 -1.839'
           '         5.408    31.76',
           '2020:001:14:24:00.000 277.11430 -23.19537      34.44       1.79 -1.839'
           '         5.408    31.76',
           '2020:001:16:48:00.000 277.13930 -23.19413      34.49       1.94 -1.839'
           '         5.408    31.76',
           '2020:001:19:12:00.000 277.16433 -23.19278      34.54       2.11 -1.839'
           '         5.408    31.76',
           '2020:001:21:36:00.000 277.18941 -23.19131      34.61       2.30 -1.839'
           '         5.408    31.76',
           '2020:002:00:00:00.000 277.21454 -23.18970      34.69       2.51 -1.839'
           '         5.408    31.76']

    assert dat.pformat_all() == exp


@pytest.mark.parametrize('obs_pos,exp_sep', [('chandra-horizons', 0.0),
                                             ('chandra', 0.74),
                                             ('earth', 23.02)])
def test_get_planet_ang_separation_scalar(obs_pos, exp_sep):
    # Position of Jupiter at time0
    time0 = '2021:001'
    ra0, dec0 = 304.89116, -20.08328
    sep = get_planet_angular_sep('jupiter', ra0, dec0, time0, observer_position=obs_pos)
    assert np.isclose(sep * 3600, exp_sep, atol=1e-2, rtol=0)


@pytest.mark.parametrize('obs_pos,exp_sep', [('chandra-horizons', [0.0, 33.98]),
                                             ('chandra', [0.74, 33.25]),
                                             ('earth', [23.02, 47.07])])
def test_get_planet_ang_separation_array(obs_pos, exp_sep):
    # Position of Jupiter at time0
    times = ['2021:001:00:00:00', '2021:001:01:00:00']
    ra0, dec0 = 304.89116, -20.08328
    sep = get_planet_angular_sep('jupiter', ra0, dec0, times, observer_position=obs_pos)
    assert np.allclose(sep * 3600, exp_sep, atol=1e-2, rtol=0)
