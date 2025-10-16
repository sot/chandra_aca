# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest
from agasc import sphere_dist
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from cxotime import CxoTime
from testr.test_helper import has_internet

from chandra_aca.planets import (
    convert_time_format_spk,
    get_earth_blocks,
    get_planet_angular_sep,
    get_planet_barycentric,
    get_planet_chandra,
    get_planet_chandra_horizons,
    get_planet_eci,
)
from chandra_aca.transform import eci_to_radec, radec_to_yagzag

HAS_INTERNET = has_internet()


@pytest.mark.skipif(not HAS_INTERNET, reason="Requires network access")
def test_planet_positions():
    # Test basic functionality and include regression values (not an independent
    # functional test)
    bary = get_planet_barycentric("saturn", "2020:324:11:44:00")
    assert np.allclose(bary, [7.92469846e08, -1.15786689e09, -5.12388561e08])

    eci = get_planet_eci("saturn", "2020:324:11:44:00")
    assert np.allclose(eci, [7.13565053e08, -1.27291136e09, -5.62273593e08])

    eci = get_planet_chandra("saturn", "2020:324:11:44")
    assert np.allclose(eci, [7.13516810e08, -1.27285190e09, -5.62368753e08])

    eci = get_planet_chandra("saturn", "2020:324:11:44", ephem_source="stk")
    assert np.allclose(eci, [7.13516810e08, -1.27285190e09, -5.62368753e08])

    # Independent functional test to compare with JPL Horizons (Saturn from
    # Chandra Observatory -151)
    ra, dec = eci_to_radec(eci)
    ra2, dec2 = 299.27358333, -21.07644444

    assert sphere_dist(ra, dec, ra2, dec2) * 3600 < 1.0


@pytest.mark.parametrize("ephem_source", ["cheta", "stk"])
def test_venus_position1(ephem_source):
    """Obsid 18695 starcat at 2017:010:05:07:20.875, approx obs star 0510z"""

    # Output from JPL Horizons for Venus from Chandra
    date = CxoTime("2017-01-10T05:10")
    sc = SkyCoord("22:36:02.59", "-09:39:07.2", unit=(u.hr, u.deg))
    q_att = [-0.54137601, 0.17071483, -0.10344611, 0.81674192]

    eci = get_planet_chandra("venus", date, ephem_source=ephem_source)
    ra, dec = eci_to_radec(eci)
    yag, zag = radec_to_yagzag(ra, dec, q_att)
    # Confirm yag value is on "left side" of CCD, opposite all stars in 18695
    assert np.isclose(yag, 209.67, rtol=0, atol=0.03)
    assert np.isclose(zag, 72.48, rtol=0, atol=0.01)

    dist = sphere_dist(ra, dec, sc.ra.to_value(u.deg), sc.dec.to_value(u.deg)) * 3600
    assert np.all(dist < 0.2)


@pytest.mark.parametrize("ephem_source", ["cheta", "stk"])
def test_venus_position2(ephem_source):
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
    date = CxoTime(dat["date"])
    sc = SkyCoord(dat["ra"], dat["dec"], unit=(u.hr, u.deg))

    eci = get_planet_chandra("venus", date, ephem_source=ephem_source)
    ra, dec = eci_to_radec(eci)

    dist = sphere_dist(ra, dec, sc.ra.to_value(u.deg), sc.dec.to_value(u.deg)) * 3600
    assert np.all(dist < 4.0)


def test_planet_positions_array():
    bary = get_planet_barycentric("saturn", ["2020:324:11:44:00", "2020:324:11:44:01"])
    assert bary.shape == (2, 3)

    eci = get_planet_eci("saturn", ["2020:324:11:44:00", "2020:324:11:44:01"])
    assert eci.shape == (2, 3)

    eci = get_planet_chandra("saturn", ["2020:324:11:44:00", "2020:324:11:44:01"])
    assert eci.shape == (2, 3)
    ra, dec = eci_to_radec(eci)

    # Value from JPL Horizons (Saturn from Chandra Observatory -151)
    ra2, dec2 = 299.27358333, -21.07644444

    assert np.all(sphere_dist(ra, dec, ra2, dec2) * 3600 < 1.0)


@pytest.mark.skipif(not HAS_INTERNET, reason="Requires network access")
@pytest.mark.parametrize("body", ["jupiter", 599])
def test_get_chandra_planet_horizons(body):
    dat = get_planet_chandra_horizons(body, "2020:001", "2020:002", n_times=11)
    exp = [
        "         time             ra       dec     rate_ra    rate_dec   mag  "
        "    surf_brt   ang_diam",
        "                         deg       deg    arcsec / h arcsec / h  mag  "
        " mag / arcsec2  arcsec ",
        "--------------------- --------- --------- ---------- ---------- ------"
        " ------------- --------",
        "2020:001:00:00:00.000 276.96494 -23.20087      34.22       0.98 -1.839"
        "         5.408    31.75",
        "2020:001:02:24:00.000 276.98978 -23.20017      34.25       1.11 -1.839"
        "         5.408    31.75",
        "2020:001:04:48:00.000 277.01463 -23.19939      34.28       1.24 -1.839"
        "         5.408    31.76",
        "2020:001:07:12:00.000 277.03951 -23.19852      34.32       1.37 -1.839"
        "         5.408    31.76",
        "2020:001:09:36:00.000 277.06441 -23.19757      34.35       1.50 -1.839"
        "         5.408    31.76",
        "2020:001:12:00:00.000 277.08934 -23.19652      34.39       1.64 -1.839"
        "         5.408    31.76",
        "2020:001:14:24:00.000 277.11430 -23.19537      34.44       1.79 -1.839"
        "         5.408    31.76",
        "2020:001:16:48:00.000 277.13930 -23.19413      34.49       1.94 -1.839"
        "         5.408    31.76",
        "2020:001:19:12:00.000 277.16433 -23.19278      34.54       2.11 -1.839"
        "         5.408    31.76",
        "2020:001:21:36:00.000 277.18941 -23.19131      34.61       2.30 -1.839"
        "         5.408    31.76",
        "2020:002:00:00:00.000 277.21454 -23.18970      34.69       2.51 -1.839"
        "         5.408    31.76",
    ]

    assert dat.pformat() == exp


@pytest.mark.skipif(not HAS_INTERNET, reason="Requires network access")
def test_get_chandra_planet_horizons_non_planet():
    dat = get_planet_chandra_horizons(
        "ACE (spacecraft)", "2020:001", "2020:002", n_times=2
    )
    exp = [
        "    ra       dec     rate_ra    rate_dec  mag    surf_brt   ang_diam",
        "   deg       deg    arcsec / h arcsec / h mag mag / arcsec2  arcsec ",
        "--------- --------- ---------- ---------- --- ------------- --------",
        "274.14524 -24.72559      10.75    -325.04  --            --       --",
        "275.29375 -23.83915     349.43     659.97  --            --       --",
    ]
    del dat["time"]
    assert dat.pformat() == exp


@pytest.mark.skipif(not HAS_INTERNET, reason="Requires network access")
@pytest.mark.parametrize(
    "obs_pos,exp_sep", [("chandra-horizons", 0.0), ("chandra", 0.09), ("earth", 23.25)]
)
def test_get_planet_ang_separation_scalar(obs_pos, exp_sep):
    # Position of Jupiter at time0
    time0 = "2021:001"
    ra0, dec0 = 304.89116, -20.08328
    sep = get_planet_angular_sep("jupiter", ra0, dec0, time0, observer_position=obs_pos)
    assert np.isclose(sep * 3600, exp_sep, atol=1e-2, rtol=0)


@pytest.mark.skipif(not HAS_INTERNET, reason="Requires network access")
@pytest.mark.parametrize(
    "obs_pos,exp_sep",
    [
        ("chandra-horizons", [0.0, 33.98]),
        ("chandra", [0.09, 33.90]),
        ("earth", [23.25, 47.66]),
    ],
)
def test_get_planet_ang_separation_array(obs_pos, exp_sep):
    # Position of Jupiter at time0
    times = ["2021:001:00:00:00", "2021:001:01:00:00"]
    ra0, dec0 = 304.89116, -20.08328
    sep = get_planet_angular_sep("jupiter", ra0, dec0, times, observer_position=obs_pos)
    assert np.allclose(sep * 3600, exp_sep, atol=1e-2, rtol=0)


def test_convert_time_format_spk_none():
    """Test bug fix where convert_time_format_spk failed when time was None"""
    time0 = convert_time_format_spk(None, "secs")
    time1 = CxoTime(None).secs
    # Times within 10 seconds
    assert np.isclose(time0, time1, atol=10, rtol=0)


def test_earth_boresight():
    """Find Earth blocks in 2023:290:00:00:00 to 2023:310:00:00:00.

    This calls get_earth_boresight_angle() so that function is implicitly tested.
    """
    start = "2023:290"
    stop = "2023:310"

    # The two long blocks are in perigee and correspond to Earth blocks seen in ACA
    # image data and manually excluded from monitor window processing:
    # EARTH_BLOCKS = [
    #   ("2023:297:12:23:00", "2023:297:12:48:37"),
    #   ("2023:300:03:49:00", "2023:300:04:16:40"),
    # ]
    exp = [
        "      datestart              datestop       duration",
        "--------------------- --------------------- --------",
        "2023:297:12:21:06.304 2023:297:12:49:59.579 1733.275",  # perigee
        "2023:298:20:54:50.436 2023:298:20:59:39.486  289.050",
        "2023:300:03:47:33.744 2023:300:04:16:31.119 1737.375",  # perigee
        "2023:300:17:11:19.997 2023:300:17:17:38.222  378.225",
        "2023:301:00:38:35.523 2023:301:00:43:43.023  307.500",
        "2023:305:21:35:24.050 2023:305:21:40:39.750  315.700",
        "2023:307:06:10:18.908 2023:307:06:15:29.483  310.575",
        "2023:309:03:33:42.369 2023:309:03:39:28.819  346.450",
        "2023:309:11:26:53.845 2023:309:11:31:18.295  264.450",
    ]

    blocks = get_earth_blocks(start, stop, min_limb_angle=10.0)
    assert blocks["datestart", "datestop", "duration"].pformat() == exp
