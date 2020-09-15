# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import os

import pytest
import requests
import numpy as np
from astropy.io import ascii
from astropy.table import Table

from Quaternion import Quat
from Chandra.Time import DateTime

import chandra_aca
from chandra_aca.transform import (snr_mag_for_t_ccd, radec_to_yagzag,
                                   yagzag_to_radec, pixels_to_yagzag, yagzag_to_pixels,
                                   eci_to_radec, radec_to_eci)
from chandra_aca import drift

dirname = os.path.dirname(__file__)

TOLERANCE = 0.05

# SI_ALIGN matrix used from just after launch through NOV0215 (Nov 2015) loads.
SI_ALIGN_CLASSIC = np.array([[1.0, 3.3742E-4, 2.7344E-4],
                             [-3.3742E-4, 1.0, 0.0],
                             [-2.7344E-4, 0.0, 1.0]]).transpose()

# Use this for testing in case the icxc version is not available.
ZERO_OFFSET_TABLE = """
# Table of zero-offset aimpoints
# Last updated: 15 November 2018
date_effective  cycle_effective  detector  chipx   chipy   chip_id  obsvis_cal
2012-12-15      15               ACIS-I    941.0   988.0   3        1.6
2012-12-15      15               ACIS-S    224.0   490.0   7        1.6
2012-12-15      15               HRC-I     7615.0  7862.0  0        1.6
2012-12-15      15               HRC-S     2075.0  8984.0  2        1.6
2013-12-15      16               ACIS-I    932.0   1009.0  3        1.7
2013-12-15      16               ACIS-S    205.0   480.0   7        1.7
2013-12-15      16               HRC-I     7606.0  7941.0  0        1.7
2013-12-15      16               HRC-S     2050.0  9023.0  2        1.7
2014-12-15      17               ACIS-I    930.2   1009.6  3        1.8a
2014-12-15      17               ACIS-S    200.7   476.9   7        1.8a
2014-12-15      17               HRC-I     7591.0  7936.1  0        1.8a
2014-12-15      17               HRC-S     2041.0  9062.7  2        1.8a
2015-12-15      18               ACIS-I    970.0   975.0   3        1.9
2015-12-15      18               ACIS-S    210.0   520.0   7        1.9
2015-12-15      18               HRC-I     7590.0  7745.0  0        1.9
2015-12-15      18               HRC-S     2195.0  8915.0  2        1.9
2016-12-15      19               ACIS-I    970.0   975.0   3        1.10
2016-12-15      19               ACIS-S    210.0   520.0   7        1.10
2016-12-15      19               HRC-I     7590.0  7745.0  0        1.10
2016-12-15      19               HRC-S     2195.0  8915.0  2        1.10
2017-12-15      20               ACIS-I    970.0   975.0   3        1.11
2017-12-15      20               ACIS-S    210.0   520.0   7        1.11
2017-12-15      20               HRC-I     7590.0  7745.0  0        1.11
2017-12-15      20               HRC-S     2195.0  8915.0  2        1.11
2018-11-07      17               ACIS-S    182.3   468.1   7        1.8a
2018-11-07      18               ACIS-S    191.6   511.2   7        1.9
2018-11-07      19               ACIS-S    191.6   511.2   7        1.10
2018-11-07      20               ACIS-S    191.6   511.2   7        1.11
2018-11-16      17               ACIS-S    200.7   476.9   7        1.8a
2018-11-16      18               ACIS-S    210.0   520.0   7        1.9
2018-11-16      19               ACIS-S    210.0   520.0   7        1.10
2018-11-16      20               ACIS-S    210.0   520.0   7        1.11
"""


def test_edge_checking():
    """Test row/col edge checking"""

    # Within limits, doesn't fail
    yag, zag = pixels_to_yagzag(511.7, -511.7)
    yagzag_to_pixels(yag, zag)

    with pytest.raises(ValueError):
        pixels_to_yagzag(512.2, 0)

    with pytest.raises(ValueError):
        pixels_to_yagzag(0, -512.2)

    yag, zag = pixels_to_yagzag(512.2, -512.2, allow_bad=True)
    with pytest.raises(ValueError):
        yagzag_to_pixels(yag, zag)


def test_pix_to_angle():
    pix_to_angle = ascii.read(os.path.join(dirname, 'data', 'pix_to_angle.txt'))

    print("testing {} row/col pairs match to {} arcsec".format(
        len(pix_to_angle), TOLERANCE))
    pyyang, pyzang = chandra_aca.pixels_to_yagzag(
        pix_to_angle['row'],
        pix_to_angle['col'])
    np.testing.assert_allclose(pix_to_angle['yang'], pyyang, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(pix_to_angle['zang'], pyzang, atol=TOLERANCE, rtol=0)


def test_pix_to_angle_flight():
    """
    This is a minimal regression test.  For actual validation that the
    values are correct see aca_track/predict_track.ipynb notebook.
    """
    yag, zag = chandra_aca.pixels_to_yagzag(100., 100., flight=True, t_aca=24.0)
    assert np.allclose([yag, zag], [-467.7295724, 475.3100623])

    yag, zag = chandra_aca.pixels_to_yagzag(100., 100., flight=True, t_aca=14.0)
    assert np.allclose([yag, zag], [-467.8793858, 475.4463912])


def test_angle_to_pix():
    angle_to_pix = ascii.read(os.path.join(dirname, 'data', 'angle_to_pix.txt'))
    print("testing {} yang/zang pairs match to {} pixels".format(
        len(angle_to_pix), TOLERANCE))
    pyrow, pycol = chandra_aca.yagzag_to_pixels(
        angle_to_pix['yang'],
        angle_to_pix['zang'])
    np.testing.assert_allclose(angle_to_pix['row'], pyrow, atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(angle_to_pix['col'], pycol, atol=TOLERANCE, rtol=0)


def test_angle_to_pix_types():
    for ftype in [int, float, np.int32, np.int64, np.float32]:
        pyrow, pycol = chandra_aca.yagzag_to_pixels(ftype(2540), ftype(1660))
        assert np.isclose(pyrow, -506.71, rtol=0, atol=0.01)
        assert np.isclose(pycol, 341.19, rtol=0, atol=0.01)


def test_pix_zero_loc():
    r, c = 100, 200
    ye, ze = chandra_aca.pixels_to_yagzag(r, c, pix_zero_loc='edge')
    yc, zc = chandra_aca.pixels_to_yagzag(r, c, pix_zero_loc='center')

    # Different by about 2.5 arcsec for sanity check
    assert np.isclose(abs(ye - yc), 2.5, rtol=0, atol=0.02)
    assert np.isclose(abs(ze - zc), 2.5, rtol=0, atol=0.02)

    # Round trips r,c => y,z => r,c
    re, ce = chandra_aca.yagzag_to_pixels(ye, ze, pix_zero_loc='edge')
    rc, cc = chandra_aca.yagzag_to_pixels(yc, zc, pix_zero_loc='center')

    # Interestingly these transforms do not round trip more accurately
    # than about 0.01 pixels.
    assert np.isclose(re - r, 0, rtol=0, atol=0.01)
    assert np.isclose(rc - r, 0, rtol=0, atol=0.01)
    assert np.isclose(ce - c, 0, rtol=0, atol=0.01)
    assert np.isclose(cc - c, 0, rtol=0, atol=0.01)


@pytest.mark.parametrize('func', [chandra_aca.pixels_to_yagzag,
                                  chandra_aca.yagzag_to_pixels])
def test_transform_broadcast(func):
    rows = [-100, 100]
    cols = [-200, 200]

    # Transform 2-d array
    r, c = np.meshgrid(rows, cols)
    y, z = func(r, c)

    # Transform flattened version of 2-d array
    yf, zf = func(r.flatten(), c.flatten())

    # Flattened output must be same as transform of flattened input
    assert np.all(y.flatten() == yf)
    assert np.all(z.flatten() == zf)

    # Make sure scalars result in scalars
    y, z = func(100, 200)
    assert y.shape == ()
    assert z.shape == ()
    assert isinstance(y, np.float64)
    assert isinstance(z, np.float64)
    assert not isinstance(y, np.ndarray)
    assert not isinstance(z, np.ndarray)


def test_radec_to_yagzag():
    ra = 0.5
    dec = 0.75
    q_att = Quat([0, 0, 0])
    yag, zag = radec_to_yagzag(ra, dec, q_att)
    assert np.allclose([yag, zag], [1800.00, 2700.10], rtol=0, atol=0.01)


def test_yagzag_to_radec():
    yag = 1800.00
    zag = 2700.10
    q_att = Quat([0, 0, 0])
    ra, dec = yagzag_to_radec(yag, zag, q_att)
    assert np.allclose([ra, dec], [0.50, 0.75], rtol=0, atol=0.00001)


def test_aca_targ_transforms():
    """
    Observation request:
     ID=13928,TARGET=(191.321250,27.125556,{Haro 9}),DURATION=(17000.000000),
     PRIORITY=9,SI=ACIS-S,GRATING=NONE,SI_MODE=TE_0045A,ACA_MODE=DEFAULT,
     TARGET_OFFSET=(0.002500,-0.004167),
     DITHER=(ON,0.002222,0.360000,0.000000,0.002222,0.509100,0.000000),
     SEGMENT=(1,15300.000000),PRECEDING=(13632),MIN_ACQ=1,MIN_GUIDE=1

    ACA (PCAD):  As-planned pointing from starcheck
      Q1,Q2,Q3,Q4: -0.18142595  -0.37811633  -0.89077416  0.17502588
    """
    # Attitude quaternion for the as-run PCAD attitude
    q_aca = Quat([-0.18142595, -0.37811633, -0.89077416, 0.17502588])

    # Target coordinates and quaternion, using the PCAD roll
    ra_targ, dec_targ = 191.321250, 27.125556

    # Offsets from OR (Target DY, DZ) in degrees
    y_off, z_off = 0.002500, -0.004167

    q_targ = chandra_aca.calc_targ_from_aca(q_aca, y_off, z_off)

    assert np.allclose(ra_targ, q_targ.ra, atol=1e-5, rtol=0)
    assert np.allclose(dec_targ, q_targ.dec, atol=1e-5, rtol=0)

    q_aca_rt = chandra_aca.calc_aca_from_targ(q_targ, y_off, z_off)
    dq = q_aca_rt.inv() * q_aca
    assert np.degrees(np.abs(dq.q[0] * 2)) < 30 / 3600.
    assert np.degrees(np.abs(dq.q[1] * 2)) < 1 / 3600.
    assert np.degrees(np.abs(dq.q[2] * 2)) < 1 / 3600.


def test_get_aimpoint():
    obstests = [('2016-08-22', 15, 'ACIS-S'),
                ('2014-08-22', 16, 'HRC-I', True),
                ('2017-09-01', 18, 'ACIS-I')]
    answers = [(224.0, 490.0, 7),
               (7606.0, 7941.0, 0),
               (970.0, 975.0, 3)]
    try:
        r = requests.head(
            "https://icxc.harvard.edu/mp/html/aimpoint_table/zero_offset_aimpoints.txt",
            timeout=5)
        assert r.status_code == 200
        zot = None
    except Exception:
        zot = Table.read(ZERO_OFFSET_TABLE, format='ascii')

    for obstest, answer in zip(obstests, answers):
        chipx, chipy, chip_id = drift.get_target_aimpoint(*obstest, zero_offset_table=zot)
        assert chipx == answer[0]
        assert chipy == answer[1]
        assert chip_id == answer[2]
    zot = Table.read("""date_effective  cycle_effective  detector  chipx   chipy   chip_id  obsvis_cal
2012-12-15      15               ACIS-I    888   999   -1        1.6""", format='ascii')
    chipx, chipy, chip_id = drift.get_target_aimpoint(
        '2016-08-22', 15, 'ACIS-I', zero_offset_table=zot)
    assert chipx == 888
    assert chipy == 999
    assert chip_id == -1


def simple_test_aca_drift():
    """
    Qualitatively test the implementation of drift model by plotting (outside
    of this function) the returned drift values and comparing with plots in
    https://github.com/sot/aimpoint_mon/blob/master/fit_aimpoint_drift-2018-11.ipynb

    Match: YES.
    """
    times = DateTime(np.arange(2013.0, 2019.0, 0.01), format='frac_year').secs
    t_ccd = -10.0 * np.ones_like(times)  # degC
    dy = drift.DRIFT_Y.calc(times, t_ccd)
    dz = drift.DRIFT_Z.calc(times, t_ccd)

    return dy, dz, times


def test_get_aca_offsets():
    """
    Test that ACA offsets are reasonable, and regression test particular values
    corresponding to cycle 17 zero-offset aimpoints used below for chip_x, chip_y inputs.

    The output reference values here have been validated as being "reasonable" for the
    given inputs.
    """
    offsets = drift.get_aca_offsets('ACIS-I', 3, 930.2, 1009.6, '2016:180:12:00:00', -15.0)
    assert np.allclose(offsets, (11.45, 2.34), atol=0.1, rtol=0)

    offsets = drift.get_aca_offsets('ACIS-S', 7, 200.7, 476.9, '2016:180:12:00:00', -15.0)
    assert np.allclose(offsets, (12.98, 3.52), atol=0.1, rtol=0)

    offsets = drift.get_aca_offsets('HRC-I', 0, 7591, 7936, '2016:180:12:00:00', -15.0)
    assert np.allclose(offsets, (14.35, 0.45), atol=0.1, rtol=0)

    offsets = drift.get_aca_offsets('HRC-S', 2, 2041, 9062, '2016:180:12:00:00', -15.0)
    assert np.allclose(offsets, (16.89, 3.10), atol=0.1, rtol=0)


def test_snr_mag():
    same = snr_mag_for_t_ccd(-11.5, ref_mag=10.3, ref_t_ccd=-11.5, scale_4c=5)
    assert np.isclose(same, 10.3, atol=0.0001, rtol=0)
    # Show a fewdifferent combinations of results based on different values
    # for ref_mag and ref_t_ccd
    arr = snr_mag_for_t_ccd(np.array([-11.5, -10, -5]),
                            ref_mag=10.3, ref_t_ccd=-11.5, scale_4c=1.59)
    assert np.allclose(arr, [10.3, 10.1112, 9.4818], atol=0.0001, rtol=0)
    arr = snr_mag_for_t_ccd(np.array([-11.5, -10, -5]), ref_mag=9.0, ref_t_ccd=-11.5, scale_4c=1.59)
    assert np.allclose(arr, [9.0, 8.8112, 8.1818], atol=0.0001, rtol=0)
    arr = snr_mag_for_t_ccd(np.array([-11.5, -10, -5]), ref_mag=9.0, ref_t_ccd=-9.5, scale_4c=1.59)
    assert np.allclose(arr, [9.2517, 9.0630, 8.4336], atol=0.0001, rtol=0)


def test_eci_to_radec():
    eci = np.array([0.92541658, 0.16317591, 0.34202014])
    ra, dec = eci_to_radec(eci)
    assert np.allclose(ra, 9.9999999129952908)
    assert np.allclose(dec, 19.999999794004037)
    assert isinstance(ra, np.float64)
    assert isinstance(dec, np.float64)


def test_vectorized_eci_to_radec():
    eci = np.array([[0.92541658, 0.16317591, 0.34202014],
                    [0.9248273, -0.16307201, 0.34365969]])
    ra, dec = eci_to_radec(eci)
    assert np.allclose(ra[0], 9.9999999129952908)
    assert np.allclose(ra[1], 349.9999997287627)
    assert np.allclose(dec[0], 19.999999794004037)
    assert np.allclose(dec[1], 20.099999743270516)
    assert isinstance(ra, np.ndarray)
    assert isinstance(dec, np.ndarray)


def test_radec_to_eci():
    eci = radec_to_eci(10, 20)
    assert np.allclose(eci[0], 0.92541658)
    assert np.allclose(eci[1], 0.16317591)
    assert np.allclose(eci[2], 0.34202014)
    assert isinstance(eci, np.ndarray)


@pytest.mark.parametrize('shape', [(24,), (6, 4), (3, 4, 2)])
def test_radec_eci_multidim(shape):
    """Test radec_to_eci and eci_to_radec for multidimensional inputs"""
    ras = np.linspace(0., 359., 24)
    decs = np.linspace(-90., 90., 24)
    ras_nd = ras.reshape(shape)
    decs_nd = decs.reshape(shape)

    # First do everything as scalars
    ecis_list = [radec_to_eci(ra, dec) for ra, dec in zip(ras.tolist(), decs.tolist())]
    ecis_nd_from_list = np.array(ecis_list).reshape(shape + (3,))

    ecis = radec_to_eci(ras_nd, decs_nd)
    assert ecis.shape == shape + (3,)
    assert np.allclose(ecis, ecis_nd_from_list)

    ras_rt, decs_rt = eci_to_radec(ecis)
    assert ras_rt.shape == shape
    assert decs_rt.shape == shape
    assert np.allclose(ras_rt, ras_nd)
    assert np.allclose(decs_rt, decs_nd)


@pytest.mark.parametrize('shape', [(24,), (6, 4), (3, 4, 2)])
def test_radec_yagzag_multidim(shape):
    """Test radec_to_yagzag and yagzag_to_radec for multidimensional inputs"""
    dec = 0.5
    ras = np.linspace(0., 1., 24)
    decs = dec  # Test broadcasting with a scalar input
    eqs = np.empty(shape=(24, 3))
    eqs[..., 0] = 0.1
    eqs[..., 1] = np.linspace(-0.1, 0.1, 24)
    eqs[..., 2] = np.linspace(-1, 1, 24)

    ras_nd = ras.reshape(shape)
    qs_nd = Quat(equatorial=eqs.reshape(shape + (3,)))

    # First do everything as scalars
    yagzags_list = [radec_to_yagzag(ra, dec, Quat(eq)) for ra, eq in zip(ras.tolist(), eqs.tolist())]
    yagzags_nd_from_list = np.array(yagzags_list).reshape(shape + (2,))

    yagzags = radec_to_eci(ras_nd, decs, qs_nd)
    assert yagzags.shape == shape + (3,)
    assert np.allclose(yagzags, yagzags_nd_from_list)

#    ras_rt, decs_rt = eci_to_radec(ecis)
#    assert ras_rt.shape == shape
#    assert decs_rt.shape == shape
#    assert np.allclose(ras_rt, ras_nd)
#    assert np.allclose(decs_rt, decs_nd)