from __future__ import print_function, division

import os

import tempfile
import numpy as np
from astropy.io import ascii
from astropy.table import Table

from Quaternion import Quat
from Chandra.Time import DateTime

import chandra_aca
from chandra_aca.plot import plot_stars, plot_compass
from chandra_aca.star_probs import t_ccd_warm_limit, mag_for_p_acq
from chandra_aca import drift

dirname = os.path.dirname(__file__)

TOLERANCE = 0.05

# SI_ALIGN matrix used from just after launch through NOV0215 (Nov 2015) loads.
SI_ALIGN_CLASSIC = np.array([[1.0, 3.3742E-4, 2.7344E-4],
                             [-3.3742E-4, 1.0, 0.0],
                             [-2.7344E-4, 0.0, 1.0]]).transpose()


def test_pix_to_angle():
    pix_to_angle = ascii.read(open(os.path.join(dirname, 'data', 'pix_to_angle.txt')))

    print("testing {} row/col pairs match to {} arcsec".format(
        len(pix_to_angle), TOLERANCE))
    pyyang, pyzang = chandra_aca.pixels_to_yagzag(
        pix_to_angle['row'],
        pix_to_angle['col'])
    np.testing.assert_allclose(pix_to_angle['yang'], pyyang, atol=TOLERANCE)
    np.testing.assert_allclose(pix_to_angle['zang'], pyzang, atol=TOLERANCE)


def test_angle_to_pix():
    angle_to_pix = ascii.read(open(os.path.join(dirname, 'data', 'angle_to_pix.txt')))
    print("testing {} yang/zang pairs match to {} pixels".format(
        len(angle_to_pix), TOLERANCE))
    pyrow, pycol = chandra_aca.yagzag_to_pixels(
        angle_to_pix['yang'],
        angle_to_pix['zang'])
    np.testing.assert_allclose(angle_to_pix['row'], pyrow, atol=TOLERANCE)
    np.testing.assert_allclose(angle_to_pix['col'], pycol, atol=TOLERANCE)


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


def test_t_ccd_warm_limit():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=(2, 8e-3))
    assert np.allclose(out[0], -15.2449, atol=0.01)
    assert np.allclose(out[1], 0.008, atol=0.0001)

    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=5.0)
    assert np.allclose(out[0], -15.143, atol=0.01)
    assert np.allclose(out[1], 5.0, atol=0.01)


def test_mag_for_p_acq():
    mag = mag_for_p_acq(0.50, date='2015:001', t_ccd=-14.0)
    assert np.allclose(mag, 10.821, rtol=0, atol=0.01)


def test_get_aimpoint():
    obstests = [('2016-08-22', 15, 'ACIS-S'),
               ('2014-08-22', 16, 'HRC-I', True),
               ('2017-09-01', 18, 'ACIS-I')]
    answers = [(224.0, 490.0, 7),
               (7606.0, 7941.0, 0),
               (970.0, 975.0, 3)]
    for obstest, answer in zip(obstests, answers):
        chipx, chipy, chip_id = drift.get_target_aimpoint(*obstest)
        assert chipx == answer[0]
        assert chipy == answer[1]
        assert chip_id == answer[2]
    zot = Table.read("""date_effective  cycle_effective  detector  chipx   chipy   chip_id  obsvis_cal
2012-12-15      15               ACIS-I    888   999   -1        1.6""", format='ascii')
    chipx, chipy, chip_id = drift.get_target_aimpoint('2016-08-22', 15, 'ACIS-I', zero_offset_table=zot)
    assert chipx == 888
    assert chipy == 999
    assert chip_id == -1



def simple_test_aca_drift():
    """
    Qualitatively test the implementation of drift model by plotting (outside
    of this function) the returned drift values and comparing with plots in
    https://github.com/sot/aimpoint_mon/blob/master/fit_aimpoint_drift.ipynb

    Match: YES.
    """
    times = DateTime(np.arange(2013.0, 2016.5, 0.01), format='frac_year').secs
    t_ccd = -13.8888889 * np.ones_like(times)  # degC, equivalent to +7.0 degC
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
    offsets = drift.get_aca_offsets('ACIS-I', 3, 930.2, 1009.6, '2016:180', -15.0)
    assert np.allclose(offsets, (11.83637884563926, 2.6860740140775334), atol=0.1)

    offsets = drift.get_aca_offsets('ACIS-S', 7, 200.7, 476.9, '2016:180', -15.0)
    assert np.allclose(offsets, (13.360706170615167, 3.8670874955935481), atol=0.1)

    offsets = drift.get_aca_offsets('HRC-I', 0, 7591, 7936, '2016:180', -15.0)
    assert np.allclose(offsets, (14.728718419826098, 0.7925650626134555), atol=0.1)

    offsets = drift.get_aca_offsets('HRC-S', 2, 2041, 9062, '2016:180', -15.0)
    assert np.allclose(offsets, (17.269560057119545, 3.4474216529603225), atol=0.1)


def test_plot():
    fig = plot_stars(attitude=(10, 20, 30), catalog=None)
    savefile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(savefile.name)
    import imghdr
    assert imghdr.what(savefile.name) == 'png'
    os.unlink(savefile.name)
