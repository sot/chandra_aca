import numpy as np
from astropy.io import ascii

from Quaternion import Quat

import chandra_aca
from chandra_aca.star_probs import t_ccd_warm_limit, mag_for_p_acq

TOLERANCE = 0.05

# SI_ALIGN matrix used from just after launch through NOV0215 (Nov 2015) loads.
SI_ALIGN_CLASSIC = np.array([[1.0, 3.3742E-4, 2.7344E-4],
                             [-3.3742E-4, 1.0, 0.0],
                             [-2.7344E-4, 0.0, 1.0]]).transpose()


def test_pix_to_angle():
    pix_to_angle = ascii.read(open('test_data/pix_to_angle.txt'))

    print "testing {} row/col pairs match to {} arcsec".format(
        len(pix_to_angle), TOLERANCE)
    pyyang, pyzang = chandra_aca.pixels_to_yagzag(
        pix_to_angle['row'],
        pix_to_angle['col'])
    np.testing.assert_allclose(pix_to_angle['yang'], pyyang, atol=TOLERANCE)
    np.testing.assert_allclose(pix_to_angle['zang'], pyzang, atol=TOLERANCE)


def test_angle_to_pix():
    angle_to_pix = ascii.read(open('test_data/angle_to_pix.txt'))
    print "testing {} yang/zang pairs match to {} pixels".format(
        len(angle_to_pix), TOLERANCE)
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

    q_targ = chandra_aca.calc_targ_from_aca(q_aca, y_off, z_off, SI_ALIGN_CLASSIC)

    assert np.allclose(ra_targ, q_targ.ra, atol=1e-5, rtol=0)
    assert np.allclose(dec_targ, q_targ.dec, atol=1e-5, rtol=0)

    q_aca_rt = chandra_aca.calc_aca_from_targ(q_targ, y_off, z_off, SI_ALIGN_CLASSIC)
    dq = q_aca_rt.inv() * q_aca
    assert np.degrees(np.abs(dq.q[0] * 2)) < 30 / 3600.
    assert np.degrees(np.abs(dq.q[1] * 2)) < 1 / 3600.
    assert np.degrees(np.abs(dq.q[2] * 2)) < 1 / 3600.


def test_t_ccd_warm_limit():
    out = t_ccd_warm_limit([9.8] * 6, date='2015:001', min_n_acq=(2, 8e-3))
    assert np.allclose(out[0], -13.3341, atol=0.01)
    assert np.allclose(out[1], 0.008, atol=0.0001)

    out = t_ccd_warm_limit([9.8] * 6, date='2015:001', min_n_acq=5.0)
    assert np.allclose(out[0], -13.2155, atol=0.01)
    assert np.allclose(out[1], 5.0, atol=0.01)


def test_mag_for_p_acq():
    mag = mag_for_p_acq(0.50, date='2015:001', t_ccd=-14.0)
    assert np.allclose(mag, 10.282, rtol=0, atol=0.01)
