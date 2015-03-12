import numpy as np
from astropy.io import ascii

import chandra_aca

TOLERANCE = 0.05


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

