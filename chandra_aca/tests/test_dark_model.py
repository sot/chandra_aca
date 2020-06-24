# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import pytest
import numpy as np
from ..dark_model import dark_temp_scale, synthetic_dark_image, get_warm_fracs
from mica.common import MICA_ARCHIVE

HAS_MICA = os.path.exists(MICA_ARCHIVE)


def test_get_warm_fracs():
    exp = {(100, '2020:001:12:00:00', -11): 294302,
           (100, '2017:001:12:00:00', -11): 251865,
           (100, '2020:001:12:00:00', -15): 215096,
           (100, '2017:001:12:00:00', -15): 182365,
           (1000, '2017:001:12:00:00', -11): 1296,
           (1000, '2017:001:12:00:00', -15): 433,
           (1000, '2020:001:12:00:00', -15): 500,
           (1000, '2020:001:12:00:00', -11): 1536}
    warmpixs = {}
    for warm_threshold in (100, 1000):
        for date in ('2017:001:12:00:00', '2020:001:12:00:00'):
            for T_ccd in (-11, -15):
                key = (warm_threshold, date, T_ccd)
                warmpixs[key] = int(get_warm_fracs(*key) * 1024 ** 2)
    for key in warmpixs:
        assert np.allclose(warmpixs[key], exp[key], rtol=1e-5, atol=1)


def test_dark_temp_scale():
    scale = dark_temp_scale(-10., -14)
    assert np.allclose(scale, 0.70)

    scale = dark_temp_scale(-10., -14, scale_4c=2.0)
    assert scale == 0.5  # Should be an exact match


@pytest.mark.skipif('not HAS_MICA')
def test_synthetic_dark_image():
    """Predict 2017:185 dark cal"""
    np.random.seed(100)
    dark = synthetic_dark_image('2017:010:12:00:00', t_ccd_ref=-13.55)
    assert dark.shape == (1024, 1024)

    # Warm pixels above threshold
    lims = [100, 200, 1000, 2000, 3000]
    wps = []
    for lim in lims:
        wps.append(np.count_nonzero(dark > lim))
    wps = np.array(wps)

    # Actual from 2017:185 dark cal at -13.55C : [218214, 83902, N/A, 88, 32]
    #                                            [100,   200, 1000, 2000, 3000]
    exp = np.array([229413, 89239, 675, 84, 27])

    assert np.all(np.abs(wps - exp) < np.sqrt(exp) * 3)


def test_get_warm_fracs_2017185():
    """Predict 2017:185 dark cal"""
    wfs = get_warm_fracs([100, 200, 1000, 2000, 3000], date='2017:010:12:00:00', T_ccd=-13.55)
    wps = wfs * 1024 ** 2

    # Actual from 2017:185 dark cal at -13.55C : [218214, 83902, N/A, 88, 32]
    #                                            [100,   200, 1000, 2000, 3000]
    exp = np.array([207845, 79207, 635, 86, 26])

    assert np.allclose(wps, exp, rtol=0.001, atol=2)
