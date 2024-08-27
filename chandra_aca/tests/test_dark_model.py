# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from mica.archive.aca_dark import get_dark_cal_props
from mica.common import MICA_ARCHIVE

from chandra_aca.dark_model import dark_temp_scale_img

from ..dark_model import dark_temp_scale, get_warm_fracs, synthetic_dark_image

HAS_MICA = os.path.exists(MICA_ARCHIVE)


def test_get_warm_fracs():
    exp = {
        (100, "2020:001:12:00:00", -11): 294302,
        (100, "2017:001:12:00:00", -11): 251865,
        (100, "2020:001:12:00:00", -15): 215096,
        (100, "2017:001:12:00:00", -15): 182365,
        (1000, "2017:001:12:00:00", -11): 1296,
        (1000, "2017:001:12:00:00", -15): 433,
        (1000, "2020:001:12:00:00", -15): 500,
        (1000, "2020:001:12:00:00", -11): 1536,
    }
    warmpixs = {}
    for warm_threshold in (100, 1000):
        for date in ("2017:001:12:00:00", "2020:001:12:00:00"):
            for T_ccd in (-11, -15):
                key = (warm_threshold, date, T_ccd)
                warmpixs[key] = int(get_warm_fracs(*key) * 1024**2)
    for key in warmpixs:
        assert np.allclose(warmpixs[key], exp[key], rtol=1e-5, atol=1)


def test_dark_temp_scale():
    scale = dark_temp_scale(-10.0, -14)
    assert np.allclose(scale, 0.70)

    scale = dark_temp_scale(-10.0, -14, scale_4c=2.0)
    assert scale == 0.5  # Should be an exact match


@pytest.mark.skipif("not HAS_MICA")
def test_synthetic_dark_image():
    """Predict 2017:185 dark cal"""
    np.random.seed(100)
    dark = synthetic_dark_image("2017:010:12:00:00", t_ccd_ref=-13.55)
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
    wfs = get_warm_fracs(
        [100, 200, 1000, 2000, 3000], date="2017:010:12:00:00", T_ccd=-13.55
    )
    wps = wfs * 1024**2

    # Actual from 2017:185 dark cal at -13.55C : [218214, 83902, N/A, 88, 32]
    #                                            [100,   200, 1000, 2000, 3000]
    exp = np.array([207845, 79207, 635, 86, 26])

    assert np.allclose(wps, exp, rtol=0.001, atol=2)


@pytest.mark.parametrize(
    "img, t_ccd, t_ref, expected",
    [
        (np.array([100, 1000, 500]), -10.0, -5.0, np.array([171.47, 1668.62, 852.79])),
        (np.array([100, 1000, 500]), -10.0, -15.0, np.array([58.31, 599.29, 293.15])),
        ([[100, 1000], [500, 600]], -15.0, -15.0, [[100, 1000], [500, 600]]),
        ([200, 2000, 600], -6.0, 2.0, np.array([478.16, 4299.02, 1399.40])),
        (2000, -6, 2, 4299.02)
    ],
)
def test_get_img_scaled(img, t_ccd, t_ref, expected):
    scaled_img = dark_temp_scale_img(img, t_ccd, t_ref)
    assert np.allclose(scaled_img, expected, atol=0.1, rtol=0)
    if np.shape(img):
        assert isinstance(scaled_img, np.ndarray)
    else:
        assert isinstance(scaled_img, float)


@pytest.mark.skipif("not HAS_MICA")
def test_get_img_scaled_real_dc():
    test_data = {}
    with open((Path(__file__).parent / "data" / "dark_scaled_img.pkl"), "rb") as f:
        test_data.update(pickle.load(f))

    dc = get_dark_cal_props("2024:001", include_image=True)
    # just use 100 square pixels instead of 1024x1024
    img = dc["image"][100:200, 100:200]
    scaled_img = dark_temp_scale_img(img, dc["t_ccd"], -3.0)
    assert np.allclose(scaled_img, test_data["scaled_img"], atol=0.1, rtol=0)
