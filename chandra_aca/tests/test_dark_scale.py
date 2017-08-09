import numpy as np
from ..dark_model import dark_temp_scale, get_warm_fracs


def test_get_warm_fracs():
    exp = {(100, '2020:001', -11): 341312,
           (100, '2017:001', -11): 278627,
           (100, '2020:001', -15): 250546,
           (100, '2017:001', -15): 200786,
           (1000, '2017:001', -11): 1703,
           (1000, '2017:001', -15): 558,
           (1000, '2020:001', -15): 798,
           (1000, '2020:001', -11): 2436}
    warmpixs = {}
    for warm_threshold in (100, 1000):
        for date in ('2017:001', '2020:001'):
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


