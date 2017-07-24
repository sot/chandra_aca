from __future__ import print_function, division

import numpy as np
from chandra_aca.star_probs import t_ccd_warm_limit, mag_for_p_acq, acq_success_prob


def test_t_ccd_warm_limit():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=(2, 8e-3))
    assert np.allclose(out[0], -15.2449, atol=0.01, rtol=0)
    assert np.allclose(out[1], 0.008, atol=0.0001, rtol=0)

    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=5.0)
    assert np.allclose(out[0], -15.143, atol=0.01, rtol=0)
    assert np.allclose(out[1], 5.0, atol=0.01, rtol=0)


def test_mag_for_p_acq():
    mag = mag_for_p_acq(0.50, date='2015:001', t_ccd=-14.0)
    assert np.allclose(mag, 10.821, rtol=0, atol=0.01)


def test_halfwidth_adjustment():
    for mag, mults in ((6.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
                       (8.5, [1.0, 1.0, 1.0, 1.0, 1.0]),
                       (9.25, [0.9, 0.9, 0.8, 0.7, 0.505]),
                       (10.0, [0.8, 0.8, 0.6, 0.4, 0.01]),
                       (11.0, [0.01, 0.01, 0.01, 0.01, 0.01])):
        pacq = acq_success_prob(mag=mag, halfwidth=120)
        p135 = acq_success_prob(mag=mag, halfwidth=135)
        p160 = acq_success_prob(mag=mag, halfwidth=160)
        p180 = acq_success_prob(mag=mag, halfwidth=180)
        p240 = acq_success_prob(mag=mag, halfwidth=240)
        p280 = acq_success_prob(mag=mag, halfwidth=280)
        exp_mults = np.array([p135, p160, p180, p240, p280]) / pacq
        assert np.allclose(mults, exp_mults)


def test_acq_success_prob():
    """
    Regression tests for star acquisition probabilities
    """
    date = ['2014:001', '2015:001', '2016:001', '2017:001']
    t_ccd = [-16, -14, -12, -10]
    mag = [9, 10, 10.3, 10.6]
    spoiler = [False, True]
    color = [0.6, 0.7, 1.5]

    p_0p7color = .4294  # probability multiplier for a B-V = 0.700 star (REF?)
    p_spoiler = .9241  # probability multiplier for a search-spoiled star (REF?)

    # Vary date
    probs = acq_success_prob(date=date, t_ccd=-10, mag=10.3, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.76563703, 0.72643584, 0.68541268, 0.64104787])

    # Vary t_ccd
    probs = acq_success_prob(date='2017:001', t_ccd=t_ccd, mag=10.3, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.85391686, 0.7958066, 0.72391895, 0.64104787])

    # Vary mag
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=mag, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.985, 0.85682991, 0.64104787, 0.32454816])

    # Vary spoiler
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=spoiler, color=0.6)
    assert np.allclose(p_spoiler, probs[1] / probs[0])

    # Vary color
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=False, color=color)
    assert np.allclose(probs, [0.64104787,  0.27526596,  0.29436984])
    assert np.allclose(p_0p7color, probs[1] / probs[0])
