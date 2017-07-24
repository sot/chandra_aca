from __future__ import print_function, division

import numpy as np
from chandra_aca.star_probs import t_ccd_warm_limit, mag_for_p_acq, acq_success_prob


def test_t_ccd_warm_limit_1():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=(2, 8e-3))
    assert np.allclose(out[0], -14.87863, atol=0.01, rtol=0)
    assert np.allclose(out[1], 0.008, atol=0.0001, rtol=0)


def test_t_ccd_warm_limit_2():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=5.0)
    assert np.allclose(out[0], -14.731, atol=0.01, rtol=0)
    assert np.allclose(out[1], 5.0, atol=0.01, rtol=0)


def test_mag_for_p_acq():
    mag = mag_for_p_acq(0.50, date='2015:001', t_ccd=-14.0)
    assert np.allclose(mag, 10.858, rtol=0, atol=0.01)


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


def test_acq_success_prob_date():
    date = ['2014:001', '2015:001', '2016:001', '2017:001']
    probs = acq_success_prob(date=date, t_ccd=-10, mag=10.3, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.78565034, 0.75125101, 0.71514977, 0.6759113])


def test_acq_success_prob_t_ccd():
    t_ccd = [-16, -14, -12, -10]
    probs = acq_success_prob(date='2017:001', t_ccd=t_ccd, mag=10.3, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.86318705, 0.81210466, 0.74903986, 0.6759113])


def test_acq_success_prob_mag():
    mag = [9, 10, 10.3, 10.6]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=mag, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.985, 0.86465667, 0.6759113, 0.37816025])


def test_acq_success_prob_spoiler():
    p_spoiler = .9241  # probability multiplier for a search-spoiled star (REF?)
    spoiler = [False, True]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=spoiler, color=0.6)
    assert np.allclose(p_spoiler, probs[1] / probs[0])


def test_acq_success_prob_color():
    p_0p7color = .4294  # probability multiplier for a B-V = 0.700 star (REF?)
    color = [0.6, 0.7, 1.5]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=False, color=color)
    assert np.allclose(probs, [0.6759113, 0.29023631, 0.2541498])
    assert np.allclose(p_0p7color, probs[1] / probs[0])
