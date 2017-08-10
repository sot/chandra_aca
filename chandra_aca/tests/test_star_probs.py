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
    mag = 10.3
    halfwidth = [40, 80, 120, 180, 240]
    p120 = acq_success_prob(mag=mag, halfwidth=120)
    pacq = acq_success_prob(mag=mag, halfwidth=halfwidth)
    mults = pacq / p120
    assert np.allclose(mults, [1.07721523, 1.04776216, 1., 0.90920264, 0.83013465])


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
