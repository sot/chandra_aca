# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import pytest
import numpy as np
from chandra_aca.star_probs import t_ccd_warm_limit, mag_for_p_acq, acq_success_prob


def test_t_ccd_warm_limit_1():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=(2, 8e-3))
    assert np.allclose(out[0], -14.9924, atol=0.01, rtol=0)
    assert np.allclose(out[1], 0.008, atol=0.0001, rtol=0)


def test_t_ccd_warm_limit_2():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=5.0)
    assert np.allclose(out[0], -14.851, atol=0.01, rtol=0)
    assert np.allclose(out[1], 5.0, atol=0.01, rtol=0)


def test_mag_for_p_acq():
    mag = mag_for_p_acq(0.50, date='2015:001', t_ccd=-14.0)
    assert np.allclose(mag, 10.848, rtol=0, atol=0.01)


def test_halfwidth_adjustment():
    mag = 10.3
    halfwidth = [40, 80, 120, 180, 240]
    p120 = acq_success_prob(mag=mag, date='2018:001', halfwidth=120)
    pacq = acq_success_prob(mag=mag, date='2018:001', halfwidth=halfwidth)
    mults = pacq / p120
    assert np.allclose(mults, [1.07260318, 1.04512285,  1., 0.91312975, 0.83667405])




def test_acq_success_prob_date():
    date = ['2014:001', '2015:001', '2016:001', '2017:001']
    probs = acq_success_prob(date=date, t_ccd=-10, mag=10.3, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.76856955,  0.74345895,  0.71609812,  0.68643974])

def test_acq_success_prob_t_ccd():
    t_ccd = [-16, -14, -12, -10]
    probs = acq_success_prob(date='2017:001', t_ccd=t_ccd, mag=10.3, spoiler=False, color=0.6)
    assert np.allclose(probs, [0.87007558,  0.81918958,  0.75767782,  0.68643974])

def test_acq_success_prob_mag():
    mag = [9, 10, 10.3, 10.6]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=mag, spoiler=False, color=0.6)
    assert np.allclose(probs, [ 0.985,  0.86868674,  0.68643974,  0.3952578])


def test_acq_success_prob_spoiler():
    p_spoiler = .9241  # probability multiplier for a search-spoiled star (REF?)
    spoiler = [False, True]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=spoiler, color=0.6)
    assert np.allclose(p_spoiler, probs[1] / probs[0])


def test_acq_success_prob_color():
    p_0p7color = .4294  # probability multiplier for a B-V = 0.700 star (REF?)
    color = [0.6, 0.699997, 0.69999999, 0.7, 0.700001, 1.5, 1.49999999]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=False, color=color)
    assert np.allclose(probs, [ 0.68643974, 0.68643974, 0.29475723, 0.29475723, 0.68643974,
                                0.29295036, 0.29295036])
    assert np.allclose(p_0p7color, probs[2] / probs[0])
    assert np.allclose(p_0p7color, probs[3] / probs[0])

HAS_AGASC = False
try:
    import agasc
    star = agasc.get_star(870058712)
    HAS_AGASC = True
except:
    HAS_AGASC = False
@pytest.mark.skipif('not HAS_AGASC', reason="Test requires AGASC")
def acq_success_prob_from_stars():
    # These are acq stars for obsid 20765
    star_ids = [118882960, 192286696, 192290008, 118758568, 118758336, 192291664, 192284944, 192288240]
    hws = [160, 160, 120, 160, 120, 120, 120, 120]
    stars = [agasc.get_star(agasc_id) for agasc_id in star_ids]
    mags = [star['MAG_ACA'] for star in stars]
    colors = [star['COLOR1'] for star in stars]
    probs = acq_success_prob(date='2018:059', t_ccd=-11.2, mag=mags, color=colors, halfwidth=hws)
    assert np.allclose(probs, [0.978, 0.967, 0.801, 0.755, 0.575, 0.089, 0.682, 0.659],
                       atol=1e-2, rtol=0)
