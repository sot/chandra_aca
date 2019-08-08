# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import itertools
import os
import pytest
import numpy as np
from astropy.table import Table

from chandra_aca.star_probs import (t_ccd_warm_limit, mag_for_p_acq, acq_success_prob,
                                    guide_count, t_ccd_warm_limit_for_guide,
                                    grid_model_acq_prob, snr_mag_for_t_ccd,
                                    binom_ppf)

# Acquisition probabilities regression test data
ACQ_PROBS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'acq_probs.dat')


def make_prob_regress_table():
    """
    Make a table that can be copy/pasted here for regression testing of the
    acq probability model(s).  This is primarily to ensure uniformity over
    platforms and potentially changes like a numpy upgrade.  One can also
    do a by-eye sanity check diff when the model params are updated.

    Usage::

      >>> from chandra_aca.tests import test_star_probs
      >>> test_star_probs.make_prob_regress_table()

    This creates ``chandra_aca/tests/data/acq_probs.dat``.
    """
    mags = [7.0, 10.0]
    t_ccds = [-15, -10]
    models = ['sota', 'spline', 'grid-floor-2018-11']
    colors = [0.7, 1.0, 1.5]
    spoilers = [True, False]
    halfwidths = [120, 160]
    rows = []
    for model, mag, t_ccd, color, spoiler, halfwidth in itertools.product(
            models, mags, t_ccds, colors, spoilers, halfwidths):
        prob = acq_success_prob(date='2018:001', t_ccd=t_ccd, mag=mag, color=color,
                                spoiler=spoiler, halfwidth=halfwidth,
                                model=model)
        rows.append([model, mag, t_ccd, color, spoiler, halfwidth, prob])
    out = Table(rows=rows,
                names=['model', 'mag', 't_ccd', 'color', 'spoiler', 'halfwidth', 'prob'])
    out['prob'].format = '.5f'
    out.write(ACQ_PROBS_FILE, format='ascii.ecsv', overwrite=True)


def test_acq_probs_values():
    dat = Table.read(ACQ_PROBS_FILE, format='ascii.ecsv', guess=False)
    for model, mag, t_ccd, color, spoiler, halfwidth, prob in dat:
        prob_now = acq_success_prob(date='2018:001', t_ccd=t_ccd, mag=mag, color=color,
                                    spoiler=spoiler, halfwidth=halfwidth,
                                    model=model)
        # Values written to file rounded to 1e-5, so test to 2e-5
        assert np.isclose(prob, prob_now, atol=2e-5, rtol=0)


def test_t_ccd_warm_limit_1():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=(2, 8e-3), model='sota')
    assert np.allclose(out[0], -14.9924, atol=0.01, rtol=0)
    assert np.allclose(out[1], 0.008, atol=0.0001, rtol=0)


def test_t_ccd_warm_limit_1_spline():
    out = t_ccd_warm_limit([10.0] * 6, date='2018:180', min_n_acq=(2, 8e-3), model='spline')
    assert np.allclose(out[0], -10.582, atol=0.01, rtol=0)
    assert np.allclose(out[1], 0.008, atol=0.0001, rtol=0)


def test_t_ccd_warm_limit_2():
    out = t_ccd_warm_limit([10.4] * 6, date='2015:001', min_n_acq=5.0, model='sota')
    assert np.allclose(out[0], -14.851, atol=0.01, rtol=0)
    assert np.allclose(out[1], 5.0, atol=0.01, rtol=0)


def test_t_ccd_warm_limit_2_spline():
    out = t_ccd_warm_limit([10.0] * 6, date='2018:180', min_n_acq=5.0, model='spline')
    assert np.allclose(out[0], -10.491, atol=0.01, rtol=0)
    assert np.allclose(out[1], 5.0, atol=0.01, rtol=0)


def test_t_ccd_warm_limit_3():
    halfwidth = [40, 80, 120, 160, 180, 240]
    box = t_ccd_warm_limit([10.4] * 6, date='2015:001', halfwidths=halfwidth, min_n_acq=(2, 8e-3), model='sota')
    assert np.allclose(box[0], -15.6325, atol=0.01, rtol=0)
    assert np.allclose(box[1], 0.008, atol=0.0001, rtol=0)


def test_t_ccd_warm_limit_3_spline():
    halfwidth = [40, 80, 120, 160, 180, 240]
    box = t_ccd_warm_limit([10.0] * 6, date='2018:180', halfwidths=halfwidth, min_n_acq=(2, 8e-3), model='spline')
    assert np.allclose(box[0], -11.0192, atol=0.01, rtol=0)
    assert np.allclose(box[1], 0.008, atol=0.0001, rtol=0)


def test_t_ccd_warm_limit_guide():
    mags = np.array([5.85] * 5)
    t_ccd = t_ccd_warm_limit_for_guide(mags, warm_t_ccd=5.0, cold_t_ccd=-16)
    assert np.isclose(t_ccd, -16, atol=0.1, rtol=0)
    mags = np.array([6.0, 6.0, 6.0, 6.0, 6.0])
    t_ccd = t_ccd_warm_limit_for_guide(mags, warm_t_ccd=5.0, cold_t_ccd=-16)
    assert np.isclose(t_ccd, 5.0, atol=0.1, rtol=0)
    mags = np.array([6.0, 6.0, 6.0, 10.3, 10.3])
    t_ccd = t_ccd_warm_limit_for_guide(mags, warm_t_ccd=5.0, cold_t_ccd=-16)
    assert np.isclose(t_ccd, -11.41, atol=0.1, rtol=0)
    mags = np.array([10.3, 10.3, 10.3, 10.3, 10.3])
    t_ccd = t_ccd_warm_limit_for_guide(mags, warm_t_ccd=5.0, cold_t_ccd=-16)
    assert np.isclose(t_ccd, -12.86, atol=0.1, rtol=0)


@pytest.mark.parametrize('count_9th', [False, True])
def test_guide_count(count_9th):
    """Test fractional guide count"""

    # Evaluate at interpolation curve reference temperature t_ccd = -10.9 C.
    mags = [5.0, 5.85, 5.9, 5.95, 9.99, 10.0, 10.1, 10.2, 10.25, 10.3, 10.35, 10.4, 10.41, 11.0]
    exps = [0.0, 0.0, 0.50, 1.0, 1.00, 1.0, 0.875, 0.75, 0.625, 0.50, 0.25, 0.000, 0.00, 0.00]
    mags = np.array(mags)

    if count_9th:
        # This corresponds to the effective mag adjustment for 9th mag counting
        mags[mags > 6.1] -= 1.0
    else:
        mags[mags > 6.1] -= 0.05

    for mag, exp in zip(mags, exps):
        cnt = guide_count([mag], t_ccd=-10.9, count_9th=count_9th)
        assert np.isclose(cnt, exp, atol=0.001, rtol=0)

    # Evaluate at different t_ccd, but change mags accordingly to the
    # SNR-equivalent mag.
    for t_ccd in (-8, -10, -12, -14):
        for mag, exp in zip(mags, exps):
            new_mag = snr_mag_for_t_ccd(t_ccd, mag, -10.9) if (mag > 6.1) else mag
            cnt = guide_count([new_mag], t_ccd=t_ccd, count_9th=count_9th)
            assert np.isclose(cnt, exp, atol=0.001, rtol=0)


def test_t_ccd_warm_limit_guide_vs_brute():
    for n in range(0, 100):
        mags = np.random.normal(loc=9.0, scale=1.5, size=5)
        warm_limit = t_ccd_warm_limit_for_guide(mags)
        check_warm_limit = stepwise_guide_warm_limit(mags, step=.01)
        assert np.isclose(warm_limit, check_warm_limit, atol=0.02, rtol=0)


def stepwise_guide_warm_limit(mags, step=0.01, min_guide_count=4.0,
                              warm_t_ccd=-5.0, cold_t_ccd=-16.0):
    """
    Solve for the warmest temperature that still gets ``min_guide_count``, but
    using a stepwise/brute method as needed.  This is slow, but good for a comparison
    for testing.
    """
    if guide_count(mags, warm_t_ccd) >= min_guide_count:
        return warm_t_ccd
    if guide_count(mags, cold_t_ccd) < min_guide_count:
        return cold_t_ccd
    t_ccds = np.arange(cold_t_ccd, warm_t_ccd, step)
    counts = np.array([guide_count(mags, t_ccd) for t_ccd in t_ccds])
    max_idx = np.flatnonzero(counts >= min_guide_count)[-1]
    return t_ccds[max_idx]


def test_mag_for_p_acq():
    mag = mag_for_p_acq(0.50, date='2015:001', t_ccd=-14.0, model='sota')
    assert np.allclose(mag, 10.848, rtol=0, atol=0.01)


def test_halfwidth_adjustment():
    mag = 10.3
    halfwidth = [40, 80, 120, 180, 240]
    p120 = acq_success_prob(mag=mag, date='2018:001', t_ccd=-19, halfwidth=120, model='sota')
    pacq = acq_success_prob(mag=mag, date='2018:001', t_ccd=-19, halfwidth=halfwidth, model='sota')
    mults = pacq / p120
    assert np.allclose(mults, [1.07260318, 1.04512285,  1., 0.91312975, 0.83667405])


def test_acq_success_prob_date():
    date = ['2014:001', '2015:001', '2016:001', '2017:001']
    probs = acq_success_prob(date=date, t_ccd=-10, mag=10.3, spoiler=False, color=0.6,
                             model='sota')
    assert np.allclose(probs, [0.76856955,  0.74345895,  0.71609812,  0.68643974])


def test_acq_success_prob_t_ccd():
    t_ccd = [-16, -14, -12, -10]
    probs = acq_success_prob(date='2017:001', t_ccd=t_ccd, mag=10.3, spoiler=False, color=0.6,
                             model='sota')
    assert np.allclose(probs, [0.87007558,  0.81918958,  0.75767782,  0.68643974])


def test_acq_success_prob_mag():
    mag = [9, 10, 10.3, 10.6]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=mag, spoiler=False, color=0.6,
                             model='sota')
    assert np.allclose(probs, [0.985, 0.86868674, 0.68643974, 0.3952578])


def test_acq_success_prob_spoiler():
    p_spoiler = .9241  # probability multiplier for a search-spoiled star (REF?)
    spoiler = [False, True]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=spoiler, color=0.6,
                             model='sota')
    assert np.allclose(p_spoiler, probs[1] / probs[0])


def test_acq_success_prob_color():
    p_0p7color = .4294  # probability multiplier for a B-V = 0.700 star (REF?)
    color = [0.6, 0.699997, 0.69999999, 0.7, 0.700001, 1.5, 1.49999999]
    probs = acq_success_prob(date='2017:001', t_ccd=-10, mag=10.3, spoiler=False, color=color,
                             model='sota')
    assert np.allclose(probs, [0.68643974, 0.68643974, 0.29475723, 0.29475723, 0.68643974,
                               0.29295036, 0.29295036])
    assert np.allclose(p_0p7color, probs[2] / probs[0])
    assert np.allclose(p_0p7color, probs[3] / probs[0])


def test_acq_success_prob_from_stars():
    """
    Test for acq stars for obsid 20765 (which had only 2 stars ID'd)
    """

    # Results from this code block are below.  Just hardwire the values since this is a
    # test on star probabilities, not the AGASC.
    #
    # star_ids = [118882960, 192286696, 192290008, 118758568,
    #             118758336, 192291664, 192284944, 192288240]
    # stars = [agasc.get_star(agasc_id, fix_color1=False) for agasc_id in star_ids]
    # mags = [star['MAG_ACA'] for star in stars]
    # colors = [star['COLOR1'] for star in stars]

    mags = [9.142868,
            9.3232698,
            10.16424,
            10.050572,
            10.406073,
            11.094579,
            10.290914,
            10.355055]
    colors = [0.66640031,
              0.27880007,
              0.50830042,
              0.40374953,
              0.93329966,
              1.5,
              0.89249933,
              0.57800025]
    hws = [160, 160, 120, 160, 120, 120, 120, 120]


    # SOTA
    probs = acq_success_prob(date='2018:059', t_ccd=-11.2, mag=mags, color=colors,
                             halfwidth=hws, model='sota')
    assert np.allclose(probs, [0.977, 0.967, 0.793, 0.775, 0.606, 0.0004, 0.704, 0.651],
                       atol=1e-2, rtol=0)

    # Spline
    probs = acq_success_prob(date='2018:059', t_ccd=-11.2, mag=mags, color=colors,
                             halfwidth=hws, model='spline')
    assert np.allclose(probs, [0.954, 0.936, 0.696, 0.739, 0.297, 0.000001, 0.491, 0.380],
                       atol=1e-2, rtol=0)


def test_grid_floor_2018_11():
    """
    Test grid-floor-2018-11 model against values computed directly in the
    source notebook fit_acq_model-2018-11-binned-poly-binom-floor.ipynb
    with the analytical (not-gridded) model.
    """

    mags = [9, 9.5, 10.5]
    t_ccds = [-10, -5]
    halfws = [60, 120, 160]
    mag, t_ccd, halfw = np.meshgrid(mags, t_ccds, halfws, indexing='ij')

    # color not 1.5
    probs = grid_model_acq_prob(mag, t_ccd, halfwidth=halfw, probit=True, color=1.0,
                                model='grid-floor-2018-11')

    exp = -np.array([-2.275, -2.275, -2.275, -2.275, -1.753, -1.467, -1.749, -1.749,
                     -1.749, -1.503, -0.948, -0.662, 0.402, 0.957, 1.244, 1.546,
                     2.101, 2.387])

    assert np.allclose(probs.flatten(), exp, rtol=0, atol=0.08)

    # color 1.5
    probs = grid_model_acq_prob(mag, t_ccd, halfwidth=halfw, probit=True, color=1.5,
                                model='grid-floor-2018-11')

    exp = -np.array([-1.657, -1.53, -1.455, -1.311, -1.033, -0.863, -1.167, -0.974,
                     -0.875, -0.695, -0.382, -0.204, 0.386, 0.758, 0.938, 1.133,
                     1.476, 1.639])
    assert np.allclose(probs.flatten(), exp, rtol=0, atol=0.001)


def test_binom_ppf():
    vals = binom_ppf(4, 5, [0.17, 0.84])
    assert np.allclose(vals, [0.55463945, 0.87748177])
