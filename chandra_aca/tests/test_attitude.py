import numpy as np
import pytest

from Quaternion import Quat
from Ska.quatutil import radec2yagzag

from ..attitude import calc_roll, calc_roll_pitch_yaw


@pytest.mark.parametrize('roll', [-1, -0.1, 50 / 3600, 0.1])
def test_calc_roll(roll):
    stars = [(-1, 1),  # ra, dec in deg
             (1, 1),
             (0, 0),
             (1, -1)]

    q0 = Quat([0, 0, 45])
    dq = Quat([0, 0, roll])
    assert np.isclose(dq.roll, roll)
    q0_roll = q0 * dq
    assert np.isclose(q0_roll.roll, q0.roll + roll)

    yags = []
    zags = []
    yags_obs = []
    zags_obs = []
    for ra, dec in stars:
        yag, zag = radec2yagzag(ra, dec, q0)
        yags.append(yag)
        zags.append(zag)

        yag, zag = radec2yagzag(ra, dec, q0_roll)
        yags_obs.append(yag)
        zags_obs.append(zag)

    out_roll = calc_roll(yags, zags, yags_obs, zags_obs)
    # Computed roll is within 0.1% of actual roll
    assert np.isclose(roll, out_roll, atol=0.0, rtol=0.001)

# Star fields.  This includes a "normal" set of stars and
# a weird set with just two stars that are close together
# but far off axis.  In this case the rotation will
# manifest as a linear offset.
stars = [[(-1, 1), (1, 1), (0, 0), (1, -1)],
         [(-0.1, 1), (0.1, 1)]]


@pytest.mark.parametrize('stars', stars)
@pytest.mark.parametrize('roll', [-10000, 10, 100])
@pytest.mark.parametrize('pitch', [-50, 8, 20])
@pytest.mark.parametrize('yaw', [-20, -8, 50])
def test_calc_roll_pitch_yaw(stars, pitch, yaw, roll):
    roll /= 3600
    pitch /= 3600
    yaw /= 3600

    q0 = Quat([0, 0, 45])
    dq = Quat([yaw, -pitch, roll])
    assert np.isclose(dq.roll, roll)
    assert np.isclose(dq.pitch, pitch)
    assert np.isclose(dq.yaw, yaw)

    q0_offset = q0 * dq
    assert np.isclose(q0_offset.roll, q0.roll + roll)

    yags = []
    zags = []
    yags_obs = []
    zags_obs = []
    for ra, dec in stars:
        yag, zag = radec2yagzag(ra, dec, q0)
        yags.append(yag * 3600)
        zags.append(zag * 3600)

        yag, zag = radec2yagzag(ra, dec, q0_offset)
        yags_obs.append(yag * 3600)
        zags_obs.append(zag * 3600)

    out_roll, out_pitch, out_yaw = calc_roll_pitch_yaw(yags, zags, yags_obs, zags_obs)
    # Computed pitch, yaw, roll within 1% of actual
    assert np.isclose(roll, out_roll, atol=0.2 / 3600, rtol=0.0)
    assert np.isclose(pitch, out_pitch, atol=0.2 / 3600, rtol=0.0)
    assert np.isclose(yaw, out_yaw, atol=0.2 / 3600, rtol=0.0)
    # print(roll, out_roll)
    # print(pitch, out_pitch)
    # print(yaw, out_yaw)
