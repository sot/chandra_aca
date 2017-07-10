import numpy as np
import pytest

from Quaternion import Quat
from Ska.quatutil import radec2yagzag

from ..attitude import calc_roll, calc_roll_pitch_yaw, calc_att


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
@pytest.mark.parametrize('roll', [-1000, 10, 100])
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

    sigma = np.arange(len(yags)) + 1
    out_roll, out_pitch, out_yaw = calc_roll_pitch_yaw(yags, zags, yags_obs, zags_obs, sigma)
    # Computed pitch, yaw, roll within 0.5 arcsec in roll, 0.02 arcsec pitch/yaw
    assert np.isclose(roll, out_roll, atol=0.5 / 3600, rtol=0.0)
    assert np.isclose(pitch, out_pitch, atol=0.02 / 3600, rtol=0.0)
    assert np.isclose(yaw, out_yaw, atol=0.02 / 3600, rtol=0.0)


def get_data_2d():
    stars0 = stars[0]

    q0 = Quat([0, 0, 45])

    yags = []
    zags = []
    for ra, dec in stars0:
        yag, zag = radec2yagzag(ra, dec, q0)
        yags.append(yag * 3600)
        zags.append(zag * 3600)

    yags_obs_list = []
    zags_obs_list = []
    times = np.linspace(0, 1000, 10)
    rolls = np.sin(2 * np.pi * times / 666) * 100 / 3600
    pitches = np.sin(2 * np.pi * times / 1000) * 50 / 3600
    yaws = np.sin(2 * np.pi * times / 707) * 30 / 3600

    qs = []
    for roll, pitch, yaw in zip(rolls, pitches, yaws):
        dq = Quat([yaw, -pitch, roll])
        q0_offset = q0 * dq
        qs.append(q0_offset)

        yags_obs = []
        zags_obs = []
        for ra, dec in stars0:
            yag, zag = radec2yagzag(ra, dec, q0_offset)
            yags_obs.append(yag * 3600)
            zags_obs.append(zag * 3600)
        yags_obs_list.append(yags_obs)
        zags_obs_list.append(zags_obs)

    return q0, rolls, pitches, yaws, qs, yags, zags, yags_obs_list, zags_obs_list


def test_calc_roll_pitch_yaw_2d():
    q0, rolls, pitches, yaws, qs, yags, zags, yags_obs_list, zags_obs_list = get_data_2d()
    # Test direct roll/pitch/yaw computation
    out = calc_roll_pitch_yaw(yags, zags, yags_obs_list, zags_obs_list)
    out_rolls, out_pitches, out_yaws = out

    # Computed pitch, yaw, roll within 0.5 arcsec in roll, 0.02 arcsec pitch/yaw
    assert np.allclose(rolls, out_rolls, atol=0.5 / 3600, rtol=0.0)
    assert np.allclose(pitches, out_pitches, atol=0.02 / 3600, rtol=0.0)
    assert np.allclose(yaws, out_yaws, atol=0.02 / 3600, rtol=0.0)


def test_calc_att():
    """Test attitude quaternion computation"""
    q0, rolls, pitches, yaws, qs, yags, zags, yags_obs_list, zags_obs_list = get_data_2d()
    q_outs = calc_att(q0, yags, zags, yags_obs_list, zags_obs_list)
    assert len(qs) == len(q_outs)
    for q, q_out in zip(qs, q_outs):
        dq = q.dq(q_out)
        assert np.abs(dq.roll0) < 0.5 / 3600
        assert np.abs(dq.pitch) < 0.02 / 3600
        assert np.abs(dq.yaw) < 0.02 / 3600

    # Test attitude quaternion computation
    q_out = calc_att(q0, yags, zags, yags_obs_list[0], zags_obs_list[0])
    assert isinstance(q_out, Quat)
    dq = qs[0].dq(q_out)
    assert np.abs(dq.roll0) < 0.5 / 3600
    assert np.abs(dq.pitch) < 0.02 / 3600
    assert np.abs(dq.yaw) < 0.02 / 3600


def test_calc_roll_pitch_yaw_sigma():
    q0, rolls, pitches, yaws, qs, yags, zags, yags_obs_list, zags_obs_list = get_data_2d()
    yags_obs_list = np.array(yags_obs_list)
    zags_obs_list = np.array(zags_obs_list)

    # Zero noise
    out = calc_roll_pitch_yaw(yags, zags, yags_obs_list, zags_obs_list)
    out_rolls, out_pitches, out_yaws = out
    dr = rolls - out_rolls
    assert np.std(dr) < 0.2 / 3600

    # Linear offset from -25 to +25 arcsec (hot pixel?) with no sigma adjustment
    sigma = [50, 1, 1, 1]
    yags_obs_list[:, 0] += np.linspace(-25, 25, len(zags_obs_list))
    zags_obs_list[:, 0] += np.linspace(-25, 25, len(zags_obs_list))

    out = calc_roll_pitch_yaw(yags, zags, yags_obs_list, zags_obs_list)
    out_rolls, out_pitches, out_yaws = out
    dr = rolls - out_rolls
    assert np.std(dr) > 280 / 3600

    # Linear offset from -25 to +25 arcsec (hot pixel?) with sigma = 50 for bad slot
    out = calc_roll_pitch_yaw(yags, zags, yags_obs_list, zags_obs_list, sigma=sigma)
    out_rolls, out_pitches, out_yaws = out
    dr = rolls - out_rolls
    assert np.std(dr) < 1 / 3600
