"""
Calculate attitude based on star centroid values using a fast linear
least-squares method.

Note this requires Python 3.5+.
"""
import numpy as np


def calc_roll(yag, zag, yag_obs, zag_obs, sigma=None):
    """Calc S/C delta roll for observed star positions relative to reference.

    This function computes a S/C delta roll that transforms the reference star
    positions yag/zag into the observed positions yag_obs/zag_obs.  The units
    for these values can be anything (arcsec, deg) but must all be consistent.

    The inputs are assumed to be a list or array that corresponds to a single
    readout of at least two stars.

    The algorithm is a simple but fast linear least-squared solution which uses
    a small angle assumption to linearize the rotation matrix from
    [[cos(th) -sin(th)], [sin(th), cos(th)]] to [[1, -th], [th, 1]].
    In practice anything below 1.0 degree is fine.

    If there are different measurement uncertainties for the different
    star inputs then one can supply an array of sigma values corresponding
    to each star.

    :param yag: reference yag (list or array)
    :param zag: reference zag (list or array)
    :param yag_obs: observed yag (list or array)
    :param zag_obs: observed zag (list or array)

    :returns: roll (deg)

    """
    yag = np.asarray(yag)
    zag = np.asarray(zag)
    yag_obs = np.asarray(yag_obs)
    zag_obs = np.asarray(zag_obs)

    if sigma is not None:
        sigma = np.asarray(sigma)
        yag = yag / sigma
        zag = zag / sigma
        yag_obs = yag_obs / sigma
        zag_obs = zag_obs / sigma

    theta = -(yag @ zag_obs - zag @ yag_obs) / (yag @ yag + zag @ zag)
    return np.degrees(theta)


def calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs, sigma=None, iter=1):
    """Calc S/C delta roll, pitch, and yaw for observed star positions relative to reference.

    This function computes a S/C delta roll that transforms the reference star
    positions yag/zag into the observed positions yag_obs/zag_obs.  The units
    for these values must be in arcsec.

    The inputs are assumed to be a list or array that corresponds to a single
    readout of at least two stars.

    The algorithm is a simple but fast linear least-squared solution which uses
    a small angle assumption to linearize the rotation matrix from
    [[cos(th) -sin(th)], [sin(th), cos(th)]] to [[1, -th], [th, 1]].
    In practice anything below 1.0 degree is fine.

    If there are different measurement uncertainties for the different
    star inputs then one can supply an array of sigma values corresponding
    to each star.

    :param yag: reference yag (list or array)
    :param zag: reference zag (list or array)
    :param yag_obs: observed yag (list or array)
    :param zag_obs: observed zag (list or array)

    :returns: roll (deg)

    """
    yag = np.array(yag)
    zag = np.array(zag)
    yag_obs = np.array(yag_obs)
    zag_obs = np.array(zag_obs)
    
    weights = None if (sigma is None) else 1 / sigma
    yag_avg = np.average(yag, weights=weights)
    zag_avg = np.average(zag, weights=weights)
    yag_obs_avg = np.average(yag_obs, weights=weights)
    zag_obs_avg = np.average(zag_obs, weights=weights)

    dyag = yag_obs_avg - yag_avg
    dzag = zag_obs_avg - zag_avg

    pitch = dzag / 3600
    yaw = -dyag / 3600

    # Remove the mean linear offset
    yag_obs -= yag_obs_avg
    zag_obs -= zag_obs_avg
    yag -= yag_avg
    zag -= zag_avg
    roll = calc_roll(yag, zag, yag_obs, zag_obs, sigma)

    if iter > 0:
        # Allow for recursive iterations to refine estimate.  In practice a single
        # additional pass is enough.
        yag_obs, zag_obs = rot(roll) @ np.array([yag_obs,
                                                 zag_obs])
        dr, dp, dy = calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs, sigma, iter - 1)
        roll += dr
        pitch += dp
        yaw += dy

    if 1:
        print('YAG:', ['{:3f}'.format(x) for x in yag])
        print('ZAG:', ['{:3f}'.format(x) for x in zag])
        print('dYAG:', ['{:3f}'.format(x) for x in yag - yag_obs])
        print('dZAG:', ['{:3f}'.format(x) for x in zag - zag_obs])

    print(roll*3600, pitch*3600, yaw*3600)
    return roll, pitch, yaw


def rot(roll):
    theta = np.radians(roll)
    out = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return out
