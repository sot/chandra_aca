# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Calculate attitude based on star centroid values using a fast linear
least-squares method.

Note this requires Python 3.5+.

Validation:
http://nbviewer.jupyter.org/url/asc.harvard.edu/mta/ASPECT/ipynb/chandra_aca/calc_att_validate.ipynb
"""

from __future__ import division

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

    Parameters
    ----------
    yag
        reference yag (list or array)
    zag
        reference zag (list or array)
    yag_obs
        observed yag (list or array)
    zag_obs
        observed zag (list or array)

    Returns
    -------
    roll (deg)
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

    # When Py2 is no longer supported...
    # theta = -(yag @ zag_obs - zag @ yag_obs) / (yag @ yag + zag @ zag)
    theta = -(yag.dot(zag_obs) - zag.dot(yag_obs)) / (yag.dot(yag) + zag.dot(zag))
    return np.degrees(theta)


def calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs, sigma=None):
    """Calc S/C delta roll, pitch, and yaw for observed star positions relative to reference.

    This function computes a S/C delta roll/pitch/yaw that transforms the
    reference star positions yag/zag into the observed positions
    yag_obs/zag_obs.  The units for these values must be in arcsec.

    The ``yag`` and ``zag`` values correspond to the reference star catalog
    positions.  These must be a 1-d list or array of length M (number of
    stars).

    The ``yag_obs`` and ``zag_obs`` values must be either a 1-d or 2-d array
    with shape M (single readout of M stars) or shape N x M (N rows of M
    stars).

    The ``sigma`` parameter can be None or a 1-d array of length M.

    The algorithm is a simple but fast linear least-squared solution which uses
    a small angle assumption to linearize the rotation matrix from
    [[cos(th) -sin(th)], [sin(th), cos(th)]] to [[1, -th], [th, 1]].
    In practice anything below 1.0 degree is fine.

    Parameters
    ----------
    yag
        reference yag (list or array, arcsec)
    zag
        reference zag (list or array, arcsec)
    yag_obs
        observed yag (list or array, arcsec)
    zag_obs
        observed zag (list or array, arcsec)
    sigma
        centroid uncertainties (None or list or array, arcsec)

    Returns
    -------
    roll, pitch, yaw (degrees)
    """
    yag = np.array(yag)
    zag = np.array(zag)
    yag_obs = np.array(yag_obs)
    zag_obs = np.array(zag_obs)

    if yag.ndim != 1 or zag.ndim != 1 or yag.shape != zag.shape:
        raise ValueError("yag and zag must be 1-d and equal length")

    if (
        yag_obs.ndim not in (1, 2)
        or zag.ndim not in (1, 2)
        or yag_obs.shape != zag_obs.shape
    ):
        raise ValueError("yag_obs and zag_obs must be 1-d or 2-d and equal shape")

    n_stars = len(yag)
    if yag_obs.shape[-1] != n_stars or zag.shape[-1] != n_stars:
        raise ValueError("inconsistent number of stars in yag_obs or zag_obs")

    one_d = yag_obs.ndim == 1
    if one_d:
        yag_obs.shape = 1, n_stars
        zag_obs.shape = 1, n_stars

    outs = []
    for yo, zo in zip(yag_obs, zag_obs):
        out = _calc_roll_pitch_yaw(yag, zag, yo, zo, sigma=sigma)
        outs.append(out)

    if one_d:
        roll, pitch, yaw = outs[0]
    else:
        vals = np.array(outs)
        roll, pitch, yaw = vals[:, 0], vals[:, 1], vals[:, 2]

    return roll, pitch, yaw


def _calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs, sigma=None, iter=1):
    """
    Internal version that does the real work of calc_roll_pitch_yaw and
    works on only one sample at a time.
    """
    weights = None if (sigma is None) else 1 / np.array(sigma)
    yag_avg = np.average(yag, weights=weights)
    zag_avg = np.average(zag, weights=weights)
    yag_obs_avg = np.average(yag_obs, weights=weights)
    zag_obs_avg = np.average(zag_obs, weights=weights)

    # Remove the mean linear offset and find roll
    roll = calc_roll(
        yag - yag_avg,
        zag - zag_avg,
        yag_obs - yag_obs_avg,
        zag_obs - zag_obs_avg,
        sigma,
    )

    # Roll the whole constellation to match the reference
    # When Py2 is no longer supported...
    # yag_obs, zag_obs = _rot(roll) @ np.array([yag_obs,
    #                                           zag_obs])
    yag_obs, zag_obs = _rot(roll).dot(np.array([yag_obs, zag_obs]))

    # Now remove the mean linear offset
    yag_obs_avg = np.average(yag_obs, weights=weights)
    zag_obs_avg = np.average(zag_obs, weights=weights)

    dyag = yag_obs_avg - yag_avg
    dzag = zag_obs_avg - zag_avg

    pitch = dzag / 3600
    yaw = -dyag / 3600

    if iter > 0:
        # Allow for recursive iterations to refine estimate.  In practice a single
        # additional pass is enough.
        yag_obs -= dyag
        zag_obs -= dzag

        dr, dp, dy = _calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs, sigma, iter - 1)
        roll += dr
        pitch += dp
        yaw += dy

    return roll, pitch, yaw


def _rot(roll):
    theta = np.radians(roll)
    out = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return out


def calc_att(att, yag, zag, yag_obs, zag_obs, sigma=None):
    """Calc S/C attitude for observed star positions relative to reference.

    This function computes a S/C attitude that transforms the
    reference star positions yag/zag into the observed positions
    yag_obs/zag_obs.  The units for these values must be in arcsec.

    The attitude ``att`` is the reference attitude for the reference star
    catalog.  It can be any value that initializes a Quat object.

    The ``yag`` and ``zag`` values correspond to the reference star catalog
    positions.  These must be a 1-d list or array of length M (number of
    stars).

    The ``yag_obs`` and ``zag_obs`` values must be either a 1-d or 2-d array
    with shape M (single readout of M stars) or shape N x M (N rows of M
    stars).

    The ``sigma`` parameter can be None or a 1-d array of length M.

    The algorithm is a simple but fast linear least-squared solution which uses
    a small angle assumption to linearize the rotation matrix from
    [[cos(th) -sin(th)], [sin(th), cos(th)]] to [[1, -th], [th, 1]].
    In practice anything below 1.0 degree is fine.

    Parameters
    ----------
    att
        reference attitude (Quat-compatible)
    yag
        reference yag (list or array, arcsec)
    zag
        reference zag (list or array, arcsec)
    yag_obs
        observed yag (list or array, arcsec)
    zag_obs
        observed zag (list or array, arcsec)
    sigma
        centroid uncertainties (None or list or array, arcsec)

    Returns
    -------
    Quat or list of Quat
    """
    from Quaternion import Quat

    q_att = Quat(att)

    rolls, pitches, yaws = calc_roll_pitch_yaw(yag, zag, yag_obs, zag_obs, sigma)

    if isinstance(rolls, np.ndarray) and rolls.ndim >= 1:
        out = []
        for roll, pitch, yaw in zip(rolls, pitches, yaws):
            dq = Quat([yaw, -pitch, roll])
            out.append(q_att * dq)
    else:
        dq = Quat([yaws, -pitches, rolls])
        out = q_att * dq

    return out
