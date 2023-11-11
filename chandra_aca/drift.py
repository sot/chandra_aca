# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Function(s) related to ACA alignment.

In particular compute the dynamical pointing offset required to achieve
a desired zero-offset target aimpoint.

A key element of this module is the fitting analysis here:
https://github.com/sot/aimpoint_mon/blob/master/fit_aimpoint_drift-2018-11.ipynb
"""
import functools
import json
import os
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.utils.data import download_file
from Chandra.Time import DateTime
from cxotime import CxoTimeLike
from ska_helpers import chandra_models
from ska_helpers.utils import LazyDict


def load_drift_pars():
    pars_txt, info = chandra_models.get_data(
        Path("chandra_models") / "aca_drift" / "aca_drift_model.json"
    )
    pars = json.loads(pars_txt)
    pars["info"] = info
    return pars


DRIFT_PARS = LazyDict(load_drift_pars)

# Define transform from aspect solution DY, DZ (mm) to CHIPX, CHIPY for
# nominal target position.  This uses the transform:
# CHIPX, CHIPY = c0 + cyz * [DY, DZ]  (dot product)
# These values come from the aimpoint_mon/asol_to_chip_transforms notebook:
# https://github.com/sot/aimpoint_mon/blob/cebe5010/asol_to_chip_transforms.ipynb
ASOL_TO_CHIP = {
    ("ACIS-I", 0): {
        "c0": [1100.806, 1110.299],
        "cyz": [[0.001, 41.742], [-41.742, 0.105]],
    },
    ("ACIS-I", 1): {
        "c0": [-76.208, 962.413],
        "cyz": [[0.001, -41.742], [41.741, 0.104]],
    },
    ("ACIS-I", 2): {
        "c0": [54.468, 1110.197],
        "cyz": [[-0.001, 41.742], [-41.742, -0.105]],
    },
    ("ACIS-I", 3): {
        "c0": [970.795, 961.857],
        "cyz": [[-0.001, -41.742], [41.742, -0.104]],
    },
    ("ACIS-S", 4): {
        "c0": [3382.364, 520.024],
        "cyz": [[-41.695, -0.021], [0.021, -41.689]],
    },
    ("ACIS-S", 5): {
        "c0": [2338.96, 519.15],
        "cyz": [[-41.691, -0.031], [0.037, -41.689]],
    },
    ("ACIS-S", 6): {
        "c0": [1296.05, 519.532],
        "cyz": [[-41.69, -0.031], [0.036, -41.689]],
    },
    ("ACIS-S", 7): {
        "c0": [251.607, 519.818],
        "cyz": [[-41.689, -0.022], [0.022, -41.689]],
    },
    ("ACIS-S", 8): {
        "c0": [-790.204, 519.784],
        "cyz": [[-41.69, -0.02], [0.02, -41.689]],
    },
    ("ACIS-S", 9): {
        "c0": [-1832.423, 519.546],
        "cyz": [[-41.693, -0.017], [0.017, -41.689]],
    },
    ("HRC-I", 0): {
        "c0": [7635.787, 7697.627],
        "cyz": [[-109.98, 109.98], [109.98, 109.98]],
    },
    ("HRC-S", 1): {
        "c0": [2196.931, 25300.796],
        "cyz": [[0.188, -155.535], [155.584, 0.19]],
    },
    ("HRC-S", 2): {
        "c0": [2197.753, 8842.306],
        "cyz": [[0.2, -155.535], [155.535, 0.184]],
    },
    ("HRC-S", 3): {
        "c0": [2196.638, -7615.26],
        "cyz": [[0.2, -155.535], [155.571, 0.184]],
    },
}

SIM_MM_TO_ARCSEC = 20.493


class AcaDriftModel(object):
    """
    Define a drift model for aspect solution SIM DY/DZ values as a function of
    time and ACA CCD temperature.  This expresses the model which is defined
    and fitted in the fit_aimpoint_drift notebook in this repo.
    """

    def __init__(self, scale, offset, trend, jumps, year0):
        self.scale = scale
        self.offset = offset
        self.trend = trend
        self.jumps = jumps
        self.year0 = year0

    def calc(self, times, t_ccd):
        """
        Calculate the drift model for aspect solution SIM DY/DZ values for input
        ``times`` and ``t_ccd``.  The two arrays are broadcasted to match.

        The returned drifts are in arcsec and provide the expected aspect solution
        SIM DY or DZ values in arcsec.  This can be converted to a drift in mm
        (corresponding to units in an ASOL file) via the scale factor 20.493 arcsec/mm.

        Parameters
        ----------
        times
            array of times (CXC secs)
        t_ccd
            CCD temperatures (degC)

        Returns
        -------
        array of ASOL SIM DY/DZ (arcsec)
        """
        # The drift model is calibrated assuming t_ccd is in degF, but we want inputs
        # in degC, so convert at this point.
        t_ccd_degF = t_ccd * 1.8 + 32.0

        times, t_ccd_degF = np.broadcast_arrays(times, t_ccd_degF)
        is_scalar = times.ndim == 0 and t_ccd_degF.ndim == 0
        times = DateTime(np.atleast_1d(times)).secs
        t_ccd_degF = np.atleast_1d(t_ccd_degF)

        if times.shape != t_ccd_degF.shape:
            raise ValueError("times and t_ccd args must match in shape")

        if np.any(np.diff(times) < 0):
            raise ValueError("times arg must be monotonically increasing")

        if times[0] < DateTime("2012:001:12:00:00").secs:
            raise ValueError("model is not applicable before 2012")

        # Years from model `year0`
        dyears = DateTime(times, format="secs").frac_year - self.year0

        # Raw offsets without jumps
        out = (t_ccd_degF - self.offset) * self.scale + dyears * self.trend

        # Put in the step function jumps
        for jump_date, jump in self.jumps:
            jump_idx = np.searchsorted(times, DateTime(jump_date).secs)
            out[jump_idx:] += jump

        return out[0] if is_scalar else out


def get_fid_offset(time: CxoTimeLike, t_ccd: float) -> tuple:
    """
    Compute the fid light offset values for a given time and temperature.

    Parameters
    ----------
    time : CxoTimeLike format
        Time for offset calculation.
    t_ccd : float
        ACA CCD temperature in degrees Celsius.

    Returns
    -------
    tuple
        A tuple containing the y-angle and z-angle offsets (in arcseconds) to apply
        additively to the nominal (FEB07) fid positions.

    Notes
    -----
    The apparent fid light positions change in accordance with the ACA alignment drift as a
    function of time and temperature. This is captured in the ACA aimpoint drift model. This
    function uses that model to provide the offsets in y-angle and z-angle (arcsec) to apply
    additively to the nominal fid positions.

    The y_offset and z_offset values in this function were calibrated using the
    2022-11 aimpoint drift model and the FEB07 fid characteristics.
    See https://github.com/sot/fid_drift_mon/blob/master/fid_offset_coeff.ipynb
    """

    # Define model instances using calibrated parameters
    drift_y = AcaDriftModel(**DRIFT_PARS["dy"])
    drift_z = AcaDriftModel(**DRIFT_PARS["dz"])

    # Compute the predicted asol DY/DZ based on time and ACA CCD temperature
    # via the predictive model calibrated in the fit_aimpoint_drift notebook
    # in this repo.  And flip the signs.
    dy_pred = -1.0 * drift_y.calc(time, t_ccd)
    dz_pred = -1.0 * drift_z.calc(time, t_ccd)

    # Apply internal offset that places the fid lights at ~zero position
    # offset during the 2022:094 to 2023:044.
    y_offset = 19.6
    z_offset = 20.1
    return dy_pred + y_offset, dz_pred + z_offset


def get_aca_offsets(detector, chip_id, chipx, chipy, time, t_ccd):
    """
    Compute the dynamical ACA offset values for the provided inputs.

    The ``time`` and ``t_ccd`` inputs can be either scalars or arrays.

    Parameters
    ----------
    detector
        one of ACIS-I, ACIS-S, HRC-I, HRC-S
    chipx
        zero-offset aimpoint CHIPX
    chipy
        zero-offset aimpoint CHIPY
    chip_id
        zero-offset aimpoint CHIP ID
    time
        time(s) of observation (any Chandra.Time compatible format)
    t_ccd
        ACA CCD temperature(s) (degC)

    Returns
    -------
    aca_offset_y, aca_offset_z (arcsec)
    """
    # Define model instances using calibrated parameters
    drift_y = AcaDriftModel(**DRIFT_PARS["dy"])
    drift_z = AcaDriftModel(**DRIFT_PARS["dz"])

    try:
        asol_to_chip = ASOL_TO_CHIP[detector, chip_id]
    except KeyError:
        raise KeyError(
            "Detector and chip combination {} not in allow values: {}".format(
                (detector, chip_id), sorted(ASOL_TO_CHIP.keys())
            )
        )

    # Compute the asol DY/DZ that would be required for the aimpoint to be
    # exactly at the desired CHIPX/Y values.  Uses the geometrical transform
    # computed via dmcoords in the asol_to_chip_transform notebook in this repo.
    # CHIPX, CHIPY = c0 + cyz * [DY, DZ]  (dot product)
    chip_xy = np.array([chipx, chipy])
    cyz_inv = np.linalg.inv(asol_to_chip["cyz"])
    dy_chip, dz_chip = (
        cyz_inv.dot(chip_xy - asol_to_chip["c0"]) * SIM_MM_TO_ARCSEC
    )  # arcsec

    # Compute the predicted asol DY/DZ based on time and ACA CCD temperature
    # via the predictive model calibrated in the fit_aimpoint_drift notebook
    # in this repo.
    dy_pred = drift_y.calc(time, t_ccd)
    dz_pred = drift_z.calc(time, t_ccd)

    # The difference is the dynamic ACA offset that must be applied to the attitude.
    # This has the same sign convention as the user-supplied TARGET OFFSET in the
    # ObsCat / OR-list.
    ddy = dy_chip - dy_pred
    ddz = dz_chip - dz_pred

    return ddy, ddz


@functools.lru_cache
def get_default_zero_offset_table():
    """
    Get official SOT MP zero offset aimpoint table.

    First try ``/data/mpcrit1/aimpoint_table/zero_offset_aimpoints.txt``.
    If that is not available use:
    https://cxc.harvard.edu/mta/ASPECT/drift/zero_offset_aimpoints.txt.
    The web version is updated weekly on Sunday via a Ska cron job.

    Note the definitive source of this file is:
    https://icxc.harvard.edu/mp/html/aimpoint_table/zero_offset_aimpoints.txt.

    Returns
    -------
    zero offset aimpoint table as astropy.Table
    """
    try:
        path = (
            Path(os.environ["SKA"])
            / "data"
            / "mpcrit1"
            / "aimpoint_table"
            / "zero_offset_aimpoints.txt"
        )
        out = Table.read(str(path), format="ascii")
    except FileNotFoundError:
        url = "https://cxc.harvard.edu/mta/ASPECT/drift/zero_offset_aimpoints.txt"
        path = download_file(url, show_progress=False, timeout=10)
        out = Table.read(path, format="ascii")

    return out


def get_target_aimpoint(date, cycle, detector, too=False, zero_offset_table=None):
    """
    Given date, proposal cycle, and detector, return aimpoint chipx, chipy, chip_id

    Parameters
    ----------
    date
        observation date
    cycle
        proposal cycle of observation
    detector
        target detector
    too
        boolean. If target is TOO use current cycle not proposal cycle.
    zero_offset_able : table (astropy or numpy) of zero offset aimpoint table
        defaults to official SOT MP version if not supplied.

    Returns
    -------
    tuple of chipx, chipy, chip_id
    """
    if zero_offset_table is None:
        zero_offset_table = get_default_zero_offset_table()
    zero_offset_table.sort(["date_effective", "cycle_effective"])
    date = DateTime(date).iso[:10]
    # Entries for this detector before the 'date' given
    ok = (zero_offset_table["detector"] == detector) & (
        zero_offset_table["date_effective"] <= date
    )
    # If a regular observation, the entry must also be before or equal to proposal cycle
    if not too:
        ok = ok & (zero_offset_table["cycle_effective"] <= cycle)
    filtered_table = zero_offset_table[ok]
    # Return the desired keys in the most recent [-1] row that matches
    return tuple(filtered_table[["chipx", "chipy", "chip_id"]][-1])
