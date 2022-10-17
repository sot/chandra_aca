# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Function(s) related to ACA alignment.

In particular compute the dynamical pointing offset required to achieve
a desired zero-offset target aimpoint.

A key element of this module is the fitting analysis here:
https://github.com/sot/aimpoint_mon/blob/master/fit_aimpoint_drift-2018-11.ipynb
"""

from Chandra.Time import DateTime
from astropy.table import Table
import numpy as np

# Capture best fit model parameters for ACA drift model.
# https://github.com/sot/aimpoint_mon/blob/7809b89/fit_aimpoint_drift.ipynb
# These are used below to instantiate corresponding AcaDriftModel objects.

DRIFT_Y_PARS = dict(scale=2.1467,  # drift per degF (NOT degC as elsewhere in this module)
                    offset=-6.012,
                    trend=-1.108,
                    jumps=(('2015:006:12:00:00', -4.600),
                           ('2015:265:12:00:00', -4.669),
                           ('2016:064:12:00:00', -1.793),
                           ('2017:066:12:00:00', -1.725),
                           ('2018:285:12:00:00', -12.505)),
                    year0=2016.0)

DRIFT_Z_PARS = dict(scale=1.004,
                    offset=-15.963,
                    trend=-0.159,
                    jumps=(('2015:006:12:00:00', -2.109),
                           ('2015:265:12:00:00', -0.368),
                           ('2016:064:12:00:00', -0.902),
                           ('2017:066:12:00:00', -0.856),
                           ('2018:285:12:00:00', -6.056)),
                    year0=2016.0)

# Define transform from aspect solution DY, DZ (mm) to CHIPX, CHIPY for
# nominal target position.  This uses the transform:
# CHIPX, CHIPY = c0 + cyz * [DY, DZ]  (dot product)
# These values come from the aimpoint_mon/asol_to_chip_transforms notebook:
# https://github.com/sot/aimpoint_mon/blob/cebe5010/asol_to_chip_transforms.ipynb
ASOL_TO_CHIP = {('ACIS-I', 0): {'c0': [1100.806, 1110.299],
                                'cyz': [[0.001, 41.742], [-41.742, 0.105]]},
                ('ACIS-I', 1): {'c0': [-76.208, 962.413],
                                'cyz': [[0.001, -41.742], [41.741, 0.104]]},
                ('ACIS-I', 2): {'c0': [54.468, 1110.197],
                                'cyz': [[-0.001, 41.742], [-41.742, -0.105]]},
                ('ACIS-I', 3): {'c0': [970.795, 961.857],
                                'cyz': [[-0.001, -41.742], [41.742, -0.104]]},
                ('ACIS-S', 4): {'c0': [3382.364, 520.024],
                                'cyz': [[-41.695, -0.021], [0.021, -41.689]]},
                ('ACIS-S', 5): {'c0': [2338.96, 519.15],
                                'cyz': [[-41.691, -0.031], [0.037, -41.689]]},
                ('ACIS-S', 6): {'c0': [1296.05, 519.532],
                                'cyz': [[-41.69, -0.031], [0.036, -41.689]]},
                ('ACIS-S', 7): {'c0': [251.607, 519.818],
                                'cyz': [[-41.689, -0.022], [0.022, -41.689]]},
                ('ACIS-S', 8): {'c0': [-790.204, 519.784],
                                'cyz': [[-41.69, -0.02], [0.02, -41.689]]},
                ('ACIS-S', 9): {'c0': [-1832.423, 519.546],
                                'cyz': [[-41.693, -0.017], [0.017, -41.689]]},
                ('HRC-I', 0): {'c0': [7635.787, 7697.627],
                               'cyz': [[-109.98, 109.98], [109.98, 109.98]]},
                ('HRC-S', 1): {'c0': [2196.931, 25300.796],
                               'cyz': [[0.188, -155.535], [155.584, 0.19]]},
                ('HRC-S', 2): {'c0': [2197.753, 8842.306],
                               'cyz': [[0.2, -155.535], [155.535, 0.184]]},
                ('HRC-S', 3): {'c0': [2196.638, -7615.26],
                               'cyz': [[0.2, -155.535], [155.571, 0.184]]}}

SIM_MM_TO_ARCSEC = 20.493

# Cache for the zero offset table
CACHE = {}


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
        Calculate the drift model for aspect solution SIM DY/DZ values for input ``times``
        and ``t_ccd``.  The two arrays are broadcasted to match.

        The returned drifts are in arcsec and provide the expected aspect solution
        SIM DY or DZ values in mm.  This can be converted to a drift in arcsec via
        the scale factor 20.493 arcsec/mm.

        :param times: array of times (CXC secs)
        :param t_ccd: CCD temperatures (degC)
        :returns: array of ASOL SIM DY/DZ (mm)
        """
        # The drift model is calibrated assuming t_ccd is in degF, but we want inputs
        # in degC, so convert at this point.
        t_ccd = t_ccd * 1.8 + 32.0

        times, t_ccd = np.broadcast_arrays(times, t_ccd)
        is_scalar = times.ndim == 0 and t_ccd.ndim == 0
        times = DateTime(np.atleast_1d(times)).secs
        t_ccd = np.atleast_1d(t_ccd)

        if times.shape != t_ccd.shape:
            raise ValueError('times and t_ccd args must match in shape')

        if np.any(np.diff(times) < 0):
            raise ValueError('times arg must be monotonically increasing')

        if times[0] < DateTime('2012:001:12:00:00').secs:
            raise ValueError('model is not applicable before 2012')

        # Years from model `year0`
        dyears = DateTime(times, format='secs').frac_year - self.year0

        # Raw offsets without jumps
        out = (t_ccd - self.offset) * self.scale + dyears * self.trend

        # Put in the step function jumps
        for jump_date, jump in self.jumps:
            jump_idx = np.searchsorted(times, DateTime(jump_date).secs)
            out[jump_idx:] += jump

        return out[0] if is_scalar else out


# Define model instances using calibrated parameters
DRIFT_Y = AcaDriftModel(**DRIFT_Y_PARS)
DRIFT_Z = AcaDriftModel(**DRIFT_Z_PARS)


def get_aca_offsets(detector, chip_id, chipx, chipy, time, t_ccd):
    """
    Compute the dynamical ACA offset values for the provided inputs.

    The ``time`` and ``t_ccd`` inputs can be either scalars or arrays.

    :param detector: one of ACIS-I, ACIS-S, HRC-I, HRC-S
    :param chipx: zero-offset aimpoint CHIPX
    :param chipy: zero-offset aimpoint CHIPY
    :param chip_id: zero-offset aimpoint CHIP ID
    :param time: time(s) of observation (any Chandra.Time compatible format)
    :param t_ccd: ACA CCD temperature(s) (degC)

    :returns: aca_offset_y, aca_offset_z (arcsec)
    """
    try:
        asol_to_chip = ASOL_TO_CHIP[detector, chip_id]
    except KeyError:
        raise KeyError('Detector and chip combination {} not in allow values: {}'
                       .format((detector, chip_id), sorted(ASOL_TO_CHIP.keys())))

    # Compute the asol DY/DZ that would be required for the aimpoint to be
    # exactly at the desired CHIPX/Y values.  Uses the geometrical transform
    # computed via dmcoords in the asol_to_chip_transform notebook in this repo.
    # CHIPX, CHIPY = c0 + cyz * [DY, DZ]  (dot product)
    chip_xy = np.array([chipx, chipy])
    cyz_inv = np.linalg.inv(asol_to_chip['cyz'])
    dy_chip, dz_chip = cyz_inv.dot(chip_xy - asol_to_chip['c0']) * SIM_MM_TO_ARCSEC  # arcsec

    # Compute the predicted asol DY/DZ based on time and ACA CCD temperature
    # via the predictive model calibrated in the fit_aimpoint_drift notebook
    # in this repo.
    dy_pred = DRIFT_Y.calc(time, t_ccd)
    dz_pred = DRIFT_Z.calc(time, t_ccd)

    # The difference is the dynamic ACA offset that must be applied to the attitude.
    # This has the same sign convention as the user-supplied TARGET OFFSET in the
    # ObsCat / OR-list.
    ddy = dy_chip - dy_pred
    ddz = dz_chip - dz_pred

    return ddy, ddz


def get_default_zero_offset_table():
    """
    Get official SOT MP zero offset aimpoint table.

    If a local copy at '/data/mpcrit1/aimpoint_table/zero_offset_aimpoints.txt' is not found
    this uses the version at
    'https://cxc.harvard.edu/mta/ASPECT/drift/zero_offset_aimpoints.txt' which is updated
    via a ska cron job.

    :returns: zero offset aimpoint table as astropy.Table
    """
    if 'ZERO_OFFSET_TABLE' in CACHE:
        return CACHE['ZERO_OFFSET_TABLE']
    try:
        CACHE['ZERO_OFFSET_TABLE'] = Table.read(
            '/data/mpcrit1/aimpoint_table/zero_offset_aimpoints.txt',
            format='ascii')
    except FileNotFoundError:
        CACHE['ZERO_OFFSET_TABLE'] = Table.read(
            "https://cxc.harvard.edu/mta/ASPECT/drift/zero_offset_aimpoints.txt",
            format='ascii')
    return CACHE['ZERO_OFFSET_TABLE']


def get_target_aimpoint(date, cycle, detector, too=False, zero_offset_table=None):
    """
    Given date, proposal cycle, and detector, return aimpoint chipx, chipy, chip_id

    :param date: observation date
    :param cycle: proposal cycle of observation
    :param detector: target detector
    :param too: boolean. If target is TOO use current cycle not proposal cycle.
    :param zero_offset_able: table (astropy or numpy) of zero offset aimpoint table
                             defaults to official SOT MP version if not supplied.

    :returns: tuple of chipx, chipy, chip_id
    """
    if zero_offset_table is None:
        zero_offset_table = get_default_zero_offset_table()
    zero_offset_table.sort(['date_effective', 'cycle_effective'])
    date = DateTime(date).iso[:10]
    # Entries for this detector before the 'date' given
    ok = (zero_offset_table['detector'] == detector) & (zero_offset_table['date_effective'] <= date)
    # If a regular observation, the entry must also be before or equal to proposal cycle
    if not too:
        ok = ok & (zero_offset_table['cycle_effective'] <= cycle)
    filtered_table = zero_offset_table[ok]
    # Return the desired keys in the most recent [-1] row that matches
    return tuple(filtered_table[['chipx', 'chipy', 'chip_id']][-1])
