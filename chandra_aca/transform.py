# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The transform modules includes:

- Coordinate conversions between ACA Pixels and ACA Y-angle, Z-angle
- Mag to count conversions
- Science target coordinate to ACA frame conversions
"""

import itertools
import os

import numba
import numpy as np
from Quaternion import Quat

from chandra_aca import dark_model

###################################################################################
# Legacy coefficients to avoid regression diffs in other packages in cases where
# updating the tests is inconvenient.
###################################################################################

# coefficients for converting from ACA angle to pixels (ground)
ACA2PIX_coeff = np.array(
    [
        [6.08840495576943, 4.92618563916467],
        [0.000376181788609776, 0.200203020554239],
        [-0.200249577514165, 0.000332284183255046],
        [-2.7052897180601e-009, -5.35702097772202e-009],
        [9.75572638037165e-010, 1.91073314224894e-008],
        [-2.94865155316778e-008, -4.85766581852866e-009],
        [8.31198018312775e-013, 2.01092070428495e-010],
        [-1.96043819238097e-010, 5.09721545876414e-016],
        [5.14134244771463e-013, 1.99339355492595e-010],
        [-1.97282476269237e-010, 2.52739834319184e-014],
    ]
)

# coefficients for converting from pixels to ACA angle in radians
PIX2ACA_coeff = np.array(
    [
        [1.471572165932506e-04, -1.195271491928579e-04],
        [4.554462091388806e-08, 2.421478755190295e-05],
        [-2.420905844425065e-05, 4.005224006503938e-08],
        [-4.989939217701764e-12, 1.188134673090465e-11],
        [-6.116309500303049e-12, 1.832694593246024e-11],
        [-2.793916972292602e-11, 5.823266376976988e-12],
        [2.420403450703432e-16, -5.923401659857833e-13],
        [5.751137659424387e-13, -1.666025332027183e-15],
        [-9.934587837231732e-16, -5.847450395792513e-13],
        [5.807475081470944e-13, -1.842673748068349e-15],
    ]
)

# From /proj/sot/ska/analysis/aca_plate_scale/calib_prelaunch/eeprom_cal.txt
# This is the full 20-coefficient solution that includes a
# temperature dependence (but not a zero-order dependence).
# The ground coefficients do not include the terms that have
# temperature term.
PIX2ACA_eeprom_arcsec = np.array(
    [
        [30.3464844, -24.6503906],  # location of (0, 0) pixel
        [9.38110352e-03, 4.99642792e00],
        [-4.99528198e00, 9.14611816e-03],
        [0.00000000e00, 0.00000000e00],
        [-1.14440918e-06, 2.53915787e-06],
        [-1.19805336e-06, 3.72529030e-06],
        [7.62343407e-06, -1.25670433e-04],
        [-5.62667847e-06, 1.03116035e-06],
        [1.40523911e-04, -1.25801563e-04],
        [0.00000000e00, 0.00000000e00],
        [4.65661287e-11, -1.20745972e-07],
        [1.17020682e-07, -3.25962901e-10],
        [1.06869265e-08, -9.35979187e-09],
        [-2.09547579e-10, -1.18976459e-07],
        [-2.56113708e-09, -6.51925802e-09],
        [-4.74276021e-07, -1.41877681e-06],
        [1.18394382e-07, -3.72529030e-10],
        [-3.70200723e-09, 1.03842467e-08],
        [5.06476499e-07, 4.46331687e-06],
    ]
)

# Convert from arcsec to radians
PIX2ACA_eeprom = np.radians(PIX2ACA_eeprom_arcsec / 3600)

# Ground (aspect pipeline) coefficients for converting from pixels (row, col) to ACA
# angle (yag, zag) (arcsec). See:
# https://nbviewer.org/urls/cxc.cfa.harvard.edu/mta/ASPECT/aca_plate_scale/calib_2020/fit-yz-plate-scale-no-rotation.ipynb
# https://cxc.cfa.harvard.edu/mta/ASPECT/aca_plate_scale/calib_2020/
#  0 [ones,
#  1 c,
#  2 r,
#  3 t,  <!! Leave at 0
#  4 c * c,
#  5 c * r,
#  6 c * t,  <= Temperature
#  7 r * r,
#  8 r * t,  <= Temperature
#  9 t * t,  <!! Leave at 0
# 10 c * c * c,
# 11 r * c * c,
# 12 t * c * c,  <= Temperature
# 13 c * r * r,
# 14 c * r * t,  <= Temperature
# 15 c * t * t,  <= Temperature
# 16 r * r * r,
# 17 t * r * r,  <= Temperature
# 18 r * t * t,  <= Temperature
#
# NOTE: the original notebook includes a 20th row `t * t * t` with a value of 0.0, but
# this is not consistent with flight or _poly_convert so we just drop it.
# This has also been adjusted as follows:
# - Convert from deg to arcsec
# - Add (+2.5, -2.5) arcsec to the zero point values. The original analysis used center
# row/col to match the aspect pipeline, but this code assumes edge row/col. Compare to
# the FLIGHT coeeff [30.3464844, -24.6503906].
PIX_TO_ANG_GROUND = np.array(
    [
        [30.3464845, -24.6503911],  # yag/zag location of row/col (0, 0) (arcsec)
        [8.23205783e-03, 4.99591058],  # linear plate scale arcsec / pixel in col
        [-4.99553676, 9.23841252e-03],  # linear plate scale arcsec / pixel in row
        [0.00000000e00, 0.00000000e00],
        [-1.37602599e-06, 2.42501739e-06],
        [-1.09954525e-06, 4.56650996e-06],
        [-2.90995528e-07, -1.39358988e-04],
        [-6.20856605e-06, 1.25436052e-06],
        [1.54786392e-04, -1.17830290e-04],
        [0.00000000e00, 0.00000000e00],
        [5.48539877e-10, -1.20837365e-07],
        [1.17004831e-07, -3.32899187e-10],
        [1.18654095e-08, -5.22646769e-09],
        [-1.47271099e-10, -1.17362370e-07],
        [-5.27874970e-09, -6.02546684e-09],
        [5.17812325e-08, -3.09403378e-07],
        [1.18612895e-07, -3.38209660e-10],
        [-3.82947347e-09, 9.81968414e-09],
        [-3.87317729e-07, 3.94460903e-06],
    ],
).transpose()

# Coefficients for converting from ACA angle (yag, zag) (arcsec) to pixel (row, col)
# This uses the final `coeffs_fit_all` coefficients from
# https://nbviewer.org/urls/cxc.harvard.edu/mta/ASPECT/aca_plate_scale/calib_2020/fit-rc-plate-scale.ipynb.
# In that notebook the actual coefficient values were not printed, but it was re-run and
# these are the values.
#  0 [ones,
#  1 z,
#  2 y,
#  3 t,  <!! Leave at 0
#  4 z * z,
#  5 z * y,
#  6 z * t,  <= Temperature
#  7 y * y,
#  8 y * t,  <= Temperature
#  9 t * t,  <!! Leave at 0
# 10 z * z * z,
# 11 y * z * z,
# 12 t * z * z,  <= Temperature
# 13 z * y * y,
# 14 z * y * t,  <= Temperature
# 15 z * t * t,  <= Temperature
# 16 y * y * y,
# 17 t * y * y,  <= Temperature
# 18 y * t * t,  <= Temperature
# 19 t * t * t]  <!! Leave at 0
ANG_TO_PIX_GROUND = np.array(
    [
        [6.08840496, 4.92618564],  # pixel location of (0, 0) yag, zag
        [3.85885523e-04, 0.200184894],  # linear plate scale pixels / arcsec in zag
        [-0.200224551, 3.85458793e-04],  # linear plate scale pixels / arcsec in yag
        [0.00000000e00, 0.00000000e00],
        [-5.83863054e-09, -1.43156559e-08],
        [-3.82253195e-09, 2.62976678e-08],
        [-6.64135115e-06, 1.48724550e-06],
        [-3.54580684e-08, -6.04764636e-09],
        [8.62191298e-07, -5.99108232e-06],
        [0.00000000e00, 0.00000000e00],
        [1.06571736e-12, 2.01178012e-10],
        [-1.96863305e-10, 4.19981450e-13],
        [1.18393143e-10, 3.80110516e-10],
        [6.45978452e-13, 1.96738124e-10],
        [2.85962857e-10, 2.03208326e-11],
        [3.96857730e-07, 1.01919863e-07],
        [-1.98850298e-10, -1.84355310e-12],
        [-7.15660261e-12, 3.04028795e-12],
        [-1.54074493e-07, 3.94104327e-07],
    ],
).transpose()

# Define flight pixel to angle coefficents with a name that is consistent with new
# coefficients from 2020 plate scale ground calibration coefficients.
PIX_TO_ANG_FLIGHT = PIX2ACA_eeprom_arcsec.transpose()

ACA_MAG0 = 10.32
ACA_CNT_RATE_MAG0 = 5263.0

# Numerical values of the ODB_SI_ALIGN in C-order.  This is taken from the Matlab science
# characteristics and was originally generated by Bill Davis. It corresponds to the OFLS
# ODB_SI_ALIGN transform matrix from post-launch through Nov-2015, but it is orthonormal
# to machine precision.  (ODB_SI_ALIGN is expressed with only 5 digits of precision.)
ODB_SI_ALIGN = np.array(
    [
        [0.999999905689160, -0.000337419984089, -0.000273439987106],
        [0.000337419984089, 0.999999943073875, -0.000000046132060],
        [0.000273439987106, -0.000000046132060, 0.999999962615285],
    ]
)


def broadcast_arrays(*args):
    r"""
    Broadcast ``*args`` inputs to same shape.

    Broadcast ``*args`` inputs to same shape and return an ``is_scalar`` flag and
    the broadcasted version of inputs.  This lets intermediate code work on
    arrays that are guaranteed to be the same shape and at least a 1-d array,
    but reshape the output at the end.

    :param args: tuple of scalar / array inputs
    :returns: [is_scalar, \*flat_args]

    """
    is_scalar = all(np.asarray(arg).ndim == 0 for arg in args)
    args = np.atleast_1d(*args)
    outs = [is_scalar] + list(np.broadcast_arrays(*args))
    return outs


def broadcast_arrays_flatten(*args):
    r"""Broadcast ``*args`` inputs to same shape and flatten.

    Broadcast ``*args`` inputs to same shape.and then return that shape and the
    flattened view of all the inputs.  This lets intermediate code work on all
    scalars or all arrays that are the same-length 1-d array and then reshape
    the output at the end (if necessary).

    :param args: tuple of scalar / array inputs
    :returns: [shape, \*flat_args]

    """
    is_scalar, *outs = broadcast_arrays(*args)
    if is_scalar:
        return [()] + list(args)

    shape = outs[0].shape
    outs = [out.ravel() for out in outs]
    return [shape] + outs


def pixels_to_yagzag(
    row, col, *, allow_bad=False, flight=False, t_aca=20.0, pix_zero_loc="edge"
):
    """
    Convert ACA row/column positions to ACA y-angle, z-angle.

    It is expected that the row and column input arguments have the same length.

    The ``pix_zero_loc`` parameter controls whether the input pixel values
    are assumed to have integral values at the pixel center or at the pixel
    lower/left edge.  The PEA flight coefficients assume lower/left edge, and
    that is the default.  However, it is generally more convenient when doing
    centroids and other manipulations to use the center.

    Parameters
    ----------
    row
        ACA pixel row (single value, list, or 1-d numpy array)
    col
        ACA pixel column (single value, list, or 1-d numpy array)
    allow_bad : boolean switch.  If true, method will not throw errors
        if the row/col values are nominally off the ACA CCD.
    flight
        Use flight EEPROM coefficients instead of default ground values.
    t_aca
        ACA temperature (degC) for use with flight (default=20C)
    pix_zero_loc
        row/col coords are integral at 'edge' or 'center'

    Returns
    -------
    (yang, zang) each vector of the same length as row/col
    """
    use_legacy = "CHANDRA_ACA_TRANSFORM_USE_LEGACY_COEFFS" in os.environ

    row = np.asarray(row, dtype=np.float64)
    col = np.asarray(col, dtype=np.float64)

    if pix_zero_loc == "center":
        # Transform row/col values from 'center' convention to 'edge'
        # convention, which is required for use in _poly_convert below.
        row = row + 0.5
        col = col + 0.5
    elif pix_zero_loc != "edge":
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    # Row/col are in edge coordinates at this point, check if they are on
    # the CCD unless allow_bad is True.
    if not allow_bad and (np.any(np.abs(row) > 512.0) or np.any(np.abs(col) > 512.0)):
        raise ValueError("Coordinate off CCD")

    if use_legacy:
        # Use legacy coefficients (19 x 2 or 10 x 2) which are in radians
        coeff = (PIX2ACA_eeprom if flight else PIX2ACA_coeff) * 3600 * np.degrees(1.0)
        yang, zang = _poly_convert(row, col, coeff, t_aca)
    else:
        # Use 2020 coefficients (2 x 19)
        coeff = PIX_TO_ANG_FLIGHT if flight else PIX_TO_ANG_GROUND
        yang = _poly_convert_numba(row, col, coeff[0], t_aca)
        zang = _poly_convert_numba(row, col, coeff[1], t_aca)

    return yang, zang


def yagzag_to_pixels(
    yang,
    zang,
    *,
    allow_bad=False,
    t_aca=20,
    pix_zero_loc="edge",
    flight=False,
):
    """
    Convert ACA y-angle/z-angle positions to ACA pixel row, column.

    It is expected that the y-angle/z-angle input arguments have the same length.

    The ``pix_zero_loc`` parameter controls whether the input pixel values
    are assumed to have integral values at the pixel center or at the pixel
    lower/left edge.  The PEA flight coefficients assume lower/left edge, and
    that is the default.  However, it is generally more convenient when doing
    centroids and other manipulations to use ``pix_zero_loc='center'``.

    Parameters
    ----------
    yang
        ACA y-angle (single value, list, or 1-d numpy array)
    zang
        ACA z-angle (single value, list, or 1-d numpy array)
    allow_bad : boolean switch.  If true, method will not throw errors
        if the resulting row/col values are nominally off the ACA CCD.
    pix_zero_loc
        row/col coords are integral at 'edge' or 'center'

    Returns
    -------
    (row, col) each vector of the same length as row/col
    """
    yang = np.asarray(yang, dtype=np.float64)
    zang = np.asarray(zang, dtype=np.float64)
    use_legacy = "CHANDRA_ACA_TRANSFORM_USE_LEGACY_COEFFS" in os.environ

    if use_legacy:
        row, col = _poly_convert(yang, zang, ACA2PIX_coeff)
    else:
        row, col = _yagzag_to_pixels_by_inversion(
            yang, zang, t_aca=t_aca, flight=flight
        )

    # Row/col are in edge coordinates at this point, check if they are on
    # the CCD unless allow_bad is True.
    if not allow_bad and (np.any(np.abs(row) > 512.0) or np.any(np.abs(col) > 512.0)):
        raise ValueError("Coordinate off CCD")

    if pix_zero_loc == "center":
        # Transform row/col values from 'edge' convention (as returned
        # by _poly_convert) to the 'center' convention requested by user.
        row -= 0.5
        col -= 0.5
    elif pix_zero_loc != "edge":
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    return row, col


def _yagzag_to_pixels_by_inversion(yang, zang, t_aca, flight):
    """Use scipy.optimize.minimize to transform yang/zang to row/col

    This uses optimization to directly invert the transform.pixels_to_yagzag function.
    """
    from scipy.optimize import minimize

    shape, yangs, zangs = broadcast_arrays_flatten(yang, zang)
    yangs = np.atleast_1d(yangs).astype(np.float64)
    zangs = np.atleast_1d(zangs).astype(np.float64)

    # Starting values for minimization
    row0s = _poly_convert_numba(yangs, zangs, ANG_TO_PIX_GROUND[0], t_aca=t_aca)
    col0s = _poly_convert_numba(yangs, zangs, ANG_TO_PIX_GROUND[1], t_aca=t_aca)
    rows = np.empty_like(yangs)
    cols = np.empty_like(zangs)
    for ii, row0, col0, yang0, zang0 in zip(
        itertools.count(), row0s, col0s, yangs, zangs
    ):

        def min_func(rc):
            r, c = rc
            # Could optimize to avoid validation / type munging
            y, z = pixels_to_yagzag(r, c, flight=flight, t_aca=t_aca, allow_bad=True)
            return (y - yang0) ** 2 + (z - zang0) ** 2  # noqa: B023

        res = minimize(min_func, x0=[row0, col0], method="Nelder-Mead")
        rows[ii] = res.x[0]
        cols[ii] = res.x[1]

    if shape:
        rows.shape = shape
        cols.shape = shape
        return rows, cols
    else:
        return rows[0], cols[0]


def _poly_convert(y, z, coeffs, t_aca=None):
    # Convert to avoid overflow errors with the polys on int32
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if y.size != z.size:
        raise ValueError("Mismatched number of Y/Z coords")

    shape, y, z = broadcast_arrays_flatten(y, z)

    if len(coeffs) == 10:
        # No temperature dependence
        yy = y * y
        zz = z * z
        poly = np.array(
            [np.ones_like(y), z, y, zz, y * z, yy, zz * z, y * zz, yy * z, yy * y]
        )

    elif len(coeffs) == 19:
        # Full temperature dependent equation
        c = z  # Use c and r corresponding to section 2.2.4 of ACA Equations document
        r = y
        ones = np.ones_like(c)
        t = t_aca * ones
        poly = np.array(
            [
                ones,
                c,
                r,
                t,
                c * c,
                c * r,
                c * t,
                r * r,
                r * t,
                t * t,
                c * c * c,
                r * c * c,
                t * c * c,
                c * r * r,
                c * r * t,
                c * t * t,
                r * r * r,
                t * r * r,
                r * t * t,
            ]
        )
    else:
        raise ValueError("Bad coefficients, length != 10 or 19")

    newy = np.sum(coeffs[:, 0] * poly.transpose(), axis=-1)
    newz = np.sum(coeffs[:, 1] * poly.transpose(), axis=-1)
    if shape:
        newy.shape = shape
        newz.shape = shape

    return newy, newz


@numba.njit(cache=True)
def _poly_convert_numba_jit(
    r: np.ndarray,
    c: np.ndarray,
    coeffs: np.ndarray,
    t_aca: float,
) -> np.ndarray:
    out = np.empty(r.size, dtype=np.float64)

    for ii in range(r.size):
        rr = r[ii]
        cc = c[ii]
        t = t_aca

        out[ii] = (
            coeffs[0]
            + coeffs[1] * cc
            + coeffs[2] * rr
            + coeffs[3] * t
            + coeffs[4] * cc * cc
            + coeffs[5] * cc * rr
            + coeffs[6] * cc * t
            + coeffs[7] * rr * rr
            + coeffs[8] * rr * t
            + coeffs[9] * t * t
            + coeffs[10] * cc * cc * cc
            + coeffs[11] * rr * cc * cc
            + coeffs[12] * t * cc * cc
            + coeffs[13] * cc * rr * rr
            + coeffs[14] * cc * rr * t
            + coeffs[15] * cc * t * t
            + coeffs[16] * rr * rr * rr
            + coeffs[17] * t * rr * rr
            + coeffs[18] * rr * t * t
        )

    return out


def _poly_convert_numba(
    r: float | np.ndarray[float],
    c: float | np.ndarray[float],
    coeffs: np.ndarray[float],
    t_aca: float = 20.0,
):
    """Convert r, c values using a polynomial conversion via numba.

    Parameters
    ----------
    r
        row values (float or array of floats)
    c
        column values (float or array of floats)
    coeffs
        polynomial coefficients (fixed at 19 elements)
    t_aca
        temperature for use with flight coefficients (degC)

    Returns
    -------
    np.ndarray
        Converted values (same shape as input r and c)
    """
    # Convert to avoid overflow errors with the polys on int32
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if not isinstance(t_aca, float):
        t_aca = float(t_aca)

    if r.size != c.size:
        raise ValueError("size mismatch")

    shape = r.shape
    vals = _poly_convert_numba_jit(r.ravel(), c.ravel(), coeffs, t_aca)
    if shape:
        vals.shape = shape
    else:
        vals = vals[0]

    return vals


def radec_to_eci(ra, dec):
    """
    Convert from RA,Dec to ECI.

    The input ``ra`` and ``dec`` values can be 1-d arrays of length N in which
    case the output ``ECI`` will be an array with shape (N, 3). The N dimension
    can actually be any multidimensional shape.

    Parameters
    ----------
    ra
        Right Ascension (degrees)
    dec
        Declination (degrees)

    Returns
    -------
    numpy array ECI (3-vector or N x 3 array)
    """
    r = np.deg2rad(ra)
    d = np.deg2rad(dec)
    out = np.broadcast(r, d)

    out = np.empty(out.shape + (3,), dtype=np.float64)
    out[..., 0] = np.cos(r) * np.cos(d)
    out[..., 1] = np.sin(r) * np.cos(d)
    out[..., 2] = np.sin(d)
    return out


def eci_to_radec(eci):
    """
    Convert from ECI vector(s) to RA, Dec.

    The input ``eci`` value can be an array of 3-vectors having shape (N, 3)
    in which case the output RA, Dec will be arrays of shape N. The N dimension
    can actually be any multidimensional shape.

    Parameters
    ----------
    eci
        ECI as 3-vector or (N, 3) array

    Returns
    -------
    scalar or array ra, dec (degrees)
    """
    eci = np.asarray(eci)
    if eci.shape[-1] != 3:
        raise ValueError("final dimension of `eci` must be 3")

    ra = np.degrees(np.arctan2(eci[..., 1], eci[..., 0]))
    dec = np.degrees(
        np.arctan2(eci[..., 2], np.sqrt(eci[..., 1] ** 2 + eci[..., 0] ** 2))
    )
    # Enforce strictly 0 <= RA < 360. Note weird corner case that one can get
    # ra being negative and very small, e.g. -7.7e-18, which then has 360 added
    # and turns into exactly 360.0 because of floating point precision.
    bad = ra < 0
    if eci.ndim > 1:
        ra[bad] += 360
    elif bad:
        ra += 360

    bad = ra >= 360
    if eci.ndim > 1:
        ra[bad] -= 360
    elif bad:
        ra -= 360

    return ra, dec


def mag_to_count_rate(mag):
    """
    Convert ACA mag to count rate in e- / sec

    Based on $CALDB/data/chandra/pcad/ccd_char/acapD1998-12-14ccdN0002.fits
    columns mag0=10.32 and cnt_rate_mag0=5263.0.  To convert to raw telemetered
    counts (ADU), divide e- by the gain of 5.0 e-/ADU.

    Parameters
    ----------
    mag
        star magnitude in ACA mag

    Returns
    -------
    count rate (e-/sec)
    """
    count_rate = ACA_CNT_RATE_MAG0 * 10.0 ** ((ACA_MAG0 - mag) / 2.5)
    return count_rate


def count_rate_to_mag(count_rate):
    """
    Convert count rate in e- / sec to ACA mag

    Based on $CALDB/data/chandra/pcad/ccd_char/acapD1998-12-14ccdN0002.fits
    columns mag0=10.32 and cnt_rate_mag0=5263.0.

    Parameters
    ----------
    count_rate
        count rate (e-/sec)

    Returns
    -------
    magnitude (ACA mag)
    """
    mag = ACA_MAG0 - 2.5 * np.log10(count_rate / ACA_CNT_RATE_MAG0)
    return mag


def snr_mag_for_t_ccd(t_ccd, ref_mag, ref_t_ccd, scale_4c=None):
    """
    Calculate signal-to-noise equivalent magnitude.

    Given a t_ccd, solve for the magnitude that has the same expected signal
    to noise as ref_mag / ref_t_ccd.

    If scale_4c is None, the value from dark_model.DARK_SCALE_4C is used.

    To solve for the magnitude that has the expected signal to noise as
    ref_mag / ref_t_ccd, we use the following derivation::

      counts(mag) / noise(t_ccd) = counts(ref_mag) / noise(ref_t_ccd)
      noise(t_ccd) = noise(ref_t_ccd) * scale
      scale = scale_4c ** ((t_ccd - ref_t_ccd) / 4.0)

    Using the definition of counts as a function of magnitude, substituting in,
    reducing, and solving the original equality for mag gives::

      counts(mag) = ACA_CNT_RATE_MAG0 * 10.0 ** ((ACA_MAG0 - mag) / 2.5)
      ref_mag = (t_ccd - ref_t_ccd) * np.log10(scale_4c) / 1.6

    Parameters
    ----------
    t_ccd : float, array
        CCD temperature (degC)
    ref_mag : float, array
        Reference magnitude (mag)
    ref_t_ccd : float, array
        Reference CCD temperature (degC)
    scale_4c : float
        Scale factor for a 4 degC change in CCD temperature (defaults to DARK_SCALE_4C)

    Returns
    -------
    float, array
    Magnitude(s) with the same expected signal to noise as ref_mag at ref_t_ccd
    """
    # Allow array inputs
    t_ccd, ref_mag, ref_t_ccd = np.broadcast_arrays(t_ccd, ref_mag, ref_t_ccd)

    if scale_4c is None:
        scale_4c = dark_model.DARK_SCALE_4C
    return ref_mag - (t_ccd - ref_t_ccd) * np.log10(scale_4c) / 1.6


def calc_aca_from_targ(targ, y_off, z_off, si_align=None):
    """
    Calculate PCAD (ACA) pointing attitude.

    Calculate the PCAD (ACA) pointing attitude from target attitude and
    Y,Z offset (per OR-list).

    Parameters
    ----------
    targ
        science target from OR/Obscat (Quat-compliant object)
    y_off
        Y offset (deg, sign per OR-list convention)
    z_off
        Z offset (deg, sign per OR-list convention)
    si_align
        SI ALIGN matrix (default=ODB_SI_ALIGN)

    Returns
    -------
    q_aca (Quat)
    """
    if si_align is None:
        si_align = ODB_SI_ALIGN

    q_si_align = Quat(si_align)
    q_targ = Quat(targ)
    q_off = Quat([y_off, z_off, 0])
    q_aca = q_targ * q_off.inv() * q_si_align.inv()

    return q_aca


def calc_targ_from_aca(aca, y_off, z_off, si_align=None):
    """
    Calculate target attitude.

    Calculate the target attitude from ACA (PCAD) pointing attitude and
    Y,Z offset (per OR-list).

    Parameters
    ----------
    aca
        ACA (PCAD) pointing attitude (any Quat-compliant object)
    y_off
        Y offset (deg, sign per OR-list convention)
    z_off
        Z offset (deg, sign per OR-list convention)
    si_align
        SI ALIGN matrix

    Returns
    -------
    q_targ (Quat)
    """
    if si_align is None:
        si_align = ODB_SI_ALIGN

    q_si_align = Quat(si_align)
    q_aca = Quat(aca)
    q_off = Quat([y_off, z_off, 0])

    q_targ = q_aca * q_si_align * q_off

    return q_targ


def calc_target_offsets(aca, ra_targ, dec_targ, si_align=None):
    """
    Calculate target offsets.

    Calculates required Y and Z offsets (deg) required from a target RA, Dec to
    arrive at the desired PCAD pointing ``aca``.

    Parameters
    ----------
    aca
        PCAD attitude (any Quat-compliant initializer)
    ra_targ
        RA of science target from OR/Obscat
    dec_targ
        Dec of science target from OR/Obscat
    si_align
        SI ALIGN matrix (default=ODB)

    Returns
    -------
    tuple (y_off, z_off) degrees
    """
    if si_align is None:
        si_align = ODB_SI_ALIGN

    # Convert si_align transform matrix into a Quaternion
    q_si_align = Quat(si_align)

    # Pointing quaternion
    q_aca = Quat(aca)

    # Pointing quaternion of nominal HRMA frame after adjusting for the alignment offset.
    # The sense of si_align is that q_hrma = q_pcad * si_align, where si_align is
    # effectively a delta quaternion.
    q_hrma = q_aca * q_si_align

    # the y and z offsets of the target in that frame, in degrees
    y_off, z_off = radec_to_yagzag(ra_targ, dec_targ, q_hrma)

    return y_off / 3600, z_off / 3600


def radec_to_yagzag(ra, dec, q_att):
    """
    Calculate ACA Y-angle, Z-angle from RA, Dec and pointing quaternion.

    The input ``ra`` and ``dec`` values can be 1-d arrays in which case the output
    ``yag`` and ``zag`` will be corresponding arrays of the same length.

    Parameters
    ----------
    ra
        Right Ascension (degrees)
    dec
        Declination (degrees)
    q_att
        ACA pointing quaternion (Quat or Quat-compatible input)

    Returns
    -------
    yag, zag (arcsec)
    """
    if not isinstance(q_att, Quat):
        q_att = Quat(q_att)
    eci = radec_to_eci(ra, dec)  # N x 3
    qt = q_att.transform.swapaxes(-2, -1)  # Transpose, allowing for leading dimensions
    d_aca = np.einsum("...jk,...k->...j", qt, eci)
    yag = np.rad2deg(np.arctan2(d_aca[..., 1], d_aca[..., 0]))
    zag = np.rad2deg(np.arctan2(d_aca[..., 2], d_aca[..., 0]))
    return yag * 3600, zag * 3600


def yagzag_to_radec(yag, zag, q_att):
    """
    Calculate RA, Dec from ACA Y-angle, Z-angle and pointing quaternion.

    The input ``yag`` and ``zag`` values can be 1-d arrays in which case the output
    ``ra`` and ``dec`` will be corresponding arrays of the same length.

    Parameters
    ----------
    yag
        ACA Y angle (arcsec)
    zag
        ACA Z angle (arcsec)
    q_att
        ACA pointing quaternion (Quat or Quat-compatible input)

    Returns
    -------
    ra, dec (degrees)
    """
    if not isinstance(q_att, Quat):
        q_att = Quat(q_att)
    yag = np.asarray(yag)
    zag = np.asarray(zag)
    out = np.broadcast(yag, zag)  # Object with the right broadcasted shape
    d_aca = np.empty(out.shape + (3,), dtype=np.float64)
    d_aca[..., 0] = np.ones(out.shape)
    d_aca[..., 1] = np.tan(np.deg2rad(yag / 3600))
    d_aca[..., 2] = np.tan(np.deg2rad(zag / 3600))
    d_aca = normalize_vector(d_aca)
    eci = np.einsum("...jk,...k->...j", q_att.transform, d_aca)
    return eci_to_radec(eci)


def normalize_vector(vec, ord=None):
    """Normalize ``vec`` over the last dimension.

    For an L x M x N input array, this normalizes over the N dimension
    using ``np.linalg.norm``.

    Parameters
    ----------
    vec
        input vector or array of vectors
    ord
        ord parameter for np.linalg.norm (default=None => 2-norm)

    Returns
    -------
    normed array of vectors
    """
    norms = np.linalg.norm(vec, axis=-1, keepdims=True, ord=ord)
    return vec / norms
