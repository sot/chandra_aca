# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The transform modules includes:

- Coordinate conversions between ACA Pixels and ACA Y-angle, Z-angle
- Mag to count conversions
- Science target coordinate to ACA frame conversions
"""

from __future__ import division

import numpy as np

from chandra_aca import dark_model
from Quaternion import Quat

# coefficients for converting from ACA angle to pixels (ground)
ACA2PIX_coeff = np.array(
    [[6.08840495576943,         4.92618563916467],
     [0.000376181788609776,     0.200203020554239],
     [-0.200249577514165,       0.000332284183255046],
     [-2.7052897180601e-009,    -5.35702097772202e-009],
     [9.75572638037165e-010,    1.91073314224894e-008],
     [-2.94865155316778e-008,   -4.85766581852866e-009],
     [8.31198018312775e-013,    2.01092070428495e-010],
     [-1.96043819238097e-010,    5.09721545876414e-016],
     [5.14134244771463e-013,    1.99339355492595e-010],
     [-1.97282476269237e-010,    2.52739834319184e-014]])

# coefficients for converting from pixels to ACA angle in radians
PIX2ACA_coeff = np.array(
    [[1.471572165932506e-04, -1.195271491928579e-04],
     [4.554462091388806e-08,  2.421478755190295e-05],
     [-2.420905844425065e-05, 4.005224006503938e-08],
     [-4.989939217701764e-12, 1.188134673090465e-11],
     [-6.116309500303049e-12, 1.832694593246024e-11],
     [-2.793916972292602e-11, 5.823266376976988e-12],
     [2.420403450703432e-16,  -5.923401659857833e-13],
     [5.751137659424387e-13,  -1.666025332027183e-15],
     [-9.934587837231732e-16, -5.847450395792513e-13],
     [5.807475081470944e-13,   -1.842673748068349e-15]])

# From /proj/sot/ska/analysis/aca_plate_scale/calib_prelaunch/eeprom_cal.txt
# This is the full 20-coefficient solution that includes a
# temperature dependence (but not a zero-order dependence).
# The ground coefficients do not include the terms that have
# temperature term.
PIX2ACA_eeprom = np.array(
    [[3.03464844e+01,  -2.46503906e+01],
     [9.38110352e-03,   4.99642792e+00],
     [-4.99528198e+00,  9.14611816e-03],
     [0.00000000e+00,   0.00000000e+00],
     [-1.14440918e-06,  2.53915787e-06],
     [-1.19805336e-06,  3.72529030e-06],
     [7.62343407e-06,  -1.25670433e-04],
     [-5.62667847e-06,  1.03116035e-06],
     [1.40523911e-04,  -1.25801563e-04],
     [0.00000000e+00,   0.00000000e+00],
     [4.65661287e-11,  -1.20745972e-07],
     [1.17020682e-07,  -3.25962901e-10],
     [1.06869265e-08,  -9.35979187e-09],
     [-2.09547579e-10, -1.18976459e-07],
     [-2.56113708e-09, -6.51925802e-09],
     [-4.74276021e-07, -1.41877681e-06],
     [1.18394382e-07,  -3.72529030e-10],
     [-3.70200723e-09,  1.03842467e-08],
     [5.06476499e-07,   4.46331687e-06]])

# Convert from arcsec to radians
PIX2ACA_eeprom = np.radians(PIX2ACA_eeprom / 3600)

ACA_MAG0 = 10.32
ACA_CNT_RATE_MAG0 = 5263.0

# Numerical values of the ODB_SI_ALIGN in C-order.  This is taken from the Matlab science
# characteristics and was originally generated by Bill Davis. It corresponds to the OFLS
# ODB_SI_ALIGN transform matrix from post-launch through Nov-2015, but it is orthonormal
# to machine precision.  (ODB_SI_ALIGN is expressed with only 5 digits of precision.)
ODB_SI_ALIGN = np.array([[0.999999905689160, -0.000337419984089, -0.000273439987106],
                         [0.000337419984089, 0.999999943073875, -0.000000046132060],
                         [0.000273439987106, -0.000000046132060, 0.999999962615285]])


def broadcast_arrays(*args):
    """
    Broadcast *args inputs to same shape and return an ``is_scalar`` flag and
    the broadcasted version of inputs.  This lets intermediate code work on
    arrays that are guaranteed to be the same shape and at least a 1-d array,
    but reshape the output at the end.

    :param args: tuple of scalar / array inputs
    :returns: [is_scalar, *flat_args]

    """
    is_scalar = all(np.array(arg).ndim == 0 for arg in args)
    args = np.atleast_1d(*args)
    outs = [is_scalar] + np.broadcast_arrays(*args)
    return outs


def broadcast_arrays_flatten(*args):
    """Broadcast *args inputs to same shape and then return that shape and the
    flattened view of all the inputs.  This lets intermediate code work on all
    scalars or all arrays that are the same-length 1-d array and then reshape
    the output at the end (if necessary).

    :param args: tuple of scalar / array inputs
    :returns: [shape, *flat_args]

    """
    is_scalar, *outs = broadcast_arrays(*args)
    if is_scalar:
        return [()] + list(args)

    shape = outs[0].shape
    outs = [out.ravel() for out in outs]
    return [shape] + outs


def pixels_to_yagzag(row, col, allow_bad=False, flight=False, t_aca=20,
                     pix_zero_loc='edge'):
    """
    Convert ACA row/column positions to ACA y-angle, z-angle.
    It is expected that the row and column input arguments have the same length.

    The ``pix_zero_loc`` parameter controls whether the input pixel values
    are assumed to have integral values at the pixel center or at the pixel
    lower/left edge.  The PEA flight coefficients assume lower/left edge, and
    that is the default.  However, it is generally more convenient when doing
    centroids and other manipulations to use the center.

    :param row: ACA pixel row (single value, list, or 1-d numpy array)
    :param col: ACA pixel column (single value, list, or 1-d numpy array)
    :param allow_bad: boolean switch.  If true, method will not throw errors
                         if the row/col values are nominally off the ACA CCD.
    :param flight: Use flight EEPROM coefficients instead of default ground values.
    :param t_aca: ACA temperature (degC) for use with flight (default=20C)
    :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'
    :rtype: (yang, zang) each vector of the same length as row/col
    """
    row = np.array(row)
    col = np.array(col)

    if pix_zero_loc == 'center':
        # Transform row/col values from 'center' convention to 'edge'
        # convention, which is required for use in _poly_convert below.
        row = row + 0.5
        col = col + 0.5
    elif pix_zero_loc != 'edge':
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    # Row/col are in edge coordinates at this point, check if they are on
    # the CCD unless allow_bad is True.
    if (not allow_bad and (np.any(np.abs(row) > 512.0) or
                           np.any(np.abs(col) > 512.0))):
        raise ValueError("Coordinate off CCD")

    coeff = PIX2ACA_eeprom if flight else PIX2ACA_coeff
    yrad, zrad = _poly_convert(row, col, coeff, t_aca)

    # Convert to arcsecs from radians
    return 3600 * np.degrees(yrad), 3600 * np.degrees(zrad)


def yagzag_to_pixels(yang, zang, allow_bad=False, pix_zero_loc='edge'):
    """
    Convert ACA y-angle/z-angle positions to ACA pixel row, column.
    It is expected that the y-angle/z-angle input arguments have the same length.

    The ``pix_zero_loc`` parameter controls whether the input pixel values
    are assumed to have integral values at the pixel center or at the pixel
    lower/left edge.  The PEA flight coefficients assume lower/left edge, and
    that is the default.  However, it is generally more convenient when doing
    centroids and other manipulations to use ``pix_zero_loc='center'``.

    :param yang: ACA y-angle (single value, list, or 1-d numpy array)
    :param zang: ACA z-angle (single value, list, or 1-d numpy array)
    :param allow_bad: boolean switch.  If true, method will not throw errors
                         if the resulting row/col values are nominally off the ACA CCD.
    :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'
    :rtype: (row, col) each vector of the same length as row/col
    """
    yang = np.array(yang)
    zang = np.array(zang)
    row, col = _poly_convert(yang, zang, ACA2PIX_coeff)

    # Row/col are in edge coordinates at this point, check if they are on
    # the CCD unless allow_bad is True.
    if (not allow_bad and (np.any(np.abs(row) > 512.0) or
                           np.any(np.abs(col) > 512.0))):
        raise ValueError("Coordinate off CCD")

    if pix_zero_loc == 'center':
        # Transform row/col values from 'edge' convention (as returned
        # by _poly_convert) to the 'center' convention requested by user.
        row = row - 0.5
        col = col - 0.5
    elif pix_zero_loc != 'edge':
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    return row, col


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
        poly = np.array([np.ones_like(y), z, y, zz, y * z, yy, zz * z, y * zz, yy * z, yy * y])

    elif len(coeffs) == 19:
        # Full temperature dependent equation
        c = z  # Use c and r corresponding to section 2.2.4 of ACA Equations document
        r = y
        ones = np.ones_like(c)
        t = t_aca * ones
        poly = np.array([ones, c, r, t, c*c, c*r, c*t, r*r, r*t, t*t, c*c*c, r*c*c,
                         t*c*c, c*r*r, c*r*t, c*t*t, r*r*r, t*r*r, r*t*t])
    else:
        raise ValueError('Bad coefficients, length != 10 or 19')

    newy = np.sum(coeffs[:, 0] * poly.transpose(), axis=-1)
    newz = np.sum(coeffs[:, 1] * poly.transpose(), axis=-1)
    if shape:
        newy.shape = shape
        newz.shape = shape

    return newy, newz


def radec_to_yagzag(ra, dec, q_att):
    """
    Given RA, Dec, and pointing quaternion, determine ACA Y-ang, Z-ang.  The
    input ``ra`` and ``dec`` values can be 1-d arrays in which case the output
    ``yag`` and ``zag`` will be corresponding arrays of the same length.

    This is a wrapper around Ska.quatutil.radec2yagzag but uses arcsec instead
    of deg for yag, zag.

    :param ra: Right Ascension (degrees)
    :param dec: Declination (degrees)
    :param q_att: ACA pointing quaternion

    :returns:  yag, zag (arcsec)
    """
    from Ska.quatutil import radec2yagzag
    yag, zag = radec2yagzag(ra, dec, q_att)
    yag *= 3600
    zag *= 3600

    return yag, zag


def yagzag_to_radec(yag, zag, q_att):
    """
    Given ACA Y-ang, Z-ang and pointing quaternion determine RA, Dec. The
    input ``yag`` and ``zag`` values can be 1-d arrays in which case the output
    ``ra`` and ``dec`` will be corresponding arrays of the same length.

    This is a wrapper around Ska.quatutil.yagzag2radec but uses arcsec instead
    of deg for yag, zag.

    :param yag: ACA Y angle (arcsec)
    :param zag: ACA Z angle (arcsec)
    :param q_att: ACA pointing quaternion

    :returns: ra, dec (arcsec)
    """
    from Ska.quatutil import yagzag2radec
    ra, dec = yagzag2radec(yag / 3600, zag / 3600, q_att)

    return ra, dec


def mag_to_count_rate(mag):
    """
    Convert ACA mag to count rate in e- / sec

    Based on $CALDB/data/chandra/pcad/ccd_char/acapD1998-12-14ccdN0002.fits
    columns mag0=10.32 and cnt_rate_mag0=5263.0.  To convert to raw telemetered
    counts (ADU), divide e- by the gain of 5.0 e-/ADU.

    :param mag: star magnitude in ACA mag
    :returns: count rate (e-/sec)
    """
    count_rate = ACA_CNT_RATE_MAG0 * 10.0 ** ((ACA_MAG0 - mag) / 2.5)
    return count_rate


def count_rate_to_mag(count_rate):
    """
    Convert count rate in e- / sec to ACA mag

    Based on $CALDB/data/chandra/pcad/ccd_char/acapD1998-12-14ccdN0002.fits
    columns mag0=10.32 and cnt_rate_mag0=5263.0.

    :param count_rate: count rate (e-/sec)
    :returns: magnitude (ACA mag)
    """
    mag = ACA_MAG0 - 2.5 * np.log10(count_rate / ACA_CNT_RATE_MAG0)
    return mag


def snr_mag_for_t_ccd(t_ccd, ref_mag, ref_t_ccd, scale_4c=None):
    """
    Given a t_ccd, solve for the magnitude that has the same expected signal
    to noise as ref_mag / ref_t_ccd.

    If scale_4c is None, the value from dark_model.DARK_SCALE_4C is used.

    To solve for the magnitude that has the expected signal to noise as
    ref_mag / ref_t_ccd, we define this equality:

    counts(mag) / noise(t_ccd) = counts(ref_mag) / noise(ref_t_ccd)

    We then assume (with some handwaving) that:

    noise(t_ccd) = noise(ref_t_ccd) * scale

    where

    scale = scale_4c ** ((t_ccd - ref_t_ccd) / 4.0)

    And we use the definition of counts as:

    counts(mag) = ACA_CNT_RATE_MAG0 * 10.0 ** ((ACA_MAG0 - mag) / 2.5)

    Substituting in, reducing, and solving the original equality for mag gives

    ref_mag - (t_ccd - ref_t_ccd) * np.log10(scale_4c) / 1.6

    """
    if scale_4c is None:
        scale_4c = dark_model.DARK_SCALE_4C
    return ref_mag - (t_ccd - ref_t_ccd) * np.log10(scale_4c) / 1.6


def calc_aca_from_targ(targ, y_off, z_off, si_align=None):
    """
    Calculate the PCAD (ACA) pointing attitude from target attitude and
    Y,Z offset (per OR-list).

    :param targ: science target from OR/Obscat (Quat-compliant object)
    :param y_off: Y offset (deg, sign per OR-list convention)
    :param z_off: Z offset (deg, sign per OR-list convention)
    :param si_align: SI ALIGN matrix (default=ODB_SI_ALIGN)

    :rtype: q_aca (Quat)
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
    Calculate the target attitude from ACA (PCAD) pointing attitude and
    Y,Z offset (per OR-list).

    :param aca: ACA (PCAD) pointing attitude (any Quat-compliant object)
    :param y_off: Y offset (deg, sign per OR-list convention)
    :param z_off: Z offset (deg, sign per OR-list convention)
    :param si_align: SI ALIGN matrix

    :rtype: q_targ (Quat)
    """
    if si_align is None:
        si_align = ODB_SI_ALIGN

    q_si_align = Quat(si_align)
    q_aca = Quat(aca)
    q_off = Quat([y_off, z_off, 0])

    q_targ = q_aca * q_si_align * q_off

    return q_targ
