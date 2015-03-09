"""
Chandra.ACA provides convenience methods for the Aspect Camera Assembly.  
It now includes coordinate conversions between ACA Pixels and ACA Y-angle, Z-angle.
"""

from __future__ import division
import numpy as np

# coefficients for converting from ACA angle to pixels
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

def pixels_to_yagzag(row, col, allow_bad=False):       
    """
    Convert ACA row/column positions to ACA y-angle, z-angle.  
    It is expected that the row and column input arguments have the same length.

    :param row: ACA pixel row (single value, list, or 1-d numpy array)
    :param col: ACA pixel column (single value, list, or 1-d numpy array)
    :param allow_bad: boolean switch.  If true, method will not throw errors
                         if the row/col values are nominally off the ACA CCD.
    :rtype: (yang, zang) each vector of the same length as row/col
    """
    row = np.array(row)
    col = np.array(col)
    if (not allow_bad and
        (np.any(row > 511.5) or np.any(row < -512.5)
         or np.any(col > 511.5) or np.any(col < -512.5))):
        raise ValueError("Coordinate off CCD")
    yrad, zrad = _poly_convert(row, col, PIX2ACA_coeff)
    # convert to arcsecs from radians
    return 3600 * np.degrees(yrad), 3600 * np.degrees(zrad)

def yagzag_to_pixels(yang, zang, allow_bad=False):
    """
    Convert ACA y-angle/z-angle positions to ACA pixel row, column.
    It is expected that the y-angle/z-angle input arguments have the same length.

    :param yang: ACA y-angle (single value, list, or 1-d numpy array)
    :param zang: ACA z-angle (single value, list, or 1-d numpy array)
    :param allow_bad: boolean switch.  If true, method will not throw errors
                         if the resulting row/col values are nominally off the ACA CCD.
    :rtype: (row, col) each vector of the same length as row/col
    """
    yang = np.array(yang)
    zang = np.array(zang)
    row, col = _poly_convert(yang, zang, ACA2PIX_coeff)
    if (not allow_bad and
        (np.any(row > 511.5) or np.any(row < -512.5)
         or np.any(col > 511.5) or np.any(col < -512.5))):
        raise ValueError("Coordinate off CCD")
    return row, col

def _poly_convert(y, z, coeffs):
    if y.size != z.size:
        raise ValueError("Mismatched number of Y/Z coords")
    yy = y * y
    zz = z * z
    poly = np.array([np.ones_like(y), z, y, zz, y * z, yy, zz * z, y * zz, yy * z, yy * y])
    newy = np.sum(coeffs[:, 0] * poly.transpose(), axis=-1)
    newz = np.sum(coeffs[:, 1] * poly.transpose(), axis=-1)
    return newy, newz

