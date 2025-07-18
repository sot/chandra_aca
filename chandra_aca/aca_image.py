# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Classes and utilities related to ACA readout images.

See :ref:`aca_image` for more details and examples.
"""

import os
from copy import deepcopy
from itertools import chain, count
from math import floor
from pathlib import Path

import numba
import numpy as np
import requests
from astropy.table import Table
from cxotime import CxoTimeLike
from ska_helpers import retry

__all__ = ["ACAImage", "centroid_fm", "AcaPsfLibrary", "EIGHT_LABELS", "get_aca_images"]

EIGHT_LABELS = np.array(
    [
        ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"],
        ["I1", "J1", "K1", "L1", "M1", "N1", "O1", "P1"],
        ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"],
        ["I2", "J2", "K2", "L2", "M2", "N2", "O2", "P2"],
        ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3"],
        ["I3", "J3", "K3", "L3", "M3", "N3", "O3", "P3"],
        ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4"],
        ["I4", "J4", "K4", "L4", "M4", "N4", "O4", "P4"],
    ]
)
"""Constant for labeling ACA image pixels using the EQ-278 spec format.
Pixel A1 has the lowest values of row and column; pixel H1 has the lowest
row and highest col; pixel I4 has the highest row and lowest column."""


def _operator_factory(operator, inplace=False):
    """
    Generate data model methods.

    Generate data model methods like __add__(self, other) and
    __iadd__(self, other).  These always operate in the coordinate
    system of the left and right operands.  If both are in ACA
    coordinates then any non-overlapping pixels are ignored.
    """
    # Define the operator and the in-place version (which might be the
    # same if op is already in-place)
    op = getattr(np.ndarray, "__{}__".format(operator))
    inplace_op = op if inplace else getattr(np.ndarray, "__i{}__".format(operator))

    def _operator(self, other):
        if isinstance(other, ACAImage) and (other._aca_coords or self._aca_coords):
            # If inplace then work on the original self, else use a copy
            out = self if inplace else self.copy()

            sz_r0, sz_c0 = self.shape
            sz_r1, sz_c1 = other.shape

            # If images overlap do this process, else return unmodified ``out``.
            if all(
                diff > 0
                for diff in [
                    self.row0 + sz_r0 - other.row0,
                    self.col0 + sz_c0 - other.col0,
                    other.row0 + sz_r1 - self.row0,
                    other.col0 + sz_c1 - self.col0,
                ]
            ):
                dr = other.row0 - self.row0
                dc = other.col0 - self.col0

                r_min, r_max = -min(0, dr), min(sz_r1, sz_r0 - dr)
                c_min, c_max = -min(0, dc), min(sz_c1, sz_c0 - dc)

                row0 = max(self.row0, other.row0)
                col0 = max(self.col0, other.col0)
                sz_r = r_max - r_min
                sz_c = c_max - c_min
                section = ACAImage(shape=(sz_r, sz_c), row0=row0, col0=col0)

                # Always use the inplace operator, but remember that ``out`` is a copy of
                # self for inplace=False (thus mimicking the non-inplace version).
                inplace_op(
                    out[section], other.view(np.ndarray)[r_min:r_max, c_min:c_max]
                )

        else:
            out = op(self, other)  # returns self for inplace ops

        return out

    return _operator


class ACAImage(np.ndarray):
    """
    ACA Image class.

    ACAImage is an ndarray subclass that supports functionality for the Chandra
    ACA. Most importantly it allows image indexing and slicing in absolute
    "aca" coordinates, where the image lower left coordinate is specified
    by object ``row0`` and ``col0`` attributes.

    It also provides a ``meta`` dict that can be used to store additional useful
    information.  Any keys which are all upper-case will be exposed as object
    attributes, e.g. ``img.BGDAVG`` <=> ``img.meta['BGDAVG']``.  The ``row0``
    attribute  is a proxy for ``img.meta['IMGROW0']``, and likewise for ``col0``.

    When initializing an ``ACAImage``, additional ``*args`` and ``**kwargs`` are
    used to try initializing via ``np.array(*args, **kwargs)``.  If this fails
    then ``np.zeros(*args, **kwargs)`` is tried.  In this way one can either
    initialize from array data or create a new array of zeros.

    Examples::

      >>> import numpy as np
      >>> from chandra_aca.aca_image import ACAImage
      >>> dat = np.random.uniform(size=(1024, 1024))
      >>> a = ACAImage(dat, row0=-512, col0=-512)
      >>> a = ACAImage([[1,2], [3,4]], meta={'BGDAVG': 5.2})
      >>> a = ACAImage(shape=(1024, 1024), row0=-512, col0=-512)

    :param row0: row coordinate of lower left image pixel (int, default=0)
    :param col0: col coordinate of lower left image pixel (int, default=0)
    :param meta: dict of object attributes
    :param ``*args``: additional args passed to np.array() or np.zeros()
    :param ``**kwargs``: additional kwargs passed to np.array() or np.zeros()
    """

    @property
    def aca(self):
        """
        Return a light copy of self with _aca_coords on.

        Return a light copy (same data) of self but with the _aca_coords
        attribute switched on so that indexing is absolute.
        """
        obj = self.view(self.__class__)
        obj.meta = self.meta
        obj._aca_coords = True
        return obj

    def __new__(cls, *args, **kwargs):
        meta = kwargs.pop("meta", {})

        # Set default row0 and col0 to 0 (if not already in meta), and
        # then override with like-named kwargs.  row0 attribute => meta['IMGROW0']
        for ax in ("row0", "col0"):
            imgax = "IMG" + ax.upper()
            meta.setdefault(imgax, 0)
            if ax in kwargs:
                meta[imgax] = np.int64(kwargs.pop(ax))

        try:
            arr = np.array(*args, **kwargs)
        except Exception:
            arr = np.zeros(*args, **kwargs)

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = arr.view(cls)

        if obj.ndim != 2:
            raise ValueError("{} must be 2-d".format(cls.__name__))

        # add the new attribute to the created instance
        obj.meta = meta
        obj._aca_coords = False

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.meta = deepcopy(getattr(obj, "meta", {}))
        self._aca_coords = getattr(obj, "_aca_coords", False)

    __add__ = _operator_factory("add")
    __sub__ = _operator_factory("sub")
    __mul__ = _operator_factory("mul")
    __truediv__ = _operator_factory("truediv")
    __floordiv__ = _operator_factory("floordiv")
    __mod__ = _operator_factory("mod")
    __pow__ = _operator_factory("pow")

    __iadd__ = _operator_factory("iadd", inplace=True)
    __isub__ = _operator_factory("isub", inplace=True)
    __imul__ = _operator_factory("imul", inplace=True)
    __itruediv__ = _operator_factory("itruediv", inplace=True)
    __ifloordiv__ = _operator_factory("ifloordiv", inplace=True)
    __imod__ = _operator_factory("imod", inplace=True)
    __ipow__ = _operator_factory("ipow", inplace=True)

    def _adjust_item(self, item):
        """
        This is the money method that does all the work of manipulating
        an item and subsequent row0/col0 when accessing and slicing.
        """  # noqa: D205
        # Allow slicing via an existing ACAImage object
        aca_coords = self._aca_coords
        if isinstance(item, ACAImage):
            item = (
                slice(item.row0, item.row0 + item.shape[0]),
                slice(item.col0, item.col0 + item.shape[1]),
            )
            aca_coords = True

        # These are new [row0, col0] values for the __getitem__ output. If either is left at None
        # then the downstream code uses the original row0 or col0 value, respectively.
        out_rc = [None, None]

        if isinstance(item, (int, np.integer)):
            item = (item,)

        if isinstance(item, tuple):
            shape = self.shape
            if aca_coords:
                # Interpret input `item` indices as being expressed in absolute
                # terms and subtract row0/col0 as needed.
                item = list(item)
                for i, it, rc0 in zip(count(), item, (self.row0, self.col0)):
                    if isinstance(it, slice):
                        start = (
                            None
                            if it.start is None
                            else np.clip(it.start - rc0, 0, shape[i])
                        )
                        stop = (
                            None
                            if it.stop is None
                            else np.clip(it.stop - rc0, 0, shape[i])
                        )
                        item[i] = slice(start, stop, it.step)
                    elif it is not ...:
                        item[i] = it - rc0
                        if np.any(item[i] < 0) or np.any(item[i] >= shape[i]):
                            raise IndexError(
                                f"index {it} is out of bounds for axis {i} with "
                                f"limits {rc0}:{rc0 + shape[i]}"
                            )
                item = tuple(item)

            # Compute new row0, col0 (stored in out_rc) based on input item
            for i, it, rc0 in zip(count(), item, (self.row0, self.col0)):
                if isinstance(it, slice):
                    if it.start is not None:
                        rc_off = it.start if it.start >= 0 else shape[i] + it.start
                        out_rc[i] = rc0 + rc_off
                elif it is not ...:
                    it_arr = np.array(it)
                    rc_off = np.where(it_arr >= 0, it_arr, shape[i] + it_arr)
                    out_rc[i] = rc0 + rc_off

        return item, out_rc[0], out_rc[1]

    def __getitem__(self, item):
        item, row0, col0 = self._adjust_item(item)

        out = super(ACAImage, self).__getitem__(item)

        if isinstance(out, ACAImage):
            if row0 is not None:
                out.row0 = row0
            if col0 is not None:
                out.col0 = col0
            out._aca_coords = False

        return out

    def __setitem__(self, item, value):
        item, row0, col0 = self._adjust_item(item)

        aca_coords = self._aca_coords
        try:
            self._aca_coords = False
            super(ACAImage, self).__setitem__(item, value)
        finally:
            self._aca_coords = aca_coords

    def __repr__(self):
        # Make an integerized version for viewing more nicely
        outarr = np.asarray(np.round(self)).astype(int)
        out = "<{} row0={} col0={}\n{}>".format(
            self.__class__.__name__, self.row0, self.col0, outarr.__repr__()
        )
        return out

    def __getattr__(self, attr):
        if attr.isupper():
            try:
                return self.meta[attr]
            except KeyError:
                pass

        return super(ACAImage, self).__getattribute__(attr)

    def __setattr__(self, attr, value):
        if attr.isupper():
            self.meta[attr] = value
        else:
            super(ACAImage, self).__setattr__(attr, value)

    def centroid_fm(self, bgd=None, pix_zero_loc="center", norm_clip=None):
        """
        First moment centroid of ``self`` using 6x6 mousebitten image for input 6x6 or 8x8 images.

        Note that the returned ``norm`` is the sum of the background-subtracted 6x6
        mousebitten image, not the entire image.

        Parameters
        ----------
        bgd
            background to subtract, scalar or NxN ndarray (float)
        pix_zero_loc
            row/col coords are integral at 'edge' or 'center'
        norm_clip : clip image norm at this min value (default is None and
            implies Exception for non-positive norm)

        Returns
        -------
        row, col, norm float
        """
        row, col, norm = centroid_fm(
            self, bgd=bgd, pix_zero_loc=pix_zero_loc, norm_clip=norm_clip
        )
        if self._aca_coords:
            row += self.row0
            col += self.col0

        return row, col, norm

    def __dir__(self):
        return sorted(super().__dir__() + list(self.meta))

    @property
    def row0(self):
        return self.meta["IMGROW0"]

    @row0.setter
    def row0(self, value):
        self.meta["IMGROW0"] = np.int64(value)

    @property
    def col0(self):
        return self.meta["IMGCOL0"]

    @col0.setter
    def col0(self, value):
        self.meta["IMGCOL0"] = np.int64(value)

    @classmethod
    def _read_flicker_cdfs(cls):
        """
        Read flickering pixel model cumulative distribution functions and associated metadata.

        Set up class variables accordingly.

        The flicker_cdf file here was created using:
        /proj/sot/ska/www/ASPECT/ipynb/chandra_aca/flickering-pixel-model.ipynb

        """
        from astropy.io import fits

        filename = Path(__file__).parent / "data" / "flicker_cdf.fits.gz"
        with fits.open(filename) as hdus:
            hdu = hdus[0]
            hdr = hdu.header

            # Get the main data, which is an n_cdf * n_cdf_x array.  Each row corresponds
            # to the CDF for a particular bin range in e-/sec, e.g. 200 to 300 e-/sec.
            # CDF will go from 0.0 to 1.0
            cls.flicker_cdfs = hdu.data.astype(np.float64)

            # CDF_x is the x-value of the distribution, namely the log-amplitude change
            # in pixel value due to a flicker event.
            cls.flicker_cdf_x = np.linspace(
                hdr["cdf_x0"], hdr["cdf_x1"], hdr["n_cdf_x"]
            )

            # CDF bin range (e-/sec) for each for in flicker_cdfs.
            cdf_bins = [hdr[f"cdf_bin{ii}"] for ii in range(hdr["n_bin"])]
            cls.flicker_cdf_bins = np.array(cdf_bins)

    def flicker_init(self, flicker_mean_time=10000, flicker_scale=1.0, seed=None):
        """Initialize instance variables to allow for flickering pixel updates.

        The ``flicker_scale`` can be interpreted as follows: if the pixel
        was going to flicker by a multiplicative factor of (1 + x), now
        make it flicker by (1 + x * flicker_scale).  This applies for flickers
        that increase the amplitude.  For flickers that make the value smaller,
        then it would be 1 / (1 + x) => 1 / (1 + x * flicker_scale).

        The flicker_cdf file here was created using:
        /proj/sot/ska/www/ASPECT/ipynb/chandra_aca/flickering-pixel-model.ipynb

        Examples and performance details at:
        /proj/sot/ska/www/ASPECT/ipynb/chandra_aca/flickering-implementation.ipynb

        The model was reviewed and approved at SS&AWG on 2019-05-22.

        Parameters
        ----------
        flicker_mean_time
            mean flickering time (sec, default=10000)
        flicker_scale : multiplicative factor beyond model default for
            flickering amplitude (default=1.0)
        seed
            random seed for reproducibility (default=None => no seed)
        """
        if not hasattr(self, "flicker_cdf_bins"):
            self._read_flicker_cdfs()

        self.flicker_mean_time = flicker_mean_time
        self.flicker_scale = flicker_scale
        self.test_idx = 1 if seed == -1 else 0

        if seed is not None and seed != -1:
            np.random.seed(seed)
            _numba_random_seed(seed)

        # Make a flattened view of the image for easier update processing.
        # Also store the initial pixel values, since flicker updates are
        # relative to the initial value, not the current value (which would
        # end up random-walking).
        self.flicker_vals = self.view(np.ndarray).ravel()
        self.flicker_vals0 = self.flicker_vals.copy()
        self.flicker_n_vals = len(self.flicker_vals)

        # Make a bool ACAImage like self to allow convenient mask/unmask of
        # pixels to flicker.  This is used in annie.  Also make the corresponding
        # 1-d ravelled version.
        self.flicker_mask = ACAImage(
            np.ones(self.shape, dtype=bool), row0=self.row0, col0=self.col0
        )
        self.flicker_mask_vals = self.flicker_mask.view(np.ndarray).ravel()

        # Get the index to the CDFs which is appropriate for each pixel
        # based on its initial value.
        self.flicker_cdf_idxs = (
            np.searchsorted(self.flicker_cdf_bins, self.flicker_vals0) - 1
        )

        # Create an array of time (secs) until next flicker for each pixel
        if seed == -1:
            # Special case of testing, use a constant flicker_mean_time initially
            t_flicker = np.ones(shape=(self.flicker_n_vals,)) * flicker_mean_time
            phase = 1.0
        else:
            # This is drawing from an exponential distribution.  For the initial
            # time assume the flickering is randomly phased within that interval.
            phase = np.random.uniform(0.0, 1.0, size=self.flicker_n_vals)
            rand_unifs = np.random.uniform(0.0, 1.0, size=self.flicker_n_vals)
            t_flicker = -np.log(1.0 - rand_unifs) * flicker_mean_time

        self.flicker_times = t_flicker * phase

    def flicker_update(self, dt, use_numba=True):
        """
        Do a flicker update.

        Propagate the image forward by ``dt`` seconds and update any pixels
        that have flickered during that interval.

        This has the option to use one of two implementations.  The default is
        to use the numba-based version which is about 6 times faster.  The
        vectorized version is left in for reference.

        Parameters
        ----------
        dt
            time (secs) to propagate image
        use_numba
            use the numba version of updating (default=True)
        """
        if not hasattr(self, "flicker_times"):
            self.flicker_init()

        if use_numba:
            _flicker_update_numba(
                dt,
                len(self.flicker_vals),
                self.test_idx,
                self.flicker_vals0,
                self.flicker_vals,
                self.flicker_mask_vals,
                self.flicker_times,
                self.flicker_cdf_idxs,
                self.flicker_cdf_x,
                self.flicker_cdfs,
                self.flicker_scale,
                self.flicker_mean_time,
            )
            if self.test_idx > 0:
                self.test_idx += 1
        else:
            self._flicker_update_vectorized(dt)

    def _flicker_update_vectorized(self, dt):
        self.flicker_times[self.flicker_mask_vals] -= dt

        # When flicker_times < 0 that means a flicker occurs
        ok = (self.flicker_times < 0) & (self.flicker_cdf_idxs >= 0)
        idxs = np.where(ok)[0]

        # Random uniform used for (1) distribution of flickering amplitude
        # via the CDFs and (2) distribution of time to next flicker.
        rand_ampls = np.random.uniform(0.0, 1.0, size=len(idxs))
        rand_times = np.random.uniform(0.0, 1.0, size=len(idxs))

        for idx, rand_time, rand_ampl in zip(
            idxs, rand_times, rand_ampls, strict=False
        ):
            # Determine the new value after flickering and set in array view.
            # First get the right CDF from the list of CDFs based on the pixel value.
            cdf_idx = self.flicker_cdf_idxs[idx]
            y = np.interp(
                fp=self.flicker_cdf_x, xp=self.flicker_cdfs[cdf_idx], x=rand_ampl
            )

            if self.flicker_scale != 1.0:
                # Express the multiplicative change as (1 + x) and change
                # it to be (1 + x * scale).  This makes sense for positive y,
                # so use abs(y) and then flip the sign back at the end.  For
                # negative y this is the same as doing this trick expressing the
                # change as 1 / (1 + x).
                dy = (10 ** np.abs(y) - 1.0) * self.flicker_scale + 1.0
                y = np.log10(dy) * np.sign(y)

            val = self.flicker_vals0[idx] * 10**y
            self.flicker_vals[idx] = val

            # Get the new time before next flicker
            t_flicker = -np.log(1.0 - rand_time) * self.flicker_mean_time
            self.flicker_times[idx] = t_flicker


@numba.jit(nopython=True)
def _numba_random_seed(seed):
    np.random.seed(seed)


@numba.jit(nopython=True)
def _flicker_update_numba(
    dt,
    nvals,
    test_idx,
    flicker_vals0,
    flicker_vals,
    flicker_mask_vals,
    flicker_times,
    flicker_cdf_idxs,
    flicker_cdf_x,
    flicker_cdfs,
    flicker_scale,
    flicker_mean_time,
):
    """
    Do a flicker update.

    Propagate the image forward by ``dt`` seconds and update any pixels
    that have flickered during that interval.
    """
    for ii in range(nvals):  # nvals
        # cdf_idx is -1 for pixels with dark current in range that does not flicker.
        # Don't flicker those ones or pixels that are masked out.
        cdf_idx = flicker_cdf_idxs[ii]
        if cdf_idx < 0 or not flicker_mask_vals[ii]:
            continue

        flicker_times[ii] -= dt

        if flicker_times[ii] > 0:
            continue

        if test_idx > 0:
            # Deterministic and reproducible but bouncy sequence that is reproducible in C
            # (which has a different random number generator).
            rand_ampl = np.abs(np.sin(float(ii + test_idx)))
            rand_time = np.abs(np.cos(float(ii + test_idx)))
        else:
            # Random uniform used for (1) distribution of flickering amplitude
            # via the CDFs and (2) distribution of time to next flicker.
            rand_ampl = np.random.uniform(0.0, 1.0)
            rand_time = np.random.uniform(0.0, 1.0)

        # Determine the new value after flickering and set in array view.
        # First get the right CDF from the list of CDFs based on the pixel value.
        y = np_interp(yin=flicker_cdf_x, xin=flicker_cdfs[cdf_idx], xout=rand_ampl)

        if flicker_scale != 1.0:
            # Express the multiplicative change as (1 + x) and change
            # it to be (1 + x * scale).  This makes sense for positive y,
            # so use abs(y) and then flip the sign back at the end.  For
            # negative y this is the same as doing this trick expressing the
            # change as 1 / (1 + x).
            dy = (10 ** np.abs(y) - 1.0) * flicker_scale + 1.0
            y = np.log10(dy) * np.sign(y)

        val = 10 ** (np.log10(flicker_vals0[ii]) + y)
        flicker_vals[ii] = val

        # Get the new time before next flicker
        t_flicker = -np.log(1.0 - rand_time) * flicker_mean_time
        flicker_times[ii] = t_flicker


@numba.jit(nopython=True)
def np_interp(yin, xin, xout):
    idx = np.searchsorted(xin, xout)

    if idx == 0:
        return yin[0]

    if idx == len(xin):
        return yin[-1]

    x0 = xin[idx - 1]
    x1 = xin[idx]
    y0 = yin[idx - 1]
    y1 = yin[idx]
    yout = (xout - x0) / (x1 - x0) * (y1 - y0) + y0

    return yout


def _prep_6x6(img, bgd=None):
    """
    Subtract background and in case of 8x8 image cut and return the 6x6 inner section.
    """
    if isinstance(bgd, np.ndarray):
        bgd = bgd.view(np.ndarray)

    # Subtract background and/or ensure a copy of img is made since downstream
    # code may modify the image.
    if bgd is not None:
        img = img - bgd
    else:
        img = img.copy()

    if img.shape == (8, 8):
        img = img[1:7, 1:7]

    return img


def centroid_fm(img, bgd=None, pix_zero_loc="center", norm_clip=None):
    """
    First moment centroid of ``img``.

    Return FM centroid in coords where lower left pixel of image has value
    (0.0, 0.0) at the center (for pix_zero_loc='center') or the lower-left edge
    (for pix_zero_loc='edge').

    Parameters
    ----------
    img
        NxN ndarray
    bgd
        background to subtract, float of NXN ndarray
    pix_zero_loc
        row/col coords are integral at 'edge' or 'center'
    norm_clip : clip image norm at this min value (default is None and
        implies Exception for non-positive norm)

    Returns
    -------
    row, col, norm float
    """
    # Cast to an ndarray (without copying)
    img = img.view(np.ndarray)

    sz_r, sz_c = img.shape
    if sz_r != sz_c:
        raise ValueError("input img must be square")

    rw, cw = np.mgrid[1:7, 1:7] if sz_r == 8 else np.mgrid[0:sz_r, 0:sz_r]

    if sz_r in (6, 8):
        img = _prep_6x6(img, bgd)
        img[[0, 0, 5, 5], [0, 5, 0, 5]] = 0

    norm = np.sum(img)
    if norm_clip is not None:
        norm = norm.clip(norm_clip, None)
    elif norm <= 0:
        raise ValueError("non-positive image norm {}".format(norm))

    row = np.sum(rw * img) / norm
    col = np.sum(cw * img) / norm

    if pix_zero_loc == "edge":
        # Transform row/col values from 'center' convention (as returned
        # by centroiding) to the 'edge' convention requested by user.
        row = row + 0.5
        col = col + 0.5
    elif pix_zero_loc != "center":
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    return row, col, norm


class AcaPsfLibrary(object):
    """
    AcaPsfLibrary class

    Access the ACA PSF library, whch is a library of 8x8 images providing the integrated
    (pixelated) ACA PSF over a grid of subpixel locations.

    Example::

      >>> from chandra_aca.aca_image import AcaPsfLibrary
      >>> apl = AcaPsfLibrary()  # Reads in PSF library data file
      >>> img = apl.get_psf_image(row=-10.456, col=250.123, norm=100000)
      >>> img
      <ACAImage row0=-14 col0=247
      array([[   39,    54,    56,    52,    37,    33,    30,    21],
             [   79,   144,   260,   252,   156,    86,    67,    36],
             [  162,   544,  2474,  5269,  2012,   443,   163,    57],
             [  255,  1420, 10083, 12688, 11273,  1627,   302,    78],
             [  186,  1423,  8926,  8480, 12292,  2142,   231,    64],
             [   80,   344,  1384,  6509,  4187,   665,   111,    43],
             [   40,    78,   241,   828,   616,   188,    65,    29],
             [   24,    39,    86,   157,   139,    69,    48,    32]])>

    :param filename: file name of ACA PSF library (default=built-in file)
    :returns: AcaPsfLibrary object
    """

    def __init__(self, filename=None):
        from astropy.table import Table  # Table is a somewhat-heavy import

        psfs = {}

        if filename is None:
            filename = os.path.join(
                os.path.dirname(__file__), "data", "aca_psf_lib.dat"
            )
        dat = Table.read(filename, format="ascii.basic", guess=False)
        self.dat = dat

        # Sub-pixel grid spacing in pixels.  This assumes the sub-pixels are
        # all the same size and square, which is indeed the case.
        self.drc = dat["row_bin_right_edge"][0] - dat["row_bin_left_edge"][0]

        for row in dat:
            ii = row["row_bin_idx"]
            jj = row["col_bin_idx"]
            psf = np.array([row[label] for label in chain(*EIGHT_LABELS)]).reshape(8, 8)
            psfs[ii, jj] = psf

        self.psfs = psfs

    def get_psf_image(
        self,
        row,
        col,
        norm=1.0,
        pix_zero_loc="center",
        interpolation="bilinear",
        aca_image=True,
    ):
        """
        Get interpolated ACA PSF image that corresponds to pixel location ``row``, ``col``.

        Parameters
        ----------
        row
            (float) row value of PSF centroid
        col
            (float) col value of PSF centroid
        norm
            (float) summed intensity of PSF image
        pix_zero_loc
            row/col coords are integral at 'edge' or 'center'
        interpolation
            'nearest' | 'bilinear' (default)
        aca_image
            return ACAImage if True, else return ndarray

        Returns
        -------
        ACAImage if (aca_image is True) else (ndarray image, row0, col0)
        """
        drc = self.drc

        if pix_zero_loc == "center":
            # Transform to 'edge' coordinates (pixel lower-left corner at 0.0, 0.0)
            row = row + 0.5
            col = col + 0.5
        elif pix_zero_loc != "edge":
            raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

        # 8x8 image row0, col0
        round_row = round(row)
        round_col = round(col)
        row0 = int(round_row) - 4
        col0 = int(round_col) - 4

        # Subpixel position in range (-0.5, 0.5)
        r = row - round_row
        c = col - round_col

        # Floating point index into PSF library in range (0.0, 10.0)
        # (assuming 10x10 grid of PSFs covering central pixel
        ix = (r + 0.5) / drc - 0.5
        iy = (c + 0.5) / drc - 0.5

        if interpolation == "nearest":
            # Int index into PSF library
            ii = int(round(ix))
            jj = int(round(iy))
            psf = self.psfs[ii, jj].copy()

        elif interpolation == "bilinear":
            # Int index into PSF library
            ii = int(floor(ix))
            jj = int(floor(iy))

            # Following wikipedia notation (Unit Square section of
            # https://en.wikipedia.org/wiki/Bilinear_interpolation)

            # Float index within subpixel bin in range (0, 1)
            x = ix - ii
            y = iy - jj

            # Finally the bilinear interpolation of the PSF images.
            f = self.psfs
            b0 = (1 - x) * (1 - y)
            b1 = x * (1 - y)
            b2 = (1 - x) * y
            b3 = x * y
            P0 = f[ii, jj]
            P1 = f[ii + 1, jj]
            P2 = f[ii, jj + 1]
            P3 = f[ii + 1, jj + 1]
            psf = P0 * b0 + P1 * b1 + P2 * b2 + P3 * b3

        else:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")

        if norm != 1.0:
            psf *= norm

        out = ACAImage(psf, row0=row0, col0=col0) if aca_image else (psf, row0, col0)
        return out


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_aca_images(
    start: CxoTimeLike, stop: CxoTimeLike, bgsub=False, source="maude", **maude_kwargs
) -> Table:
    """
    Get ACA images and ancillary data from either the MAUDE or CXC data sources.

    The returned table of ACA images and ancillary data will include the default
    columns returned by chandra_aca.maude_decom.get_aca_images or
    mica.archive.aca_l0.get_aca_images. Additionally, an IMGSIZE column will be
    added to the maude_decom aca_images so images from either source will have
    that column::

             name            dtype  unit
      --------------------- ------- -----------
         IMGSIZE              int32  pixels

    If bgsub is True then the table will also include columns::

             name            dtype  unit
      --------------------- ------- -----------
         IMG_BGSUB          float64  DN
         IMG_DARK           float64  DN
         T_CCD_SMOOTH       float64  degC

    where:

    - 'IMG_BGSUB': background subtracted image
    - 'IMG_DARK': dark current image
    - 'T_CCD_SMOOTH': smoothed CCD temperature

    The IMG_DARK individual values are only calculated if within the 1024x1024
    dark current map, otherwise they are set to 0.  In practice this is not an
    issue in that IMG and IMG_BGSUB must be within the CCD to be tracked.

    Parameters
    ----------
    start : CxoTimeLike
        Start time.
    stop : CxoTimeLike
        Stop time (CXC sec).
    bgsub : bool
        Include background subtracted images in output table. Default is False.
    source : str
        Data source for image and temperature telemetry ('maude' or 'cxc'). For 'cxc',
        the image telemetry is from mica and temperature telemetry is from CXC L0 via
        cheta.
    **maude_kwargs
        Additional kwargs for maude data source.

    Returns
    -------
    imgs_table : astropy.table.Table
        Table of ACA images and ancillary data.
    """
    import mica.archive.aca_dark
    import mica.archive.aca_l0

    import chandra_aca.dark_subtract
    import chandra_aca.maude_decom

    # Set up configuration for maude or cxc
    if source == "maude":
        get_aca_images_func = chandra_aca.maude_decom.get_aca_images
    elif source == "cxc":
        get_aca_images_func = mica.archive.aca_l0.get_aca_images
        # Explicitly set maude_kwargs to empty dict for cxc source
        maude_kwargs = {}
    else:
        raise ValueError(f"source must be 'maude' or 'cxc', not {source}")

    # Get images
    imgs_table = get_aca_images_func(start, stop, **maude_kwargs)

    # Add an IMGSIZE column if not present (maude)
    # IMGTYPE 4 -> 8, 1 -> 6, 0 -> 4
    if "IMGSIZE" not in imgs_table.colnames:
        imgs_table["IMGSIZE"] = np.zeros(len(imgs_table), dtype=np.int32)
        for itype, size in zip([4, 1, 0], [8, 6, 4], strict=True):
            imgs_table["IMGSIZE"][imgs_table["IMGTYPE"] == itype] = size

    # If bgsub is False, then just return the table as-is.
    if not bgsub:
        return imgs_table

    # If bgsub is True, then calculate and add to the table.
    # If the table has zero length then just add the columns with zero length.
    if len(imgs_table) > 0:
        dark_data = mica.archive.aca_dark.get_dark_cal_props(
            imgs_table["TIME"].min(),
            select="nearest",
            include_image=True,
            aca_image=False,
        )
        img_dark = dark_data["image"]
        tccd_dark = dark_data["ccd_temp"]
        t_ccds = chandra_aca.dark_subtract.get_tccd_data(
            imgs_table["TIME"], source=source, **maude_kwargs
        )
        imgs_dark = chandra_aca.dark_subtract.get_dark_current_imgs(
            imgs_table, img_dark, tccd_dark, t_ccds
        )
        imgs_bgsub = imgs_table["IMG"] - imgs_dark
        imgs_bgsub.clip(0, None)

        imgs_table["IMG_BGSUB"] = imgs_bgsub
        imgs_table["IMG_DARK"] = imgs_dark
        imgs_table["T_CCD_SMOOTH"] = t_ccds
    else:
        # Add the columns to the table even if there are no rows
        imgs_table["IMG_BGSUB"] = np.zeros(shape=(0, 8, 8))
        imgs_table["IMG_DARK"] = np.zeros(shape=(0, 8, 8))
        imgs_table["T_CCD_SMOOTH"] = np.zeros(shape=0)

    return imgs_table
