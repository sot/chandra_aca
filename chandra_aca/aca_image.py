# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from math import floor
from itertools import count, chain
from copy import deepcopy
from pathlib import Path

import six
from six.moves import zip

import numba
import numpy as np
from astropy.utils.compat.misc import override__dir__

__all__ = ['ACAImage', 'centroid_fm', 'AcaPsfLibrary', 'EIGHT_LABELS']

EIGHT_LABELS = np.array([['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1'],
                         ['I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1'],
                         ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2'],
                         ['I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'O2', 'P2'],
                         ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3'],
                         ['I3', 'J3', 'K3', 'L3', 'M3', 'N3', 'O3', 'P3'],
                         ['A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4'],
                         ['I4', 'J4', 'K4', 'L4', 'M4', 'N4', 'O4', 'P4']])
"""Constant for labeling ACA image pixels using the EQ-278 spec format.
Pixel A1 has the lowest values of row and column; pixel H1 has the lowest
row and highest col; pixel I4 has the highest row and lowest column."""


def _operator_factory(operator, inplace=False):
    """
    Generate data model methods like __add__(self, other) and
    __iadd__(self, other).  These always operate in the coordinate
    system of the left and right operands.  If both are in ACA
    coordinates then any non-overlapping pixels are ignored.
    """
    # Define the operator and the in-place version (which might be the
    # same if op is already in-place)
    op = getattr(np.ndarray, '__{}__'.format(operator))
    inplace_op = op if inplace else getattr(np.ndarray, '__i{}__'.format(operator))

    def _operator(self, other):

        if isinstance(other, ACAImage) and (other._aca_coords or self._aca_coords):
            # If inplace then work on the original self, else use a copy
            out = self if inplace else self.copy()

            sz_r0, sz_c0 = self.shape
            sz_r1, sz_c1 = other.shape

            # If images overlap do this process, else return unmodified ``out``.
            if all(diff > 0 for diff in [self.row0 + sz_r0 - other.row0,
                                         self.col0 + sz_c0 - other.col0,
                                         other.row0 + sz_r1 - self.row0,
                                         other.col0 + sz_c1 - self.col0]):

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
                inplace_op(out[section], other.view(np.ndarray)[r_min:r_max, c_min:c_max])

        else:
            out = op(self, other)  # returns self for inplace ops

        return out
    return _operator


class ACAImage(np.ndarray):
    """
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
        Return a light copy (same data) of self but with the _aca_coords
        attribute switched on so that indexing is absolute.
        """
        obj = self.view(self.__class__)
        obj.meta = self.meta
        obj._aca_coords = True
        return obj

    def __new__(cls, *args, **kwargs):

        meta = kwargs.pop('meta', {})

        # Set default row0 and col0 to 0 (if not already in meta), and
        # then override with like-named kwargs.  row0 attribute => meta['IMGROW0']
        for ax in ('row0', 'col0'):
            imgax = 'IMG' + ax.upper()
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
            raise ValueError('{} must be 2-d'.format(cls.__name__))

        # add the new attribute to the created instance
        obj.meta = meta
        obj._aca_coords = False

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.meta = deepcopy(getattr(obj, 'meta', {}))
        self._aca_coords = getattr(obj, '_aca_coords', False)

    __add__ = _operator_factory('add')
    __sub__ = _operator_factory('sub')
    __mul__ = _operator_factory('mul')
    if not six.PY3:
        __div__ = _operator_factory('div')
    __truediv__ = _operator_factory('truediv')
    __floordiv__ = _operator_factory('floordiv')
    __mod__ = _operator_factory('mod')
    __pow__ = _operator_factory('pow')

    __iadd__ = _operator_factory('iadd', inplace=True)
    __isub__ = _operator_factory('isub', inplace=True)
    __imul__ = _operator_factory('imul', inplace=True)
    if not six.PY3:
        __idiv__ = _operator_factory('idiv', inplace=True)
    __itruediv__ = _operator_factory('itruediv', inplace=True)
    __ifloordiv__ = _operator_factory('ifloordiv', inplace=True)
    __imod__ = _operator_factory('imod', inplace=True)
    __ipow__ = _operator_factory('ipow', inplace=True)

    def _adjust_item(self, item):
        """
        This is the money method that does all the work of manipulating
        an item and subsequent row0/col0 when accessing and slicing.
        """
        # Allow slicing via an existing ACAImage object
        aca_coords = self._aca_coords
        if isinstance(item, ACAImage):
            item = (slice(item.row0, item.row0 + item.shape[0]),
                    slice(item.col0, item.col0 + item.shape[1]))
            aca_coords = True

        out_rc = [None, None]  # New [row0, col0]

        if isinstance(item, (int, np.int)):
            item = (item,)

        if isinstance(item, tuple):
            if aca_coords:
                # Interpret input `item` indices as being expressed in absolute
                # terms and subtract row0/col0 as needed.
                item = list(item)
                for i, it, rc0 in zip(count(), item, (self.row0, self.col0)):
                    if isinstance(it, slice):
                        start = None if it.start is None else it.start - rc0
                        stop = None if it.stop is None else it.stop - rc0
                        item[i] = slice(start, stop, it.step)
                    else:
                        item[i] = it - rc0
                item = tuple(item)

            # Compute new row0, col0 (stored in out_rc) based on input item
            for i, it, rc0 in zip(count(), item, (self.row0, self.col0)):
                if isinstance(it, slice):
                    if it.start is not None:
                        out_rc[i] = rc0 + it.start
                else:
                    out_rc[i] = rc0 + it

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
        out = '<{} row0={} col0={}\n{}>'.format(self.__class__.__name__,
                                                self.row0, self.col0,
                                                outarr.__repr__())
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

    def centroid_fm(self, bgd=None, pix_zero_loc='center', norm_clip=None):
        """
        First moment centroid of ``self`` using 6x6 mousebitten image for input
        6x6 or 8x8 images.

        Note that the returned ``norm`` is the sum of the background-subtracted 6x6
        mousebitten image, not the entire image.

        :param bgd: background to subtract, scalar or NxN ndarray (float)
        :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'
        :param norm_clip: clip image norm at this min value (default is None and
                          implies Exception for non-positive norm)

        :returns: row, col, norm float
        """
        row, col, norm = centroid_fm(self, bgd=bgd, pix_zero_loc=pix_zero_loc,
                                     norm_clip=norm_clip)
        if self._aca_coords:
            row += self.row0
            col += self.col0

        return row, col, norm

    @override__dir__
    def __dir__(self):
        return list(self.meta)

    @property
    def row0(self):
        return self.meta['IMGROW0']

    @row0.setter
    def row0(self, value):
        self.meta['IMGROW0'] = np.int64(value)

    @property
    def col0(self):
        return self.meta['IMGCOL0']

    @col0.setter
    def col0(self, value):
        self.meta['IMGCOL0'] = np.int64(value)

    @classmethod
    def _read_flicker_cdfs(cls):
        """Read flickering pixel model cumulative distribution functions
        and associated metadata.  Set up class variables accordingly.

        The flicker_cdf file here was created using:
        /proj/sot/ska/www/ASPECT/ipynb/chandra_aca/flickering-pixel-model.ipynb

        """
        from astropy.io import fits

        filename = Path(__file__).parent / 'data' / 'flicker_cdf.fits.gz'
        with fits.open(filename) as hdus:
            hdu = hdus[0]
            hdr = hdu.header

            # Get the main data, which is an n_cdf * n_cdf_x array.  Each row corresponds
            # to the CDF for a particular bin range in e-/sec, e.g. 200 to 300 e-/sec.
            # CDF will go from 0.0 to 1.0
            cls.flicker_cdfs = hdu.data.astype(np.float64)

            # CDF_x is the x-value of the distribution, namely the log-amplitude change
            # in pixel value due to a flicker event.
            cls.flicker_cdf_x = np.linspace(hdr['cdf_x0'], hdr['cdf_x1'], hdr['n_cdf_x'])

            # CDF bin range (e-/sec) for each for in flicker_cdfs.
            cdf_bins = []
            for ii in range(hdr['n_bin']):
                cdf_bins.append(hdr[f'cdf_bin{ii}'])
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

        :param flicker_mean_time: mean flickering time (sec, default=10000)
        :param flicker_scale: multiplicative factor beyond model default for
               flickering amplitude (default=1.0)
        :param seed: random seed for reproducibility (default=None => no seed)
        """
        if not hasattr(self, 'flicker_cdf_bins'):
            self._read_flicker_cdfs()

        self.flicker_mean_time = flicker_mean_time
        self.flicker_scale = flicker_scale

        if seed is not None:
            np.random.seed(seed)
            _numba_random_seed(seed)

        # Make a flattened view of the image for easier update processing.
        # Also store the initial pixel values, since flicker updates are
        # relative to the initial value, not the current value (which would
        # end up random-walking).
        self.flicker_vals = self.view(np.ndarray).ravel()
        self.flicker_vals0 = self.flicker_vals.copy()
        self.flicker_n_vals = len(self.flicker_vals)

        # Get the index to the CDFs which is appropriate for each pixel
        # based on its initial value.
        self.flicker_cdf_idxs = np.searchsorted(self.flicker_cdf_bins,
                                                self.flicker_vals0) - 1

        # Create an array of time (secs) until next flicker for each pixel
        # This is drawing from an exponential distribution.  For the initial
        # time assume the flickering is randomly phased within that interval.
        phase = np.random.uniform(0.0, 1.0, size=self.flicker_n_vals)
        rand_unifs = np.random.uniform(0.0, 1.0, size=self.flicker_n_vals)
        t_flicker = -np.log(1.0 - rand_unifs) * flicker_mean_time
        self.flicker_times = t_flicker * phase

        # Pixels where self.flicker_cdf_idxs == 0 have val < 50 (no CDF) and are
        # modeled as not flickering.  Make a mask to indicate which ones flicker.
        self.flicker_mask = self.flicker_cdf_idxs >= 0

    def flicker_update(self, dt, use_numba=True):
        """
        Propagate the image forward by ``dt`` seconds and update any pixels
        that have flickered during that interval.

        This has the option to use one of two implementations.  The default is
        to use the numba-based version which is about 6 times faster.  The
        vectorized version is left in for reference.

        :param dt: time (secs) to propagate image
        :param use_numba: use the numba version of updating (default=True)
        """
        if not hasattr(self, 'flicker_times'):
            self.flicker_init()

        if use_numba:
            _flicker_update_numba(dt, len(self.flicker_vals),
                                  self.flicker_vals0,
                                  self.flicker_vals,
                                  self.flicker_mask,
                                  self.flicker_times,
                                  self.flicker_cdf_idxs,
                                  self.flicker_cdf_x,
                                  self.flicker_cdfs,
                                  self.flicker_scale,
                                  self.flicker_mean_time)
        else:
            self._flicker_update_vectorized(dt)

    def _flicker_update_vectorized(self, dt):

        self.flicker_times[self.flicker_mask] -= dt

        # When flicker_times < 0 that means a flicker occurs
        idxs = np.where(self.flicker_times < 0)[0]

        # Random uniform used for (1) distribution of flickering amplitude
        # via the CDFs and (2) distribution of time to next flicker.
        rand_ampls = np.random.uniform(0.0, 1.0, size=len(idxs))
        rand_times = np.random.uniform(0.0, 1.0, size=len(idxs))

        for idx, rand_time, rand_ampl in zip(idxs, rand_times, rand_ampls):
            # Determine the new value after flickering and set in array view.
            # First get the right CDF from the list of CDFs based on the pixel value.
            cdf_idx = self.flicker_cdf_idxs[idx]
            y = np.interp(fp=self.flicker_cdf_x,
                          xp=self.flicker_cdfs[cdf_idx],
                          x=rand_ampl)

            if self.flicker_scale != 1.0:
                # Express the multiplicative change as (1 + x) and change
                # it to be (1 + x * scale).  This makes sense for positive y,
                # so use abs(y) and then flip the sign back at the end.  For
                # negative y this is the same as doing this trick expressing the
                # change as 1 / (1 + x).
                dy = (10 ** np.abs(y) - 1.0) * self.flicker_scale + 1.0
                y = np.log10(dy) * np.sign(y)

            val = 10 ** (np.log10(self.flicker_vals0[idx]) + y)
            self.flicker_vals[idx] = val

            # Get the new time before next flicker
            t_flicker = -np.log(1.0 - rand_time) * self.flicker_mean_time
            self.flicker_times[idx] = t_flicker


@numba.jit(nopython=True)
def _numba_random_seed(seed):
    np.random.seed(seed)


@numba.jit(nopython=True)
def _flicker_update_numba(dt, nvals,
                          flicker_vals0,
                          flicker_vals,
                          flicker_mask,
                          flicker_times,
                          flicker_cdf_idxs,
                          flicker_cdf_x,
                          flicker_cdfs,
                          flicker_scale,
                          flicker_mean_time):
    """
    Propagate the image forward by ``dt`` seconds and update any pixels
    that have flickered during that interval.
    """
    for ii in range(nvals):  # nvals
        if not flicker_mask[ii]:
            continue

        flicker_times[ii] -= dt

        if flicker_times[ii] > 0:
            continue

        # Random uniform used for (1) distribution of flickering amplitude
        # via the CDFs and (2) distribution of time to next flicker.
        rand_ampl = np.random.uniform(0.0, 1.0)
        rand_time = np.random.uniform(0.0, 1.0)

        # Determine the new value after flickering and set in array view.
        # First get the right CDF from the list of CDFs based on the pixel value.
        cdf_idx = flicker_cdf_idxs[ii]
        y = np_interp(yin=flicker_cdf_x,
                      xin=flicker_cdfs[cdf_idx],
                      xout=rand_ampl)

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
    Subtract background and in case of 8x8 image
    cut and return the 6x6 inner section.
    """
    if isinstance(bgd, np.ndarray):
        bgd = bgd.view(np.ndarray)

    if bgd is not None:
        img = img - bgd

    if img.shape == (8, 8):
        img = img[1:7, 1:7]

    return img


def centroid_fm(img, bgd=None, pix_zero_loc='center', norm_clip=None):
    """
    First moment centroid of ``img``.

    Return FM centroid in coords where lower left pixel of image has value
    (0.0, 0.0) at the center (for pix_zero_loc='center') or the lower-left edge
    (for pix_zero_loc='edge').

    :param img: NxN ndarray
    :param bgd: background to subtract, float of NXN ndarray
    :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'
    :param norm_clip: clip image norm at this min value (default is None and
                      implies Exception for non-positive norm)

    :returns: row, col, norm float
    """
    # Cast to an ndarray (without copying)
    img = img.view(np.ndarray)

    sz_r, sz_c = img.shape
    if sz_r != sz_c:
        raise ValueError('input img must be square')

    rw, cw = np.mgrid[1:7, 1:7] if sz_r == 8 else np.mgrid[0:sz_r, 0:sz_r]

    if sz_r in (6, 8):
        img = _prep_6x6(img, bgd)
        img[[0, 0, 5, 5], [0, 5, 0, 5]] = 0

    norm = np.sum(img)
    if norm_clip is not None:
        norm = norm.clip(norm_clip, None)
    else:
        if norm <= 0:
            raise ValueError('non-positive image norm {}'.format(norm))

    row = np.sum(rw * img) / norm
    col = np.sum(cw * img) / norm

    if pix_zero_loc == 'edge':
        # Transform row/col values from 'center' convention (as returned
        # by centroiding) to the 'edge' convention requested by user.
        row = row + 0.5
        col = col + 0.5
    elif pix_zero_loc != 'center':
        raise ValueError("pix_zero_loc can be only 'edge' or 'center'")

    return row, col, norm


class AcaPsfLibrary(object):
    """
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
            filename = os.path.join(os.path.dirname(__file__), 'data', 'aca_psf_lib.dat')
        dat = Table.read(filename, format='ascii.basic', guess=False)
        self.dat = dat

        # Sub-pixel grid spacing in pixels.  This assumes the sub-pixels are
        # all the same size and square, which is indeed the case.
        self.drc = dat['row_bin_right_edge'][0] - dat['row_bin_left_edge'][0]

        for row in dat:
            ii = row['row_bin_idx']
            jj = row['col_bin_idx']
            psf = np.array([row[label] for label in chain(*EIGHT_LABELS)]).reshape(8, 8)
            psfs[ii, jj] = psf

        self.psfs = psfs

    def get_psf_image(self, row, col, norm=1.0, pix_zero_loc='center',
                      interpolation='bilinear', aca_image=True):
        """
        Get interpolated ACA PSF image that corresponds to pixel location
        ``row``, ``col``.

        :param row: (float) row value of PSF centroid
        :param col: (float) col value of PSF centroid
        :param norm: (float) summed intensity of PSF image
        :param pix_zero_loc: row/col coords are integral at 'edge' or 'center'
        :param interpolation: 'nearest' | 'bilinear' (default)
        :param aca_image: return ACAImage if True, else return ndarray

        :returns: ACAImage if (aca_image is True) else (ndarray image, row0, col0)
        """
        drc = self.drc

        if pix_zero_loc == 'center':
            # Transform to 'edge' coordinates (pixel lower-left corner at 0.0, 0.0)
            row = row + 0.5
            col = col + 0.5
        elif pix_zero_loc != 'edge':
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

        if interpolation == 'nearest':
            # Int index into PSF library
            ii = int(round(ix))
            jj = int(round(iy))
            psf = self.psfs[ii, jj].copy()

        elif interpolation == 'bilinear':
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
