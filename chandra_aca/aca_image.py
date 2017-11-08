# Licensed under a 3-clause BSD style license - see LICENSE.rst
from itertools import count
from copy import deepcopy
from six.moves import zip

import numpy as np
from astropy.utils.compat.misc import override__dir__

__all__ = ['ACAImage', 'centroid_fm']


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


def _prep_6x6(img, bgd=None):
    """
    Subtract background and in case of 8x8 image
    cut and return the 6x6 inner section.
    """
    # Cast to an ndarray (without copying)
    img = img.view(np.ndarray)

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
