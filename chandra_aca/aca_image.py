from itertools import izip, count
from copy import deepcopy

import numpy as np


class ACAImage(np.ndarray):
    """
    Ndarray subclass that supports functionality for the Chandra ACA.
    Most importantly it allows image indexing and slicing in absolute
    "aca" coordinates, where the image lower left coordinate is specified
    by object ``row0`` and ``col0`` attributes.  It also provides a
    ``meta`` dict that can be used to store additional useful information.
    """

    @property
    def aca(self):
        """
        Return a light copy (same data) of self but with the _aca_coords
        attribute switched on so that indexing is absolute.
        """
        obj = self[()]
        obj._aca_coords = True
        return obj

    def __new__(cls, *args, **kwargs):

        row0 = kwargs.pop('row0', 0)
        col0 = kwargs.pop('col0', 0)
        meta = kwargs.pop('meta', {})

        try:
            arr = np.array(*args, **kwargs)
        except Exception:
            arr = np.zeros(*args, **kwargs)

        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(arr).view(cls)

        if obj.ndim != 2:
            raise ValueError('{} must be 2-d'.format(cls.__name__))

        # add the new attribute to the created instance
        obj.row0 = row0
        obj.col0 = col0
        obj.meta = meta
        obj._aca_coords = False

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.row0 = getattr(obj, 'row0', 0)
        self.col0 = getattr(obj, 'col0', 0)
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
                for i, it, rc0 in izip(count(), item, (self.row0, self.col0)):
                    if isinstance(it, slice):
                        start = None if it.start is None else it.start - rc0
                        stop = None if it.stop is None else it.stop - rc0
                        item[i] = slice(start, stop, it.step)
                    else:
                        item[i] = it - rc0
                item = tuple(item)

            # Compute new row0, col0 (stored in out_rc) based on input item
            for i, it, rc0 in izip(count(), item, (self.row0, self.col0)):
                if isinstance(it, slice):
                    if it.start is not None:
                        out_rc[i] = rc0 + it.start
                else:
                    out_rc[i] = rc0 + it

        return item, out_rc[0], out_rc[1]

    def __getitem__(self, item):
        # New values after getitem
        row0 = None
        col0 = None

        item, row0, col0 = self._adjust_item(item)

        out = super(ACAImage, self).__getitem__(item)

        if isinstance(out, ACAImage):
            if row0 is not None:
                out.row0 = row0
            if col0 is not None:
                out.col0 = col0

        return out

    def __setitem__(self, item, value):
        item, row0, col0 = self._adjust_item(item)

        super(ACAImage, self).__setitem__(item, value)

    def __repr__(self):
        out = '<{} row0={} col0={}\n{}>'.format(self.__class__.__name__,
                                                self.row0, self.col0,
                                                np.asarray(self).__repr__())
        return out
