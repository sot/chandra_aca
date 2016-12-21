from itertools import izip, count

import numpy as np


class CCDImage(np.ndarray):

    @property
    def ccd(self):
        obj = self[()]
        obj._ccd_index = True
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
        obj._ccd_index = False

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        self.row0 = getattr(obj, 'row0', 0)
        self.col0 = getattr(obj, 'col0', 0)
        self.meta = getattr(obj, 'meta', {})
        self._ccd_index = getattr(obj, '_ccd_index', False)

    def _adjust_item(self, item):
        ccd_index = self._ccd_index
        if isinstance(item, CCDImage):
            item = (slice(item.row0, item.row0 + item.shape[0]),
                    slice(item.col0, item.col0 + item.shape[1]))
            ccd_index = True

        if isinstance(item, tuple) and len(item) == 2 and ccd_index:
            item = list(item)
            for i, it, offset in izip(count(), item, (self.row0, self.col0)):
                if isinstance(it, slice):
                    start = None if it.start is None else it.start - offset
                    stop = None if it.stop is None else it.stop - offset
                    item[i] = slice(start, stop, it.step)
                else:
                    item[i] = it - offset
            item = tuple(item)

        return item

    def __getitem__(self, item):
        if isinstance(item, (tuple, CCDImage)):
            item = self._adjust_item(item)

        return super(CCDImage, self).__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, (tuple, CCDImage)):
            item = self._adjust_item(item)

        return super(CCDImage, self).__setitem__(item, value)

    def __repr__(self):
        out = '<{} row0={} col0={}\n{}>'.format(self.__class__.__name__,
                                                self.row0, self.col0,
                                                np.asarray(self).__repr__())
        return out
