import numpy as np

from ..aca_image import ACAImage

im6 = np.arange(6**2).reshape((6, 6))
im8 = np.arange(8**2).reshape((8, 8))


def test_init():
    # Init from Python list of lists
    a = ACAImage([[1, 2], [3, 4]])
    assert a.shape == (2, 2)
    assert np.all(a == [[1, 2], [3, 4]])

    # Init from ndarray
    a = ACAImage(im6)
    assert np.all(a == im6)
    assert a.row0 == 0
    assert a.col0 == 0
    assert a.shape == (6, 6)

    a = ACAImage(im6, row0=1, col0=2)
    assert a.row0 == 1
    assert a.col0 == 2
    assert a.meta == {'IMGROW0': 1, 'IMGCOL0': 2}

    # Init as zeroes with shape
    a = ACAImage(shape=(1024, 1024), row0=-512.0, col0=-512.0)
    assert np.all(a == np.zeros((1024, 1024)))
    assert type(a.row0) is np.int64
    assert type(a.col0) is np.int64

    a = ACAImage(im6, meta={'IMGROW0': 1, 'IMGCOL0': 2})
    assert a.row0 == 1
    assert a.col0 == 2


def test_row_col_set():
    a = ACAImage(im6)
    a.row0 = -10.0
    a.col0 = -20.0
    assert a.row0 == -10
    assert a.col0 == -20
    assert type(a.row0) is np.int64
    assert type(a.col0) is np.int64


def test_meta_set():
    a = ACAImage(im6)
    a.ATTR = 10
    assert a.ATTR == 10
    assert a.meta['ATTR'] == 10
    a.meta['ATTR'] = 20
    assert a.ATTR == 20


def test_slice():
    a = ACAImage(im6, row0=1, col0=2)

    # Slicing with no funny business
    assert np.all(a[1] == im6[1])
    assert np.all(a[:, 2] == im6[:, 2])
    assert np.all(a[1, 2] == im6[1, 2])
    assert np.all(a[-3:, 2:4] == im6[-3:, 2:4])

    # Row0/col0 update properly for normal slicing
    a2 = a[2:, 3:]
    assert a2.row0 == 3  # 1 + 2
    assert a2.col0 == 5  # 2 + 3

    # Slice start=None does not change row/col0
    a2 = a.aca[:, :]
    assert a2.row0 == a.row0
    assert a2.col0 == a.col0

    # Slicing in ACA coordinates
    assert np.all(a.aca[1] == im6[0])
    assert np.all(a.aca[:, 2] == im6[:, 0])
    assert np.all(a.aca[1, 2] == im6[0, 0])
    assert np.all(a.aca[4:, 2:4] == im6[3:, 0:2])

    # Row0/col0 update properly for ACA slicing
    a2 = a.aca[2:, 3:]
    assert a2.row0 == 2  # 1 + (2 - 1)
    assert a2.col0 == 3  # 2 + (3 - 2)
    assert a2._aca_coords is False  # new object has normal slicing
    assert np.all(a2 == im6[1:, 1:])  # 2 - 1, 3 - 2
    assert a2[1, 2] == im6[1 + 1, 1 + 2]

    # Set from slice
    a = ACAImage(im6, row0=1, col0=2)
    a.aca[2:, 4:] = 0
    im60 = im6.copy()
    im60[1:, 2:] = 0
    assert np.all(a == im60)

    # Slice is a view
    a2 = a.aca[2:, 4:]
    a2[()] = 10
    im60[1:, 2:] = 10
    assert np.all(a == im60)

    # Slice using an ACAImage
    a2 = ACAImage(im8, row0=1.0, col0=1.0)
    assert np.all(a2[a] == im8[0:6, 1:7])

    # Set slice using an ACAImage
    a2[a] = 0
    im80 = im8.copy()
    im80[0:6, 1:7] = 0
    assert np.all(a2 == im80)


def test_slice_list():
    a = ACAImage(im6, row0=1, col0=2)
    r = [1, 2, 3]
    c = [3, 4, 5]
    a2 = a[r, c]
    assert np.all(a2 == im6[r, c])

    a2 = a.aca[r, c]
    assert np.all(a2 == im6[r - a.row0, c - a.col0])


def test_meta_ref():
    a = ACAImage(im6, row0=1, col0=2)
    assert a.meta is a.aca.meta

    a2 = a[1:, 2:]
    assert a2.meta is not a.meta
