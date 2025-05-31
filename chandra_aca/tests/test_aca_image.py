# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
import mica.common
import numpy as np
import pytest
from cxotime import CxoTime
from mica.archive.aca_dark import get_dark_cal_props

import chandra_aca.aca_image
from chandra_aca.aca_image import ACAImage, centroid_fm

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
    assert a.meta == {"IMGROW0": 1, "IMGCOL0": 2}

    # Init as zeroes with shape
    a = ACAImage(shape=(1024, 1024), row0=-512.0, col0=-512.0)
    assert np.all(a == np.zeros((1024, 1024)))
    assert type(a.row0) is np.int64
    assert type(a.col0) is np.int64

    a = ACAImage(im6, meta={"IMGROW0": 1, "IMGCOL0": 2})
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


def test_negative_index():
    img = ACAImage(im8, row0=11, col0=-22)
    img2 = img[-3, :]
    assert np.all(img2 == im8[-3, :])
    assert img2.row0 == img[5, :].row0
    assert img2.col0 == img.col0

    img2 = img[-3:-1, :]
    assert np.all(img2 == im8[-3:-1, :])
    assert img2.row0 == img[5:7, :].row0
    assert img2.col0 == img.col0


def test_out_of_bounds():
    img = ACAImage(im8, row0=11, col0=-22)
    # OK in slice
    img2 = img.aca[10:13, -24:-20]
    assert img2.row0 == 11
    assert img2.col0 == -22
    assert img2.shape == (2, 2)

    with pytest.raises(
        IndexError, match="index 10 is out of bounds for axis 0 with limits 11:19"
    ):
        img.aca[10, :]

    with pytest.raises(
        IndexError, match="index 10 is out of bounds for axis 1 with limits -22:-14"
    ):
        img.aca[:, 10]

    with pytest.raises(
        IndexError, match="index -10 is out of bounds for axis 0 with size 8"
    ):
        img[[-10, -1, 2], :]

    with pytest.raises(
        IndexError, match="index -9 is out of bounds for axis 1 with size 8"
    ):
        img[:, -9]


def test_meta_set():
    a = ACAImage(im6)
    a.ATTR = 10
    assert a.ATTR == 10
    assert a.meta["ATTR"] == 10
    a.meta["ATTR"] = 20
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
    i = np.array([1], dtype=int)
    assert np.all(a.aca[i[0]] == im6[0])
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


def test_ellipsis():
    from chandra_aca import aca_image

    row0 = 11
    col0 = 22
    inp = np.arange(64).reshape(8, 8)
    img = aca_image.ACAImage(inp, row0=row0, col0=col0)

    assert np.all(img[...] == inp)
    assert img[...].row0 == row0
    assert img[...].col0 == col0

    img2 = img[-1, ...]
    assert np.all(img2 == inp[-1, ...])
    # assert img2.row0 == img[size - 1 , ...].row0  # Fails independently of this PR
    assert img2.col0 == col0

    img2 = img[1, ...]
    assert np.all(img2 == inp[1, ...])
    assert img2.row0 == row0 + 1
    assert img2.col0 == col0

    img2 = img[..., -2]
    assert np.all(img2 == inp[..., -2])
    assert img2.row0 == row0
    # assert img2.col0 == img[..., size - 2]  # fails independently of this PR

    img2 = img[..., 2]
    assert np.all(img[..., 2] == inp[..., 2])
    assert img[..., 2].row0 == row0
    assert img[..., 2].col0 == col0 + 2

    img2 = img.aca[row0 + 1, ...]
    assert np.all(img2 == inp[1, ...])
    assert img2.row0 == row0 + 1
    assert img2.col0 == col0

    img2 = img.aca[..., col0 + 2]
    assert np.all(img2 == inp[..., 2])
    assert img2.row0 == row0
    assert img2.col0 == col0 + 2

    with pytest.raises(IndexError):
        # an index can only have a single ellipsis
        img[..., ...]


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


def test_fm_centroid():
    # 6x6 image
    img = np.zeros((6, 6), dtype=float)
    img[0, 0] = 1000  # Should be ignored by mouse-bite
    img[2, 2] = 100
    img_orig = img.copy()
    row, col, norm = centroid_fm(img)
    assert np.isclose(row, 2.0)
    assert np.isclose(col, 2.0)
    assert np.isclose(norm, 100)
    assert np.all(img == img_orig)

    # 8x8 image with background of 10
    img = np.zeros((8, 8), dtype=float) + 10
    img[0, 0] = 1000  # Should be ignored by mouse-bite
    img[1, 1] = 1000  # Should be ignored by mouse-bite
    img[3, 3] = 100
    img_orig = img.copy()
    row, col, norm = centroid_fm(img, bgd=10)
    assert np.isclose(row, 3.0)
    assert np.isclose(col, 3.0)
    assert np.isclose(norm, 90)
    assert np.all(img == img_orig)

    # Check 'edge' coordinates
    row, col, norm = centroid_fm(img, bgd=10, pix_zero_loc="edge")
    assert np.isclose(row, 3.5)
    assert np.isclose(col, 3.5)
    assert np.isclose(norm, 90)

    # Non-zero background
    img = np.zeros((8, 8), dtype=float) + 10
    img[3, 3] += 100
    img[4, 4] += 100
    row, col, norm = centroid_fm(img)
    assert np.isclose(row, 3.5)
    assert np.isclose(col, 3.5)
    assert np.isclose(norm, 32 * 10 + 200)

    # Exceptions
    with pytest.raises(ValueError) as err:
        row, col, norm = centroid_fm(img, bgd=100)
    assert "non-positive" in str(err)

    with pytest.raises(ValueError) as err:
        row, col, norm = centroid_fm(img, pix_zero_loc="FAIL")
    assert "pix_zero_loc" in str(err)

    # Norm clip (with expected bogus centroid value)
    row, col, norm = centroid_fm(img, bgd=100, pix_zero_loc="edge", norm_clip=1.0)
    assert np.isclose(row, -9379.5)
    assert np.isclose(col, -9379.5)


@pytest.mark.parametrize("aca", [True, False])
def test_aca_image_fm_centroid(aca):
    # 6x6 image
    row0 = -150
    col0 = 200
    img = np.zeros((6, 6), dtype=float)
    img = ACAImage(img, row0=row0, col0=col0)
    img[0, 0] = 1000  # Should be ignored by mouse-bite
    img[2, 2] = 100
    row, col, norm = (img.aca if aca else img).centroid_fm()
    assert np.isclose(row, 2.0 + (row0 if aca else 0))
    assert np.isclose(col, 2.0 + (col0 if aca else 0))
    assert np.isclose(norm, 100)

    # 8x8 image with background of 10
    img = np.zeros((8, 8), dtype=float) + 10
    img = ACAImage(img, row0=row0, col0=col0)
    img[0, 0] = 1000  # Should be ignored by mouse-bite
    img[1, 1] = 1000  # Should be ignored by mouse-bite
    img[3, 3] = 100
    row, col, norm = (img.aca if aca else img).centroid_fm(bgd=10)
    assert np.isclose(row, 3.0 + (row0 if aca else 0))
    assert np.isclose(col, 3.0 + (col0 if aca else 0))
    assert np.isclose(norm, 90)

    # Check 'edge' coordinates
    row, col, norm = (img.aca if aca else img).centroid_fm(bgd=10, pix_zero_loc="edge")
    assert np.isclose(row, 3.5 + (row0 if aca else 0))
    assert np.isclose(col, 3.5 + (col0 if aca else 0))
    assert np.isclose(norm, 90)

    # Non-zero background
    img = np.zeros((8, 8), dtype=float) + 10
    img = ACAImage(img, row0=row0, col0=col0)
    img[3, 3] += 100
    img[4, 4] += 100
    row, col, norm = (img.aca if aca else img).centroid_fm()
    assert np.isclose(row, 3.5 + (row0 if aca else 0))
    assert np.isclose(col, 3.5 + (col0 if aca else 0))
    assert np.isclose(norm, 32 * 10 + 200)

    # Exceptions
    with pytest.raises(ValueError) as err:
        row, col, norm = (img.aca if aca else img).centroid_fm(bgd=100)
    assert "non-positive" in str(err)

    with pytest.raises(ValueError) as err:
        row, col, norm = (img.aca if aca else img).centroid_fm(pix_zero_loc="FAIL")
    assert "pix_zero_loc" in str(err)

    # Norm clip (with expected bogus centroid value)
    row, col, norm = (img.aca if aca else img).centroid_fm(
        bgd=100, pix_zero_loc="edge", norm_clip=1.0
    )
    assert np.isclose(row, -9379.5 + (row0 if aca else 0))
    assert np.isclose(col, -9379.5 + (col0 if aca else 0))


def test_aca_image_operators():
    row0 = 10
    col0 = 20

    # Test case of adding in native (non-ACA) coordinates.  Just like normal
    # numpy array add.
    a = ACAImage(shape=(4, 4), row0=row0, col0=col0) + 2
    b = ACAImage(np.arange(1, 17).reshape(4, 4), row0=row0 - 2, col0=col0 - 1) * 10
    # In [8]: a
    # <ACAImage row0=10 col0=10
    # array([[2, 2, 2, 2],
    #        [2, 2, 2, 2],
    #        [2, 2, 2, 2],
    #        [2, 2, 2, 2]])>

    # In [9]: b
    # <ACAImage row0=8 col0=9
    # array([[ 10,  20,  30,  40],
    #        [ 50,  60,  70,  80],
    #        [ 90, 100, 110, 120],
    #        [130, 140, 150, 160]])>

    ab = [[12, 22, 32, 42], [52, 62, 72, 82], [92, 102, 112, 122], [132, 142, 152, 162]]
    assert np.all(a + b == ab)

    # Now test adding in ACA coordinates with partially overlapping images.
    ab_aca = [[102, 112, 122, 2], [142, 152, 162, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
    assert np.all(a + b.aca == ab_aca)
    assert np.all(a.aca + b == ab_aca)
    assert np.all(a.aca + b.aca == ab_aca)

    # Inplace operations.  Note that a.aca += <anything> does not work because
    # that is trying to update the attribute instead of the returned object.
    a = ACAImage(shape=(4, 4), row0=row0, col0=col0) + 2
    a += b
    assert np.all(a == ab)

    a = ACAImage(shape=(4, 4), row0=row0, col0=col0) + 2
    a += b.aca
    assert np.all(a == ab_aca)

    # Test for one image fully enclosed in the other
    a = ACAImage(shape=(4, 4), row0=row0, col0=col0) + 2
    b = ACAImage(np.arange(1, 17).reshape(4, 4), row0=row0 + 2, col0=col0 - 2) * 10

    assert np.all(
        a + b.aca == [[2, 2, 2, 2], [2, 2, 2, 2], [32, 42, 2, 2], [72, 82, 2, 2]]
    )

    a = ACAImage(np.arange(16).reshape(4, 4), row0=row0, col0=col0) + 2
    b = ACAImage([[100, 200], [300, 400]], row0=row0 + 1, col0=col0 + 1)

    # Shape mismatch
    with pytest.raises(ValueError):
        a + b

    # right side is enclosed within left side
    out = a + b.aca
    assert np.all(
        out == [[2, 3, 4, 5], [6, 107, 208, 9], [10, 311, 412, 13], [14, 15, 16, 17]]
    )

    assert out.row0 == 10
    assert out.col0 == 20

    # right side is a superset of left side
    out = b + a.aca
    assert np.all(out == [[107, 208], [311, 412]])
    assert out.row0 == 11
    assert out.col0 == 21

    # Subtraction
    out = a - b.aca
    assert np.all(
        out == [[2, 3, 4, 5], [6, -93, -192, 9], [10, -289, -388, 13], [14, 15, 16, 17]]
    )


def test_flicker_numba():
    a = ACAImage(np.linspace(0, 800, 9).reshape(3, 3))
    a.flicker_init(flicker_mean_time=1000, flicker_scale=1.5, seed=10)
    for _ii in range(10):
        a.flicker_update(100.0, use_numba=True)

    assert np.all(np.round(a) == [[0, 81, 200], [326, 176, 609], [659, 720, 1043]])


def test_flicker_vectorized():
    a = ACAImage(np.linspace(0, 800, 9).reshape(3, 3))
    a.flicker_init(flicker_mean_time=1000, flicker_scale=1.5, seed=10)
    for _ii in range(10):
        a.flicker_update(100.0, use_numba=False)

    assert np.all(np.round(a) == [[0, 111, 200], [219, 436, 531], [470, 829, 822]])


def test_flicker_no_seed():
    """Make sure results vary when seed is not supplied"""
    a = ACAImage(np.linspace(0, 800, 9).reshape(3, 3))
    a.flicker_init(flicker_mean_time=300)
    for _ii in range(10):
        a.flicker_update(100.0)

    b = ACAImage(np.linspace(0, 800, 9).reshape(3, 3))
    b.flicker_init(flicker_mean_time=300)
    for _ii in range(10):
        b.flicker_update(100.0)

    assert np.any(a != b)


def test_flicker_test_sequence():
    """Test the deterministic testing sequence that allows for
    cross-implementation comparison.  This uses the seed=-1 pathway.
    """
    a = ACAImage(np.linspace(0, 800, 9).reshape(3, 3))
    a.flicker_init(seed=-1, flicker_mean_time=15)
    dt = 10
    assert np.all(np.round(a) == [[0, 100, 200], [300, 400, 500], [600, 700, 800]])

    a.flicker_update(dt)
    assert np.all(np.round(a) == [[0, 100, 200], [300, 400, 500], [600, 700, 800]])

    a.flicker_update(dt)
    assert np.all(np.round(a) == [[0, 80, 229], [447, 374, 531], [952, 693, 815]])
    a.flicker_update(dt)
    assert np.all(np.round(a) == [[0, 80, 229], [278, 374, 531], [592, 693, 815]])

    a.flicker_update(dt)
    assert np.all(np.round(a) == [[0, 80, 185], [278, 374, 531], [592, 693, 815]])

    assert np.allclose(
        a.flicker_times,
        np.array(
            [
                15.0,
                49.06630191,
                48.34713118,
                38.34713118,
                28.34713118,
                1.03039723,
                26.30875397,
                16.30875397,
                7.40192939,
            ]
        ),
    )


def test_get_short_range_aca_images():
    date = "2025:120:03:59:29.057"
    images = chandra_aca.aca_image.get_aca_images(
        start=date, stop=CxoTime(date) + 8 * u.s, source="maude", bgsub=False
    )
    assert len(images) == 16


def test_get_aca_image_maude_channel():
    # This is a SIM time when we had 4x4 images in all slots
    date = "2025:071:15:15:20.020"
    images_asvt = chandra_aca.aca_image.get_aca_images(
        start=date,
        stop=CxoTime(date) + 7 * u.s,
        source="maude",
        bgsub=False,
        channel="ASVT",
    )
    assert len(images_asvt) == 56
    # This is a 4x4
    assert np.count_nonzero(~images_asvt[0]["IMG"].mask) == 16

    # For the same times in flight these are different
    images_flight = chandra_aca.aca_image.get_aca_images(
        start=date,
        stop=CxoTime(date) + 7 * u.s,
        source="maude",
        bgsub=False,
        channel="FLIGHT",
    )
    # This is 8x8
    assert np.count_nonzero(~images_flight[0]["IMG"].mask) == 64
    # and there are fewer images
    assert len(images_flight) == 16


def images_check_range(start, stop, img_table, *, bgsub):
    tstart = CxoTime(start).secs
    tstop = CxoTime(stop).secs

    # Check that start and stop of the range are within 4.1 secs of the requested times
    assert np.all(np.abs(img_table["TIME"][0] - tstart) < 4.1)
    assert np.all(np.abs(img_table["TIME"][-1] - tstop) < 4.1)

    # Check that all the times in the table are within the requested range
    assert np.all((img_table["TIME"] >= tstart) & (img_table["TIME"] < tstop))

    # Check that all the slots are in there
    assert np.all(np.isin(np.arange(8), img_table["IMGNUM"]))

    # Check that if the table has BGSUB column that IMG - DARK = BGSUB
    if bgsub:
        assert np.allclose(
            img_table["IMG"] - img_table["IMG_DARK"], img_table["IMG_BGSUB"]
        )

    # If there's a DARK column, then check it against the dark image from the dark cal
    if bgsub:
        dc = get_dark_cal_props(
            tstart, select="nearest", include_image=True, aca_image=True
        )
        full_img_dark = dc["image"]
        tccd_dark = dc["ccd_temp"]
        for row in img_table:
            dark_row = row["IMG_DARK"]
            tccd_row = row["T_CCD_SMOOTH"]
            row8x8 = row["IMGROW0_8X8"]
            col8x8 = row["IMGCOL0_8X8"]

            # If row8x8 or col8x8 is < -512 just skip the test for now
            if row8x8 < -512 or col8x8 < -512:
                continue

            dark_ref = full_img_dark.aca[row8x8 : row8x8 + 8, col8x8 : col8x8 + 8]
            from chandra_aca.dark_model import dark_temp_scale_img

            dark_scale = dark_temp_scale_img(dark_ref, tccd_dark, tccd_row)
            # Default to using 1.696 integ time if not present in the image table
            integ = row["INTEG"] if "INTEG" in img_table.colnames else 1.696
            img_dark = ACAImage(dark_scale * integ / 5, row0=row8x8, col0=col8x8)

            # Make an 8x8 expected image (even if the dark current goes off the CCD)
            expected_img = (
                ACAImage(np.zeros((8, 8)), row0=row8x8, col0=col8x8).aca + img_dark
            )

            # Compare
            assert np.allclose(expected_img, dark_row)


HAS_ACA0_ARCHIVE = (Path(mica.common.MICA_ARCHIVE) / "aca0").exists()


@pytest.mark.skipif(not HAS_ACA0_ARCHIVE, reason="No ACA0 archive")
def test_get_aca_images_cxc_and_maude():
    """Test get_aca_images in the cxc mode

    This checks that the dark images are reasonable and the answers match maude.
    """
    tstart = "2012:270:02:44:00"
    tstop = "2012:270:02:47:00"

    # Get CXC data and check that it looks reasonable
    img_table_cxc = chandra_aca.aca_image.get_aca_images(
        tstart, tstop, source="cxc", bgsub=False
    )
    images_check_range(tstart, tstop, img_table_cxc, bgsub=False)

    img_table_cxc_bgsub = chandra_aca.aca_image.get_aca_images(
        tstart, tstop, source="cxc", bgsub=True
    )
    images_check_range(tstart, tstop, img_table_cxc_bgsub, bgsub=True)

    # Get MAUDE data and check that it looks reasonable
    img_table_maude = chandra_aca.aca_image.get_aca_images(
        tstart, tstop, source="maude", bgsub=False
    )
    images_check_range(tstart, tstop, img_table_maude, bgsub=False)

    img_table_maude_bgsub = chandra_aca.aca_image.get_aca_images(
        tstart, tstop, source="maude", bgsub=True
    )
    images_check_range(tstart, tstop, img_table_maude_bgsub, bgsub=True)

    assert np.allclose(img_table_maude["IMG"], img_table_maude_bgsub["IMG"])
    assert np.allclose(img_table_maude["TIME"], img_table_maude_bgsub["TIME"])

    # Check that the two mica tables are the same in the key columns
    assert np.allclose(img_table_cxc["IMG"], img_table_cxc_bgsub["IMG"])
    assert np.allclose(img_table_cxc["TIME"], img_table_cxc_bgsub["TIME"])

    # Check that the tables are the same
    img_table_maude.sort(["TIME", "IMGNUM"])
    img_table_maude_bgsub.sort(["TIME", "IMGNUM"])
    img_table_cxc.sort(["TIME", "IMGNUM"])
    img_table_cxc_bgsub.sort(["TIME", "IMGNUM"])

    assert np.allclose(img_table_maude["IMG"], img_table_cxc["IMG"])
    assert np.allclose(img_table_maude["TIME"], img_table_cxc["TIME"])
    assert np.allclose(img_table_maude_bgsub["IMG"], img_table_cxc_bgsub["IMG"])
    assert np.allclose(img_table_maude_bgsub["TIME"], img_table_cxc_bgsub["TIME"])
