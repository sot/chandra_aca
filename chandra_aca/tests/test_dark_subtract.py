from pathlib import Path

import cheta.fetch
import mica.common
import numpy as np
import pytest
from astropy.table import Table
from mica.archive.aca_dark import get_dark_cal_props
from numpy import ma

from chandra_aca import maude_decom
from chandra_aca.dark_subtract import (
    _get_dcsub_aca_images,
    get_dark_backgrounds,
    get_dark_current_imgs,
    get_tccd_data,
)

HAS_ACA0_ARCHIVE = (Path(mica.common.MICA_ARCHIVE) / "aca0").exists()


# Make a test fixture for the mock dark current image
@pytest.fixture
def mock_dc():
    """
    Make a mock dark current image with some values in the corner.
    """
    mock_dc = np.zeros((1024, 1024))
    # put some ints in the corner
    for i in range(16):
        for j in range(16):
            mock_dc[i, j] = i + j
    return mock_dc


# Make a test fixture for the mock 8x8 data
@pytest.fixture
def mock_img_table():
    """
    Make a mock 8x8 image table.

    The intent is that these image coordinates should overlap with the mock dark current.
    """
    img_table = Table()
    img_table["TIME"] = np.arange(8)
    img_table["IMGROW0_8X8"] = ma.arange(8) - 512
    img_table["IMGCOL0_8X8"] = ma.arange(8) - 512
    img_table["IMG"] = np.ones((8, 8, 8)) * 16 * 1.696 / 5
    img_table["IMGNUM"] = 0
    return img_table


# Test fixture array of dark current images in DN
@pytest.fixture
def dc_imgs_dn():
    """
    Save expected results of extracted dark current images in this array.
    """
    dc_imgs_dn = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            ],
            [
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            ],
            [
                [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            ],
            [
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            ],
            [
                [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
                [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            ],
            [
                [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
                [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            ],
            [
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
                [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
                [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            ],
            [
                [14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
                [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
                [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
                [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
                [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
            ],
        ]
    )
    return dc_imgs_dn


def test_dcsub_aca_images(mock_dc, mock_img_table, dc_imgs_dn):
    """
    Confirm that the pattern of background subtraction is correct.
    """

    imgs_bgsub = _get_dcsub_aca_images(
        aca_image_table=mock_img_table,
        dc_img=mock_dc,
        dc_tccd=-10,
        t_ccd=np.repeat(-10, 8),
        t_ccd_times=mock_img_table["TIME"],
    )
    assert imgs_bgsub.shape == (8, 8, 8)
    # Note that the mock unsubtracted data is originally 16 * 1.696 / 5
    exp = 16 - dc_imgs_dn
    assert np.allclose(imgs_bgsub * 5 / 1.696, exp, atol=1e-6, rtol=0)


def test_get_tccd_data():
    start = "2021:001:00:00:00.000"
    stop = "2021:001:00:00:30.000"
    t_ccd_maude, t_ccd_times_maude = get_tccd_data(start, stop, source="maude")
    assert np.isclose(t_ccd_maude[0], -6.55137778)

    # Confirm the t_ccd values are the same for maude as default fetch data source
    # but only bother if cxc is in the sources.  Technically this should be a skipif.
    if "cxc" in cheta.fetch.data_source.sources():
        t_ccd, t_ccd_times = get_tccd_data(start, stop)
        assert np.allclose(t_ccd_maude, t_ccd)
        assert np.allclose(t_ccd_times_maude, t_ccd_times)


def test_get_aca_images(mock_dc, mock_img_table, dc_imgs_dn):
    """
    Confirm the pattern of dark current images matches the reference.
    """
    dat = _get_dcsub_aca_images(
        aca_image_table=mock_img_table,
        dc_img=mock_dc,
        dc_tccd=-10,
        t_ccd=np.repeat(-10, 8),
        t_ccd_times=mock_img_table["TIME"],
        full_table=True,
    )
    for col in ["IMG", "IMGROW0_8X8", "IMGCOL0_8X8", "TIME"]:
        assert col in dat.colnames
        assert np.allclose(dat[col], mock_img_table[col], atol=1e-6, rtol=0)
    assert dat["IMGBGSUB"].shape == (8, 8, 8)
    exp = 16 - dc_imgs_dn
    assert np.allclose(dat["IMGBGSUB"] * 5 / 1.696, exp, atol=1e-6, rtol=0)


def test_get_dark_images(mock_dc, mock_img_table, dc_imgs_dn):
    """
    Confirm the pattern of dark current images matches the reference.
    The temperature scaling is set here to be a no-op.
    """
    dc_imgs = get_dark_current_imgs(
        img_table=mock_img_table,
        dc_img=mock_dc,
        dc_tccd=-10,
        t_ccd=np.repeat(-10, 8),
        t_ccd_times=mock_img_table["TIME"],
    )
    assert dc_imgs.shape == (8, 8, 8)
    dc_imgs_raw = dc_imgs * 5 / 1.696
    assert np.allclose(dc_imgs_raw, dc_imgs_dn, atol=1e-6, rtol=0)


def test_get_dark_backgrounds(mock_dc, mock_img_table, dc_imgs_dn):
    """
    Confirm the pattern of dark current matches the reference.
    """
    dc_imgs_raw = get_dark_backgrounds(
        mock_dc,
        mock_img_table["IMGROW0_8X8"].data.filled(1025),
        mock_img_table["IMGCOL0_8X8"].data.filled(1025),
    )
    assert dc_imgs_raw.shape == (8, 8, 8)
    assert np.allclose(dc_imgs_raw, dc_imgs_dn, atol=1e-6, rtol=0)


def test_dc_consistent():
    """
    Confirm extracted dark current is reasonably consistent with the images
    edges of a 6x6.
    """
    tstart = 746484519.429
    img_table = maude_decom.get_aca_images(tstart, tstart + 30)
    # This slot and this time show an overall decent correlation of dark current
    # and edge pixel values in the 6x6 image - but don't make a great test.
    slot = 4
    img_table_slot = img_table[img_table["IMGNUM"] == slot]
    img = img_table_slot[0]["IMG"]

    dc = get_dark_cal_props(tstart, "nearest", include_image=True, aca_image=False)
    dc_imgs = get_dark_current_imgs(
        img_table_slot,
        dc_img=dc["image"],
        dc_tccd=dc["t_ccd"],
        t_ccd=np.repeat(-9.9, len(img_table_slot)),
        t_ccd_times=img_table_slot["TIME"],
    )
    edge_mask_6x6 = np.zeros((8, 8), dtype=bool)
    edge_mask_6x6[1, 2:6] = True
    edge_mask_6x6[6, 2:6] = True
    edge_mask_6x6[2:6, 1] = True
    edge_mask_6x6[2:6, 6] = True

    # Here just show that a lot of pixels are within 50 e-/s of the dark current
    # In the 6x6, star spoiling is a large effect.
    assert (
        np.percentile(img[edge_mask_6x6].filled(0) - dc_imgs[0][edge_mask_6x6], 50) < 50
    )


def test_dcsub_aca_images_maude():
    """
    Test that dark current background subtracted images match
    references saved in the test.
    """
    # This one is just a regression test
    tstart = 471139130.93277466  # dwell tstart for obsid 15600 - random obsid
    tstop = tstart + 20
    imgs_bgsub = _get_dcsub_aca_images(tstart, tstop, source="maude")

    exp0 = np.array(
        [
            [7, 8, 22, 49, 71, 75, 23, 17],
            [22, 31, 33, 75, 125, 83, 54, 24],
            [28, 50, 166, 367, 491, 311, 144, 45],
            [48, 152, 748, 4124, 5668, 1548, 435, 94],
            [50, 202, 966, 5895, 8722, 2425, 615, 141],
            [29, 88, 331, 999, 1581, 631, 382, 103],
            [16, 36, 108, 279, 302, 229, 112, 62],
            [14, 14, 33, 69, 71, 49, 33, 25],
        ]
    )
    np.allclose(exp0, imgs_bgsub[0][0].filled(0), atol=1, rtol=0)

    # slot 4 is 6x6 and masked
    exp4 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 35, 41, 51, 63, 0, 0],
            [0, 84, 301, 572, 498, 124, 56, 0],
            [0, 311, 1982, 4991, 4314, 475, 116, 0],
            [0, 773, 4551, 3157, 4647, 687, 100, 0],
            [0, 617, 2586, 3120, 2617, 296, 55, 0],
            [0, 0, 425, 1372, 344, 70, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    np.allclose(exp4, imgs_bgsub[4][0].filled(0), atol=1, rtol=0)
