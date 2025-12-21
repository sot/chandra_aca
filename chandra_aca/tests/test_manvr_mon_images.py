import numpy as np

from chandra_aca.dark_model import dark_temp_scale
from chandra_aca.manvr_mon_images import DN_TO_ELEC, read_manvr_mon_images


def test_read_manvr_mon_images_no_temperature_correction():
    """Test reading without temperature correction"""
    # This also tests specifying a start time before the start of data, which is on
    # 2023:295.
    dat = read_manvr_mon_images(start="2023:100", stop="2023:310", t_ccd_ref=None)
    assert np.allclose(dat["img_raw"] * DN_TO_ELEC, dat["img_corr"])
    assert len(dat) == 46065


def test_read_manvr_mon_images_with_temperature_correction():
    """Test reading without temperature correction"""
    # This also tests specifying a start time before the start of data, which is on
    # 2023:295.
    dat = read_manvr_mon_images(
        start="2025:300",
        stop="2025:302",
        t_ccd_ref=-5.0,
        scale_4c=1.5,
    )
    dark_scale = dark_temp_scale(dat["t_ccd"], t_ccd_ref=-5.0, scale_4c=1.5)
    img_corr = (
        dat["img_raw"] * DN_TO_ELEC * dark_scale.astype(np.float32)[:, None, None, None]
    )
    assert np.allclose(dat["img_corr"], img_corr)

    assert dat.colnames == [
        "time",
        "img_raw",
        "mask",
        "sum_outlier",
        "corr_sum_outlier",
        "bad_pixels",
        "t_ccd",
        "earth_limb_angle",
        "moon_limb_angle",
        "rate",
        "idx_manvr",
        "row0",
        "col0",
        "img_corr",
    ]

    assert dat.info(out=None).pformat() == [
        "      name        dtype    shape   unit format description class  n_bad length",
        "---------------- ------- --------- ---- ------ ----------- ------ ----- ------",
        "            time float64                   .3f             Column     0   6434",
        "         img_raw   int16 (8, 8, 8)                         Column     0   6434",
        "            mask   uint8      (8,)                         Column     0   6434",
        "     sum_outlier    bool      (8,)                         Column     0   6434",
        "corr_sum_outlier    bool      (8,)                         Column     0   6434",
        "      bad_pixels    bool      (8,)                         Column     0   6434",
        "           t_ccd float64                   .2f             Column     0   6434",
        "earth_limb_angle float32                                   Column     0   6434",
        " moon_limb_angle float32                                   Column     0   6434",
        "            rate float32                                   Column     0   6434",
        "       idx_manvr   int32                                   Column     0   6434",
        "            row0   int16      (8,)                         Column     0   6434",
        "            col0   int16      (8,)                         Column     0   6434",
        "        img_corr float32 (8, 8, 8)         .0f             Column     0   6434",
    ]


def test_read_manvr_mon_images_require_same_row_col():
    """Test reading with require_same_row_col option"""
    dat = read_manvr_mon_images(
        start="2025:001",
        stop="2025:002",
        require_same_row_col=True,
    )

    # Check that all row0 and col0 values are the same as the median
    median_row0 = np.median(dat["row0"], axis=0)
    median_col0 = np.median(dat["col0"], axis=0)
    assert np.all(dat["row0"] == median_row0[None, :])
    assert np.all(dat["col0"] == median_col0[None, :])


def test_read_manvr_mon_images_require_same_row_col_false():
    """Test reading with require_same_row_col option"""
    dat = read_manvr_mon_images(
        start="2025:001",
        stop="2025:002",
        require_same_row_col=False,
    )

    # Check for known case where the col0 values differ (in slot 7)
    median_col0 = np.median(dat["col0"], axis=0)
    assert not np.all(dat["col0"] == median_col0[None, :])
