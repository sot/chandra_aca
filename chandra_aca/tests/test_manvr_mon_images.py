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
        "idx_manvr",
        "img_corr",
    ]

    assert dat.info(out=None).pformat() == [
        "      name        dtype    shape   unit format description class  n_bad length",
        "---------------- ------- --------- ---- ------ ----------- ------ ----- ------",
        "            time float64                                   Column     0   6434",
        "         img_raw   int16 (8, 8, 8)                         Column     0   6434",
        "            mask   uint8      (8,)                         Column     0   6434",
        "     sum_outlier    bool      (8,)                         Column     0   6434",
        "corr_sum_outlier    bool      (8,)                         Column     0   6434",
        "      bad_pixels    bool      (8,)                         Column     0   6434",
        "           t_ccd float64                                   Column     0   6434",
        "       idx_manvr   int32                                   Column     0   6434",
        "        img_corr float32 (8, 8, 8)                         Column     0   6434",
    ]
