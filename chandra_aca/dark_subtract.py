import numba
import numpy as np
import requests.exceptions
from astropy.table import Table
from cheta import fetch_sci
from cxotime import CxoTimeLike
from mica.archive.aca_dark import get_dark_cal_props
from ska_helpers import retry

from chandra_aca import maude_decom
from chandra_aca.dark_model import dark_temp_scale_img


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_tccd_data(start: CxoTimeLike, stop: CxoTimeLike, source="maude"):
    """
    Get the CCD temperature for a given time range.

    This is a light wrapper around cheta.fetch_sci.Msid("aacccdpt", start, stop).

    Parameters
    ----------
    start : CxoTimeLike
        Start time of the time range.
    stop : CxoTimeLike
        Stop time of the time range.
    source : str, optional
        Source of the CCD temperature data. If 'maude', override the fetch_sci data source
        to 'maude allow_subset=False'. Else, use the cheta default.

    Returns
    -------
    t_ccd : np.array
        CCD temperature values.
    t_ccd_times : np.array
        Times corresponding to the CCD temperature values.
    """
    if source == "maude":
        # Override the data_source to be explicit about maude source.
        with fetch_sci.data_source("maude allow_subset=False"):
            dat = fetch_sci.Msid("aacccdpt", start, stop)
    else:
        dat = fetch_sci.Msid("aacccdpt", start, stop)
    t_ccd = dat.vals
    t_ccd_times = dat.times
    return t_ccd, t_ccd_times


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def _get_dcsub_aca_images(
    start=None,
    stop=None,
    aca_image_table=None,
    t_ccd=None,
    t_ccd_times=None,
    dc_img=None,
    dc_tccd=None,
    source="maude",
    full_table=False,
):
    """
    Get dark current subtracted ACA images for a given time range.

    Parameters
    ----------
    start : str, optional
        Start time of the time range. Default is None.
    stop : str, optional
        Stop time of the time range. Default is None.
    aca_image_table : dict, optional
        Dictionary of ACA image tables. Should be keyed by slot. Default is None.
        This is intended for testing.
    t_ccd : array-like, optional
        Array of CCD temperatures. Default is None.
    t_ccd_times : array-like, optional
        Array of CCD temperature times. Default is None.
    dc_img : array-like, optional
        Dark current image. Default is None.
    dc_tccd : float, optional
        Dark current CCD temperature. Default is None.
    source : str, optional
        Source of the ACA images. Can be 'maude' or 'mica'. Default is 'maude'.
    full_table : bool, optional
        If True, return the full table of ACA images with dark and bgsub columns.

    Returns
    -------
    dict
        If full_table - dictionary of aca slot data keyed by slot including raw and
        dark current subtracted images.
        If not full_table - dictionary of dark current subtracted ACA images keyed by slot.

    Raises
    ------
    ValueError
        If the dark calibration does not have t_ccd or ccd_temp.
        If aca_image_table is not defined or source is not 'maude' or 'mica'.

    """

    if aca_image_table is not None:
        # If supplied an image table, use that, but confirm it has the right pieces.
        assert type(aca_image_table) is Table
        # Require aca_image_table have columns IMG, IMGROW0_8X8, IMGCOL0_8X8, TIME
        assert "IMG" in aca_image_table.colnames
        assert "IMGROW0_8X8" in aca_image_table.colnames
        assert "IMGCOL0_8X8" in aca_image_table.colnames
        assert "TIME" in aca_image_table.colnames
        # require that IMGROW0_8X8 and IMGCOL0_8X8 are masked columns
        assert hasattr(aca_image_table["IMGROW0_8X8"], "mask")
        assert hasattr(aca_image_table["IMGCOL0_8X8"], "mask")
        # Make a copy so we don't modify the input table
        img_table = aca_image_table.copy()
    elif source == "maude":
        if start is None or stop is None:
            raise ValueError("start and stop must be defined for maude source")
        img_table = maude_decom.get_aca_images(start, stop)
    elif source == "mica":
        raise NotImplementedError("mica source not yet implemented")
    else:
        raise ValueError(
            "aca_image_table must be defined or source must be maude or mica"
        )

    if dc_img is not None:
        assert dc_img.shape == (1024, 1024)

    if start is None:
        start = img_table["TIME"].min()
    if stop is None:
        stop = img_table["TIME"].max()

    if dc_img is None:
        dark_data = get_dark_cal_props(
            start, select="nearest", include_image=True, aca_image=False
        )
        dc_img = dark_data["image"]
        dc_tccd = dark_data["ccd_temp"]

    if t_ccd is None:
        t_ccd, t_ccd_times = get_tccd_data(start, stop, source=source)

    imgs_bgsub = {}
    imgs_dark = {}

    imgs_dark = get_dark_current_imgs(img_table, dc_img, dc_tccd, t_ccd, t_ccd_times)
    imgs_bgsub = img_table["IMG"] - imgs_dark
    imgs_bgsub.clip(0, None)

    if full_table:
        img_table["IMGBGSUB"] = imgs_bgsub
        img_table["IMGDARK"] = imgs_dark
        img_table["AACCCDPT"] = np.interp(img_table["TIME"], t_ccd_times, t_ccd)

    if not full_table:
        return imgs_bgsub
    else:
        return img_table


def get_dark_current_imgs(img_table, dc_img, dc_tccd, t_ccd, t_ccd_times):
    """
    Get the scaled dark current values for a table of ACA images.

    Parameters
    ----------
    img_table : astry.table.Table
        A table containing information about the images.
    dc_img : 1024x1024 array
        The dark calibration image.
    dc_tccd : float
        The reference temperature of the dark calibration image.
    t_ccd : array
        The temperature values covering the time range of the img_table.
    t_ccd_times : type
        The corresponding times for the temperature values.

    Returns
    -------
    dark : np.array
        An array containing the temperature scaled dark current for each ACA image
        in the img_table in e-/s.

    """
    assert dc_img.shape == (1024, 1024)
    assert len(t_ccd) == len(t_ccd_times)

    dark_raw = get_dark_backgrounds(
        dc_img,
        img_table["IMGROW0_8X8"].data.filled(1025),
        img_table["IMGCOL0_8X8"].data.filled(1025),
    )
    dt_ccd = np.interp(img_table["TIME"], t_ccd_times, t_ccd)

    dark = np.zeros((len(dark_raw), 8, 8), dtype=np.float64)

    # Scale the dark current at each dark cal 8x8 image to the ccd temperature and e-/s
    for i, (eight, t_ccd_i, img) in enumerate(
        zip(dark_raw, dt_ccd, img_table, strict=True)
    ):
        img_sc = dark_temp_scale_img(eight, dc_tccd, t_ccd_i)
        # Default to using 1.696 integ time if not present in the image table
        integ = img["INTEG"] if "INTEG" in img.colnames else 1.696
        dark[i] = img_sc * integ / 5

    return dark


def get_dark_backgrounds(raw_dark_img, imgrow0, imgcol0, size=8):
    """
    Get the dark background for a stack/table of ACA image.

    Parameters
    ----------
    raw_dark_img : np.array
        The dark calibration image.
    imgrow0 : np.array
        The row of the ACA image.
    imgcol0 : np.array
        The column of the ACA image.
    size : int, optional
        The size of the ACA image. Default is 8.

    Returns
    -------
    dark_img : np.array
        The dark backgrounds for the ACA images.
    """

    # Borrowed from the agasc code
    @numba.jit(nopython=True)
    def staggered_aca_slice(array_in, array_out, row, col):
        for i in np.arange(len(row)):
            if row[i] + size < 1024 and col[i] + size < 1024:
                array_out[i] = array_in[row[i] : row[i] + size, col[i] : col[i] + size]

    dark_img = np.zeros([len(imgrow0), size, size], dtype=np.float64)
    staggered_aca_slice(
        raw_dark_img.astype(float), dark_img, 512 + imgrow0, 512 + imgcol0
    )
    return dark_img


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_aca_images(
    start=None,
    stop=None,
    aca_image_table=None,
    source="maude",
):
    """
    Get aca images for a given time range.

    Parameters
    ----------
    start : str, optional
        Start time of the time range. Default is None.
    stop : str, optional
        Stop time of the time range. Default is None.
    aca_images : dict, optional
        Dictionary of ACA images. Should be keyed by slot. Default is None.
    source : str, optional
        Source of the ACA images. Can be 'maude' or 'mica'. Default is 'maude'.

    Returns
    -------
    dict
        Dictionary of dark current subtracted ACA images for each slot.

    Raises
    ------
    ValueError
        If aca_image_table is not defined or source is not 'maude' or 'mica'.

    """
    return _get_dcsub_aca_images(
        start=start,
        stop=stop,
        aca_image_table=aca_image_table,
        source=source,
        full_table=True,
    )
