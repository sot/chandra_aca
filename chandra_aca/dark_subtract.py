from functools import lru_cache

import numba
import numpy as np
import requests.exceptions
from astropy.table import Table, vstack
from cheta import fetch_sci
from cxotime import CxoTime, CxoTimeLike
from mica.archive import aca_l0
from mica.archive.aca_dark import get_dark_cal_props
from ska_helpers import retry

from chandra_aca import maude_decom
from chandra_aca.dark_model import dark_temp_scale_img


def get_dark_data(time: CxoTimeLike):
    """
    Retrieve the dark calibration image and temperature for a given time.

    This is a light wrapper around mica.archive.aca_dark.get_dark_cal_props.

    Parameters
    ----------
    time : CxoTimeLike
        Reference time for the dark calibration image.

    Returns
    -------
    dc_img : np.array
        Dark calibration image.
    dc_tccd : float
        Dark calibration CCD temperature.
    """
    dc = get_dark_cal_props(time, "nearest", include_image=True, aca_image=False)
    dc_img = dc["image"]
    if "t_ccd" in dc:
        dc_tccd = dc["t_ccd"]
    elif "ccd_temp" in dc:
        dc_tccd = dc["ccd_temp"]
    else:
        raise ValueError("Dark cal must have t_ccd or ccd_temp")
    return dc_img, dc_tccd


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
def get_dcsub_aca_images(
    start=None,
    stop=None,
    aca_images=None,
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
    aca_images : dict, optional
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
    if dc_img is not None:
        assert dc_img.shape == (1024, 1024)

    if dc_img is None:
        dc_img, dc_tccd = get_dark_data(start)

    if t_ccd is None:
        t_ccd, t_ccd_times = get_tccd_data(start, stop, source=source)

    if aca_images is not None:
        # If supplied a dictionary of aca images, use that.  Should be keyed by slot.
        # This is for testing.
        assert isinstance(aca_images, dict)
        # For each slot, require aca_image have columns IMG, IMGROW0_8X8, IMGCOL0_8X8, TIME
        for slot in aca_images:
            assert "IMG" in aca_images[slot].colnames
            assert "IMGROW0_8X8" in aca_images[slot].colnames
            assert "IMGCOL0_8X8" in aca_images[slot].colnames
            assert "TIME" in aca_images[slot].colnames
            # require that IMGROW0_8X8 and IMGCOL0_8X8 are masked columns
            assert hasattr(aca_images[slot]["IMGROW0_8X8"], "mask")
            assert hasattr(aca_images[slot]["IMGCOL0_8X8"], "mask")
        imgs = aca_images.copy()
    elif source == "maude":
        imgs = get_maude_images(start, stop)
    elif source == "mica":
        imgs = get_mica_images(start, stop)
    else:
        raise ValueError(
            "aca_image_table must be defined or source must be maude or mica"
        )

    imgs_bgsub = {}
    dark_imgs = {}
    for slot in range(8):
        if slot not in imgs:
            continue
        dark_imgs[slot] = get_dark_current_imgs(
            imgs[slot], dc_img, dc_tccd, t_ccd, t_ccd_times
        )
        img_col = "IMG"
        imgs_bgsub[slot] = imgs[slot][img_col] - dark_imgs[slot]
        imgs_bgsub[slot].clip(0, None)
        if full_table:
            imgs[slot]["IMGBGSUB"] = imgs_bgsub[slot]
            imgs[slot]["IMGDARK"] = dark_imgs[slot]
            imgs[slot]["AACCCDPT"] = np.interp(imgs[slot]["TIME"], t_ccd_times, t_ccd)

    if not full_table:
        return imgs_bgsub
    else:
        return imgs


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


@lru_cache(maxsize=2)
def get_mica_images(start: CxoTimeLike, stop: CxoTimeLike, cols=None):
    """
    Get ACA images from mica for a given time range.

    These mica images are centered in the 8x8 images via the centered_8x8=True option
    to aca_l0.get_slot_data to match the maude images.

    Parameters
    ----------
    start : CxoTimeLike
        Start time of the time range.
    stop : CxoTimeLike
        Stop time of the time range.

    Returns
    -------
    dict
        dict with a astropy.table of ACA image data for each slot.

    """
    if cols is None:
        cols = [
            "TIME",
            "QUALITY",
            "IMGFUNC1",
            "IMGRAW",
            "IMGSIZE",
            "IMGROW0",
            "IMGCOL0",
            "BGDAVG",
            "INTEG",
        ]
    else:
        for col in ["TIME", "IMGRAW", "IMGROW0", "IMGCOL0"]:
            if col not in cols:
                cols.append(col)

    slot_data = {}
    for slot in range(8):
        slot_data[slot] = aca_l0.get_slot_data(
            start,
            stop,
            imgsize=[4, 6, 8],
            slot=slot,
            columns=cols,
            centered_8x8=True,
        )
        slot_data[slot] = Table(slot_data[slot])

        # Rename and rejigger colums to match maude
        IMGROW0_8X8 = slot_data[slot]["IMGROW0"].copy()
        IMGCOL0_8X8 = slot_data[slot]["IMGCOL0"].copy()
        IMGROW0_8X8[slot_data[slot]["IMGSIZE"] == 6] -= 1
        IMGCOL0_8X8[slot_data[slot]["IMGSIZE"] == 6] -= 1
        IMGROW0_8X8[slot_data[slot]["IMGSIZE"] == 4] -= 2
        IMGCOL0_8X8[slot_data[slot]["IMGSIZE"] == 4] -= 2
        slot_data[slot]["IMGROW0_8X8"] = IMGROW0_8X8
        slot_data[slot]["IMGCOL0_8X8"] = IMGCOL0_8X8
        slot_data[slot].rename_column("IMGRAW", "IMG")

    return slot_data


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
@lru_cache(maxsize=2)
def get_maude_images(start: CxoTimeLike, stop: CxoTimeLike, fetch_limit_hours=100):
    """
    Get ACA images from MAUDE for a given time range.

    Parameters
    ----------
    start : CxoTimeLike
        Start time of the time range.
    stop : CxoTimeLike
        Stop time of the time range.
    fetch_limit_hours : int
        Maximum number of hours to fetch from MAUDE in one go.

    Returns
    -------
    dict
        dict with a astropy.table of ACA image data for each slot.
    """
    tstart = CxoTime(start).secs
    tstop = CxoTime(stop).secs

    if tstop - tstart > fetch_limit_hours * 60 * 60:
        raise ValueError("Time range too large for maude fetch, limit is 100 hours")

    slot_chunks = []
    # Break maude fetches into max 3 hour chunks required by maude_decom fetch
    for mstart in np.arange(tstart, tstop, 60 * 60 * 3):
        mstop = np.min([tstop, mstart + 60 * 60 * 3])
        # Add a retry here if necessary

        imgs = maude_decom.get_aca_images(mstart, mstop)
        slot_chunks.append(imgs)
    slot_chunks = vstack(slot_chunks)

    slot_data = {}
    for slot in range(8):
        slot_data[slot] = slot_chunks[slot_chunks["IMGNUM"] == slot]

    return slot_data


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_aca_images(
    start=None,
    stop=None,
    aca_images=None,
    t_ccd=None,
    t_ccd_times=None,
    dc_img=None,
    dc_tccd=None,
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

    Returns
    -------
    dict
        Dictionary of dark current subtracted ACA images for each slot.

    Raises
    ------
    ValueError
        If the dark calibration does not have t_ccd or ccd_temp.
        If aca_image_table is not defined or source is not 'maude' or 'mica'.

    """
    return get_dcsub_aca_images(
        start=start,
        stop=stop,
        aca_images=aca_images,
        t_ccd=t_ccd,
        t_ccd_times=t_ccd_times,
        dc_img=dc_img,
        dc_tccd=dc_tccd,
        source=source,
        full_table=True,
    )
