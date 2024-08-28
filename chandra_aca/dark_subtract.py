from functools import lru_cache

import numba
import numpy as np
from astropy.table import Table, vstack
from cheta import fetch_sci
from cxotime import CxoTime, CxoTimeLike
from mica.archive import aca_l0
from mica.archive.aca_dark import get_dark_cal_props

from chandra_aca import maude_decom
from chandra_aca.dark_model import dark_temp_scale_img


@lru_cache(maxsize=2)
def get_mica_images(start: CxoTimeLike, stop: CxoTimeLike):
    """
    Get ACA images from MICA for a given time range.

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
    slot_data = {}
    for slot in range(8):
        slot_data[slot] = aca_l0.get_slot_data(
            start,
            stop,
            imgsize=[4, 6, 8],
            slot=slot,
            columns=[
                "TIME",
                "IMGRAW",
                "IMGSIZE",
                "IMGROW0",
                "IMGCOL0",
            ],
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
        imgs = maude_decom.get_aca_images(mstart, mstop)
        slot_chunks.append(imgs)
    slot_chunks = vstack(slot_chunks)

    slot_data = {}
    for slot in range(8):
        slot_data[slot] = slot_chunks[slot_chunks["IMGNUM"] == slot]

    return slot_data


def get_dcsub_aca_images(
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
    Get dark current subtracted ACA images for a given time range.

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
    if dc_img is not None:
        assert dc_img.shape == (1024, 1024)

    if dc_img is None:
        dc = get_dark_cal_props(start, "nearest", include_image=True, aca_image=False)
        dc_img = dc["image"]
        if "t_ccd" in dc:
            dc_tccd = dc["t_ccd"]
        elif "ccd_temp" in dc:
            dc_tccd = dc["ccd_temp"]
        else:
            raise ValueError("Dark cal must have t_ccd or ccd_temp")

    if t_ccd is None:
        if source == "maude":
            with fetch_sci.data_source("maude allow_subset=False"):
                dat = fetch_sci.Msid("aacccdpt", start, stop)
        else:
            with fetch_sci.data_source("cxc"):
                dat = fetch_sci.Msid("aacccdpt", start, stop)
        t_ccd = dat.vals
        t_ccd_times = dat.times

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
        imgs = aca_images
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

    return imgs_bgsub


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
        The temperature values for the time range of the img_table.
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
    for i, (eight, t_ccd_i) in enumerate(zip(dark_raw, dt_ccd, strict=True)):
        img_sc = dark_temp_scale_img(eight, dc_tccd, t_ccd_i)
        # Note that this just uses standard integration time of 1.696 sec
        dark[i] = img_sc * 1.696 / 5

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
