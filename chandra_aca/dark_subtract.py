import numba
import numpy as np
import requests.exceptions
import scipy.signal
from cheta import fetch_sci
from ska_helpers import retry
from ska_numpy import smooth

from chandra_aca.dark_model import dark_temp_scale_img

__all__ = [
    "get_tccd_data",
    "get_aca_images_bgd_sub",
    "get_dark_current_imgs",
    "get_dark_backgrounds",
]


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_tccd_data(
    times, smooth_window=30, median_window=3, source="maude", channel=None
):
    """
    Get the CCD temperature for given times and interpolate and smooth.

    This is a light wrapper around cheta.fetch_sci.Msid("aacccdpt", start, stop)
    to handle an option to use the maude data source in an explicit way.

    Parameters
    ----------
    times: np.array (float)
        Sampling times for CCD temperature data (CXC seconds).
    source : str, optional
        Source of CCD temperature data ('maude' (default) or 'cxc' for cheta archive).
    median_window : int, optional
        Median filter window to remove outliers (default=3).
    smooth_window : int, optional
        Smooth data using a hanning window of this length in samples (default=30).
    channel : str, optional
        Maude channel to use (default is flight).

    Returns
    -------
    vals : np.array
        CCD temperature values.
    """

    # If no times are given, return an empty array.
    if len(times) == 0:
        return np.array([])

    pad = 600  # 600 seconds default pad

    # increase the pad if the smooth window is large
    pad = np.max([smooth_window * 32.8 / 2, pad])

    fetch_start = times[0] - pad
    fetch_stop = times[-1] + pad

    if source == "maude":
        # Override the cheta data_source to be explicit about maude source.
        data_source = "maude allow_subset=False"
        if channel is not None:
            data_source += f" channel='{channel}'"
        with fetch_sci.data_source(data_source):
            dat = fetch_sci.Msid("aacccdpt", fetch_start, fetch_stop)
    elif source == "cxc":
        with fetch_sci.data_source("cxc"):
            dat = fetch_sci.Msid("aacccdpt", fetch_start, fetch_stop)
    else:
        raise ValueError(f"Unknown source: {source}")

    yin = dat.vals.copy()

    if median_window > 0:
        # Filter the data using a median filter from scipy
        yin = scipy.signal.medfilt(yin, kernel_size=median_window)

    if smooth_window > 0:
        # And then smooth it with hanning and window_len = smooth_window
        yin = smooth(yin, window_len=smooth_window)

    # Interpolate the data to the requested times
    vals = np.interp(times, dat.times, yin)

    return vals


def get_aca_images_bgd_sub(img_table, t_ccd_vals, img_dark, tccd_dark):
    """
    Get the background subtracted ACA images.

    Parameters
    ----------
    img_table : astropy.table.Table
        Table of ACA images with columns 'IMG', 'IMGROW0_8X8', 'IMGCOL0_8X8', 'INTEG'.
    t_ccd_vals : np.array
        CCD temperature values at the times of the ACA images (deg C).
    img_dark : np.array
        Dark calibration image. Must be 1024x1024 (e-/s).
    tccd_dark : float
        Reference temperature of the dark calibration image (deg C).

    Returns
    -------
    tuple (imgs_bgsub, imgs_dark)
        imgs_bgsub : np.array
            Background subtracted ACA images (DN).
        imgs_dark : np.array
            Dark current images (DN).
    """
    imgs_dark = get_dark_current_imgs(img_table, img_dark, tccd_dark, t_ccd_vals)
    imgs_bgsub = img_table["IMG"] - imgs_dark
    imgs_bgsub.clip(0, None)

    return imgs_bgsub, imgs_dark


def get_dark_current_imgs(img_table, img_dark, tccd_dark, t_ccds):
    """
    Get the scaled dark current values for a table of ACA images.

    This scales the dark current to the appropriate temperature and integration time,
    returning the dark current in DN matching the ACA images in img_table.

    Parameters
    ----------
    img_table : astropy.table.Table
        Table of ACA images with columns 'IMG', 'IMGROW0_8X8', 'IMGCOL0_8X8', 'INTEG'.
    img_dark : 1024x1024 array
        Dark calibration image.
    tccd_dark : float
        Reference temperature of the dark calibration image.
    t_ccds : array
        Cheta temperature values at the times of img_table.

    Returns
    -------
    imgs_dark : np.array (len(img_table), 8, 8)
        Temperature scaled dark current for each ACA image in img_table (DN).

    """
    if len(img_table) != len(t_ccds):
        raise ValueError("img_table and t_ccds must have the same length")

    if img_dark.shape != (1024, 1024):
        raise ValueError("dark image shape is not 1024x1024")

    imgs_dark_unscaled = get_dark_backgrounds(
        img_dark,
        img_table["IMGROW0_8X8"],
        img_table["IMGCOL0_8X8"],
    )

    imgs_dark = np.zeros((len(img_table), 8, 8), dtype=np.float64)

    # Scale the dark current at each dark cal 8x8 image to the ccd temperature and e-/s
    for i, (img_dark_unscaled, t_ccd_i, img) in enumerate(
        zip(imgs_dark_unscaled, t_ccds, img_table, strict=True)
    ):
        img_scaled = dark_temp_scale_img(img_dark_unscaled, tccd_dark, t_ccd_i)
        # Default to using 1.696 integ time if not present in the image table
        integ = img["INTEG"] if "INTEG" in img.colnames else 1.696
        imgs_dark[i] = img_scaled * integ / 5

    return imgs_dark


CCD_MAX = 1024
CCD_MIN = 0


@numba.jit(nopython=True)
def staggered_aca_slice(array_in, array_out, row, col):
    """Make cutouts of array_in at positions row, col and put them in array_out.

    array_out must be 3D with shape (N, sz_r, sz_c) and be initialized to 0.
    row and col must be 1D arrays of length N.
    """
    for idx in np.arange(array_out.shape[0]):
        for i, r in enumerate(np.arange(row[idx], row[idx] + array_out.shape[1])):
            for j, c in enumerate(np.arange(col[idx], col[idx] + array_out.shape[2])):
                if r >= CCD_MIN and r < CCD_MAX and c >= CCD_MIN and c < CCD_MAX:
                    array_out[idx, i, j] = array_in[r, c]


def get_dark_backgrounds(raw_dark_img, imgrow0, imgcol0, size=8):
    """
    Get dark background cutouts at a set of ACA image positions.

    Parameters
    ----------
    raw_dark_img : np.array
        Dark calibration image.
    imgrow0 : np.array (int)
        Row of ACA image.
    imgcol0 : np.array (int)
        Column of ACA image.
    size : int, optional
        Size of ACA image (default=8).

    Returns
    -------
    imgs_dark : np.array (len(imgrow0), size, size)
        Dark backgrounds for image locations sampled from raw_dark_img (e-/s).
        Pixels outside raw_dark_img are set to 0.0.
    """
    imgs_dark = np.zeros([len(imgrow0), size, size], dtype=np.float64)
    staggered_aca_slice(
        raw_dark_img.astype(float), imgs_dark, 512 + imgrow0, 512 + imgcol0
    )
    return imgs_dark
