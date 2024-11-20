import numba
import numpy as np
import requests.exceptions
import scipy.signal
from cheta import fetch_sci
from ska_helpers import retry
from ska_numpy import smooth

from chandra_aca.dark_model import dark_temp_scale_img


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_tccd_data(
    times, smooth_window=30, median_window=3, source="maude", maude_channel=None
):
    """
    Get the CCD temperature for given times and interpolate and smooth.

    This is a light wrapper around cheta.fetch_sci.Msid("aacccdpt", start, stop)
    to handle an option to use the maude data source in an explicit way.

    Parameters
    ----------
    times: np.array float of Chandra seconds
        The times for which to get the CCD temperature data.
    source : str, optional
        Source of the CCD temperature data. If 'maude', override the fetch_sci data source
        to 'maude allow_subset=False'. Else, use the cheta default.
    median_window : int, optional
        3-sample Median filter to to remove outliers.
    smooth_window : int, optional
        Smooth the data using a hanning window of this length in samples.


    Returns
    -------
    vals : np.array
        CCD temperature values.
    """

    pad = 600  # 600 seconds default pad

    # increase the pad if the smooth window is large
    pad = np.max([smooth_window * 32.8 / 2, pad])

    fetch_start = times[0] - pad
    fetch_stop = times[-1] + pad

    if source == "maude":
        # Override the cheta data_source to be explicit about maude source.
        data_source = "maude allow_subset=False"
        if maude_channel is not None:
            data_source += f" channel={maude_channel}"
        with fetch_sci.data_source(data_source):
            dat = fetch_sci.Msid("aacccdpt", fetch_start, fetch_stop)
    else:
        dat = fetch_sci.Msid("aacccdpt", fetch_start, fetch_stop)

    yin = dat.vals.copy()

    if median_window > 0:
        # Filter the data using a 3 point median filter from scipy
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
        The table of ACA images.
    t_ccd_vals : np.array
        The CCD temperature values at the times of the ACA images.
    img_dark : np.array
        The dark calibration image. Must be 1024x1024.
    tccd_dark : float
        The reference temperature of the dark calibration image.

    Returns
    -------
    tuple (imgs_bgsub, imgs_dark)
        imgs_bgsub : np.array
            The background subtracted ACA images.
        imgs_dark : np.array
            The dark current images.
    """
    imgs_dark = get_dark_current_imgs(img_table, img_dark, tccd_dark, t_ccd_vals)
    imgs_bgsub = img_table["IMG"] - imgs_dark
    imgs_bgsub.clip(0, None)

    return imgs_bgsub, imgs_dark


def get_dark_current_imgs(img_table, img_dark, tccd_dark, t_ccds):
    """
    Get the scaled dark current values for a table of ACA images.

    Parameters
    ----------
    img_table : astropy.table.Table
    img_dark : 1024x1024 array
        The dark calibration image.
    tccd_dark : float
        The reference temperature of the dark calibration image.
    t_ccds : array
        The cheta temperature values at the times of img_table.

    Returns
    -------
    imgs_dark : np.array
        An array containing the temperature scaled dark current for each ACA image
        in the img_table in e-/s.

    """
    if len(img_table) != len(t_ccds):
        raise ValueError("img_table and t_ccds must have the same length")

    if img_dark.shape != (1024, 1024):
        raise ValueError("Dark image shape is not 1024x1024")

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
    imgs_dark : np.array
        The dark backgrounds for the ACA images sampled from raw_dark_img.
        This will have the same length as imgrow0 and imgcol0, with shape
        (len(imgrow0), size, size).  These pixels have not been scaled.
    """

    # Borrowed from the agasc code
    @numba.jit(nopython=True)
    def staggered_aca_slice(array_in, array_out, row, col):
        for i in np.arange(len(row)):
            if row[i] + size < 1024 and col[i] + size < 1024:
                array_out[i] = array_in[row[i] : row[i] + size, col[i] : col[i] + size]

    imgs_dark = np.zeros([len(imgrow0), size, size], dtype=np.float64)
    staggered_aca_slice(
        raw_dark_img.astype(float), imgs_dark, 512 + imgrow0, 512 + imgcol0
    )
    return imgs_dark
