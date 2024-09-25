import numba
import numpy as np
import requests.exceptions
from astropy.table import Table
from cheta import fetch_sci
from cxotime import CxoTimeLike
from mica.archive import aca_l0
from mica.archive.aca_dark import get_dark_cal_props
from ska_helpers import retry

from chandra_aca import maude_decom
from chandra_aca.dark_model import dark_temp_scale_img


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_tccd_data(start: CxoTimeLike, stop: CxoTimeLike, source="maude"):
    """
    Get the CCD temperature for a given time range.

    This is a light wrapper around cheta.fetch_sci.Msid("aacccdpt", start, stop)
    to handle an option to use the maude data source in an explicit way.

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
    t_ccd_vals : np.array
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
    return dat.vals, dat.times


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def _get_dcsub_aca_images(
    start: CxoTimeLike = None,
    stop: CxoTimeLike = None,
    aca_image_table=None,
    t_ccd_vals=None,
    t_ccd_times=None,
    img_dark=None,
    tccd_dark=None,
    source="maude",
):
    """
    Get dark current subtracted ACA images for a given time range.

    This private method has expanded options for unit testing.
    Users should use get_dcsub_aca_images instead.

    Parameters
    ----------
    start : CxoTimeLike, optional
        Start time of the time range. Default is None.
    stop : CxoTimeLike, optional
        Stop time of the time range. Default is None.
    aca_image_table : astropy table of ACA images, optional

    t_ccd_vals : array-like, optional
        Array of CCD temperatures. Default is None.
    t_ccd_times : array-like, optional
        Array of CCD temperature times. Default is None.
    img_dark : array-like, optional
        Dark current image. Default is None.
    tccd_dark : float, optional
        Dark current CCD temperature. Default is None.
    source : str, optional
        Source of the ACA images. Can be 'maude' or 'mica'. Default is 'maude'.

    Returns
    -------
    astropy.table.Table
        Table of aca image data including raw and dark current subtracted images.
        Raw images are in column 'IMG', dark current subtracted images are in column 'IMGBGSUB',
        dark current cut-out images are in column 'IMGDARK', and interpolated
        CCD temperature values are in column 'AACCCDPT'.

    """

    if start is None and aca_image_table is None:
        raise ValueError("start must be defined if aca_image_table is not defined")
    if stop is None and aca_image_table is None:
        raise ValueError("stop must be defined if aca_image_table is not defined")

    if aca_image_table is not None:
        # If supplied an image table, use that, but confirm it has the right pieces.
        assert type(aca_image_table) is Table

        # Require aca_image_table have columns IMG, IMGROW0_8X8, IMGCOL0_8X8, TIME.
        # If this is a table of mica data that doesn't have those columns, see
        # mica.archive.aca_l0.get_aca_images for how to get the data or use that method.
        assert "IMG" in aca_image_table.colnames
        assert "IMGROW0_8X8" in aca_image_table.colnames
        assert "IMGCOL0_8X8" in aca_image_table.colnames
        assert "TIME" in aca_image_table.colnames
        # require that IMGROW0_8X8 and IMGCOL0_8X8 are masked columns
        assert hasattr(aca_image_table["IMGROW0_8X8"], "mask")
        assert hasattr(aca_image_table["IMGCOL0_8X8"], "mask")
        # Make a copy so we don't modify the input table
        img_table = aca_image_table.copy()

        # And it doesn't make sense to also define start and stop so raise errors if so
        if start is not None:
            raise ValueError("Do not define start if aca_image_table is defined")
        if stop is not None:
            raise ValueError("Do not define stop if aca_image_table is defined")
    elif source == "maude":
        img_table = maude_decom.get_aca_images(start, stop)
    elif source == "mica":
        img_table = aca_l0.get_aca_images(start, stop)
    else:
        raise ValueError(
            "aca_image_table must be defined or source must be maude or mica"
        )

    if start is None:
        start = img_table["TIME"].min()
    if stop is None:
        stop = img_table["TIME"].max()

    if img_dark is None:
        dark_data = get_dark_cal_props(
            start, select="nearest", include_image=True, aca_image=False
        )
        img_dark = dark_data["image"]
        tccd_dark = dark_data["ccd_temp"]

    assert img_dark.shape == (1024, 1024)

    if t_ccd_vals is None:
        t_ccd_vals, t_ccd_times = get_tccd_data(start, stop, source=source)

    imgs_dark = get_dark_current_imgs(
        img_table, img_dark, tccd_dark, t_ccd_vals, t_ccd_times
    )
    imgs_bgsub = img_table["IMG"] - imgs_dark
    imgs_bgsub.clip(0, None)

    img_table["IMGBGSUB"] = imgs_bgsub
    img_table["IMGDARK"] = imgs_dark
    img_table["AACCCDPT"] = np.interp(img_table["TIME"], t_ccd_times, t_ccd_vals)

    return img_table


def get_dark_current_imgs(img_table, img_dark, tccd_dark, t_ccd_vals, t_ccd_times):
    """
    Get the scaled dark current values for a table of ACA images.

    Parameters
    ----------
    img_table : astropy.table.Table
        A table containing the ACA images and expected metadata.
    img_dark : 1024x1024 array
        The dark calibration image.
    tccd_dark : float
        The reference temperature of the dark calibration image.
    t_ccd_vals : array
        The cheta temperature values covering the time range of the img_table.
    t_ccd_times : array
        The corresponding times for the temperature values.

    Returns
    -------
    imgs_dark : np.array
        An array containing the temperature scaled dark current for each ACA image
        in the img_table in e-/s.

    """
    assert img_dark.shape == (1024, 1024)
    assert len(t_ccd_vals) == len(t_ccd_times)

    imgs_dark_unscaled = get_dark_backgrounds(
        img_dark,
        img_table["IMGROW0_8X8"].data.filled(1025),
        img_table["IMGCOL0_8X8"].data.filled(1025),
    )
    dt_ccd = np.interp(img_table["TIME"], t_ccd_times, t_ccd_vals)

    imgs_dark = np.zeros((len(img_table), 8, 8), dtype=np.float64)

    # Scale the dark current at each dark cal 8x8 image to the ccd temperature and e-/s
    for i, (img_dark_unscaled, t_ccd_i, img) in enumerate(
        zip(imgs_dark_unscaled, dt_ccd, img_table, strict=True)
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


@retry.retry(exceptions=requests.exceptions.RequestException, delay=5, tries=3)
def get_dcsub_aca_images(
    start: CxoTimeLike = None,
    stop: CxoTimeLike = None,
    aca_image_table=None,
    source="maude",
):
    """
    Get aca images for a given time range.

    One must supply either aca_image_table or start and stop times.  If start and stop
    times are supplied, an aca image table will be fetched from the maude or mica data
    and a table will be returned that includes dark current background subtracted images
    in one of the astropy table columns.  If an aca_image_table is supplied, the images
    within that table will be used as the raw images, and a new table that includes
    dark current background subtracted images will be returned.


    Parameters
    ----------
    start : CxoTimeLike, optional
        Start time of the time range. Default is None.
    stop : CxoTimeLike, optional
        Stop time of the time range. Default is None.
    aca_image_table : astropy.table.Table
        Table including ACA images.
    source : str, optional
        Source of the ACA images. Can be 'maude' or 'mica'. Default is 'maude'.

    Returns
    -------
    astropy.table.Table
        Table of aca image data including raw and dark current subtracted images.
        Raw images are in column 'IMG', dark current subtracted images are in column 'IMGBGSUB',
        dark current cut-out images are in column 'IMGDARK', and interpolated
        CCD temperature values are in column 'AACCCDPT'.

    """
    return _get_dcsub_aca_images(
        start=start,
        stop=stop,
        aca_image_table=aca_image_table,
        source=source,
    )
