"""Read ACA maneuver monitor window image data"""

# Standard library imports
import enum
import os
from pathlib import Path
from typing import Generator

# Third-party imports
import astropy.table as apt
import astropy.units as u
import numpy as np
from cxotime import CxoTime, CxoTimeLike

# Ska/Chandra imports
from chandra_aca.dark_model import dark_temp_scale

DN_TO_ELEC = np.float32(5.0 / 1.696)  # Convert DN (per readout) to e-/s
BAD_PIXEL_LOW = -5  # Limit below which a pixel is considered corrupted (DN)
BAD_PIXEL_HIGH = 30000  # Upper limit for bad pixel (DN)


class ImgStatus(enum.Enum):
    """Bit flags for image masks."""

    SUM_OUTLIER = np.uint8(0b001)
    CORR_SUM_OUTLIER = np.uint8(0b010)
    HAS_BAD_PIX = np.uint8(0b100)


def get_years_doys(
    start: CxoTimeLike, stop: CxoTimeLike
) -> Generator[tuple[str, str], None, None]:
    """Generate (year, doy) tuples for days between start and stop with margin."""
    start = CxoTime(start) - 1.5 * u.d
    stop = CxoTime(stop) + 1.5 * u.d
    date = start
    while date < stop:
        yield date.date[:4], date.date[5:8]
        date += 1 * u.d


def imgs_root_dir_path(data_dir: Path | str | None = None) -> Path:
    """Get root path to store monitor window images."""
    if data_dir is None:
        data_dir = Path(os.environ["SKA"]) / "data" / "manvr_mon_images"
    return Path(data_dir)


def read_manvr_mon_images(  # noqa: PLR0915
    start: CxoTimeLike,
    stop: CxoTimeLike,
    t_ccd_ref: float | None = -10.0,
    scale_4c: float | None = None,
    data_dir: Path | str | None = None,
) -> apt.Table:
    """Read ACA maneuver monitor window images from archived data files.

    This function loads processed monitor window images from the compressed .npz
    archive files created by save_imgs(). It concatenates data across multiple
    files and date ranges, applies temperature correction if requested, and
    returns an astropy Table.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for data retrieval (any format accepted by CxoTime).
    stop : CxoTimeLike
        Stop time for data retrieval (any format accepted by CxoTime).
    t_ccd_ref : float or None, optional
        Reference CCD temperature in Celsius for dark current scaling. If None, no
        temperature correction is applied. Default is -10.0.
    scale_4c : float or None, optional
        Scaling factor in dark current temperature dependence. If None, uses default
        from dark_temp_scale(). Default is None.
    data_dir : Path or str, optional
        Root directory containing the archived image files organized as
        data_dir/YYYY/DOY/*.npz. Default is ``$SKA/data/manvr_mon_images``.

    Returns
    -------
    dat : apt.Table
        Table containing concatenated monitor window data with columns:
        - time: observation times in CXO seconds since 1998.0
        - img_raw: raw monitor window images in DN [slot, row, col]
        - img_corr: corrected images in e-/s with temperature scaling [slot, row, col]
        - sum_outlier: boolean flag for images with total sum outliers [slot]
        - corr_sum_outlier: boolean flag for bgd-subtracted sum outliers [slot]
        - bad_pixels: boolean flag for images with bad pixels [slot]
        - t_ccd: CCD temperatures in Celsius
        - idx_manvr: maneuver index for each sample

    Notes
    -----
    - Raw images in DN are converted to e-/s using factor 5.0/1.696
    - Temperature correction uses dark_temp_scale() from chandra_aca.dark_model
    - Early CCD temperature samples are replaced with 5th sample to avoid artifacts
    """
    start = CxoTime(start)
    stop = CxoTime(stop)
    data_path = imgs_root_dir_path(data_dir)

    imgs_list = []
    masks_list = []
    times_list = []
    row0s_list = []
    col0s_list = []
    t_ccds_list = []
    earth_limb_angles_list = []
    idx_manvrs_list = []
    idx_manvr = 0

    for year, doy in get_years_doys(start, stop):
        dir = Path(data_path, year, doy)
        for path in sorted(dir.glob("*.npz")):
            with np.load(path) as dat:
                n_samp = len(dat["t_ccd"])
                # Move the sample dimension (-1) to the front so they concat properly.
                # Imgs will then be indexed as [sample, slot, row, col].
                imgs_list.append(np.moveaxis(dat["slot_imgs"], -1, 0))
                # Masks will be [sample, slot].
                masks_list.append(np.moveaxis(dat["slot_masks"], -1, 0))
                # Repeat slot_row0s n_samp times for each sample.
                row0s_list.append(np.repeat(dat["slot_row0s"][None, :], n_samp, axis=0))
                col0s_list.append(np.repeat(dat["slot_col0s"][None, :], n_samp, axis=0))
                t_ccds = dat["t_ccd"]
                t_ccds[:5] = t_ccds[5]
                t_ccds_list.append(t_ccds)
                earth_limb_angles_list.append(dat["earth_limb_angle"])
                times_list.append(dat["time0"] + 4.1 * np.arange(n_samp))
                idx_manvrs_list.append(idx_manvr + np.zeros(n_samp, dtype=np.int32))
                idx_manvr += 1

    dat = apt.Table()

    dat["time"] = np.concatenate(times_list)
    dat["img_raw"] = np.concatenate(imgs_list)
    masks = np.concatenate(masks_list)
    dat["mask"] = masks
    dat["sum_outlier"] = (masks & ImgStatus.SUM_OUTLIER.value) != 0
    dat["corr_sum_outlier"] = (masks & ImgStatus.CORR_SUM_OUTLIER.value) != 0
    dat["bad_pixels"] = (masks & ImgStatus.HAS_BAD_PIX.value) != 0
    dat["t_ccd"] = np.concatenate(t_ccds_list)
    dat["earth_limb_angle"] = np.concatenate(earth_limb_angles_list)
    dat["idx_manvr"] = np.concatenate(idx_manvrs_list)
    dat["row0"] = np.concatenate(row0s_list)
    dat["col0"] = np.concatenate(col0s_list)

    ok = (dat["time"] >= start.secs) & (dat["time"] < stop.secs)
    dat = dat[ok]

    # Make idx_manvr start at 0
    dat["idx_manvr"] -= dat["idx_manvr"][0]

    dat["img_corr"] = dat["img_raw"] * DN_TO_ELEC
    if t_ccd_ref is not None:
        dark_scale = dark_temp_scale(dat["t_ccd"], t_ccd_ref, scale_4c=scale_4c)
        dat["img_corr"] *= dark_scale.astype(np.float32)[:, None, None, None]

    # Archived bad_pixels flag is based on img_raw < 0, which is too strict. Many pixels
    # are slightly negative due to background subtraction. Recompute here using -5 DN.
    # Note that slot 5 goes to -15 DN for some reason but we treat all slots the same
    # here.
    bad_pix = np.any((dat["img_raw"] < -5) | (dat["img_raw"] > 30000), axis=(2, 3))
    dat["bad_pixels"] = bad_pix

    # Remake masks now as well
    dat["mask"][:] = np.uint8(0)
    dat["mask"][dat["sum_outlier"]] |= ImgStatus.SUM_OUTLIER.value
    dat["mask"][dat["corr_sum_outlier"]] |= ImgStatus.CORR_SUM_OUTLIER.value
    dat["mask"][dat["bad_pixels"]] |= ImgStatus.HAS_BAD_PIX.value

    # Formatting
    dat["time"].info.format = ".3f"
    dat["img_corr"].info.format = ".0f"
    dat["t_ccd"].info.format = ".2f"

    return dat
