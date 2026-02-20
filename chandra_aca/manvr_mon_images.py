"""Read ACA maneuver monitor window image data"""

# Standard library imports
import enum
import itertools
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

    SUM_OUTLIER = np.uint8(1)
    CORR_SUM_OUTLIER = np.uint8(2)
    HAS_BAD_PIX = np.uint8(4)
    EARTH_VIOLATION = np.uint8(8)
    MOON_VIOLATION = np.uint8(16)
    RATE_VIOLATION = np.uint8(32)
    CCD_RECOVERY = np.uint8(64)


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
    filter_constraints: bool | dict = False,
    require_same_row_col: bool = True,
    exact_interval: bool = False,
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
    filter_constraints : bool or dict, optional
        If True, filter images based on Earth / moon limb angle and rate constraints. If
        a dict, use the dict as keyword arguments to get_constraint_flags() to specify
        custom constraints. Default is False.
    require_same_row_col : bool, optional
        If True, only include images where all slots have the same row0 and col0
        values across the entire time range. This uses the median values to filter.
        Default is True.
    exact_interval : bool, optional
        If True, only include images with times exactly within start and stop.
        Otherwise include all times within the maneuvers that are included within start
        and stop. Default is False.
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
        - mask: combined bit mask for image status flags [slot]
        - sum_outlier: boolean flag for images with total sum outliers [slot]
        - corr_sum_outlier: boolean flag for bgd-subtracted sum outliers [slot]
        - bad_pixels: boolean flag for images with bad pixels [slot]
        - t_ccd: CCD temperatures in Celsius
        - zero_offsets: zero offset values for each quadrant [quad]
        - earth_limb_angle: Earth limb angle in degrees
        - moon_limb_angle: Moon limb angle in degrees
        - rate: spacecraft rate in arcsec/sec
        - idx_manvr: maneuver index for each sample
        - row0: row0 position for each slot [slot]
        - col0: col0 position for each slot [slot]

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
    moon_limb_angles_list = []
    rates_list = []
    idx_manvrs_list = []
    zero_offsets_list = []
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
                zero_offsets_list.append(dat["zero_offsets"])
                earth_limb_angles_list.append(dat["earth_limb_angle"])
                moon_limb_angles_list.append(dat["moon_limb_angle"])
                rates_list.append(dat["rate"])
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
    dat["zero_offsets"] = np.concatenate(zero_offsets_list)
    dat["earth_limb_angle"] = np.concatenate(earth_limb_angles_list)
    dat["moon_limb_angle"] = np.concatenate(moon_limb_angles_list)
    dat["rate"] = np.concatenate(rates_list)
    dat["idx_manvr"] = np.concatenate(idx_manvrs_list)
    dat["row0"] = np.concatenate(row0s_list)
    dat["col0"] = np.concatenate(col0s_list)

    dat["img_corr"] = dat["img_raw"] * DN_TO_ELEC
    if t_ccd_ref is not None:
        dark_scale = dark_temp_scale(dat["t_ccd"], t_ccd_ref, scale_4c=scale_4c)
        dat["img_corr"] *= dark_scale.astype(np.float32)[:, None, None, None]

    # Archived bad_pixels flag is based on img_raw < 0, which is too strict. Many pixels
    # are slightly negative due to background subtraction. Recompute here using -5 DN.
    # Note that slot 5 goes to -15 DN for some reason but we treat all slots the same
    # here.
    bad_pix = np.any(
        (dat["img_raw"] < BAD_PIXEL_LOW) | (dat["img_raw"] > BAD_PIXEL_HIGH),
        axis=(2, 3),
    )
    dat["bad_pixels"] = bad_pix

    # Remake masks now as well
    dat["mask"][:] = np.uint8(0)
    dat["mask"][dat["sum_outlier"]] |= ImgStatus.SUM_OUTLIER.value
    dat["mask"][dat["corr_sum_outlier"]] |= ImgStatus.CORR_SUM_OUTLIER.value
    dat["mask"][dat["bad_pixels"]] |= ImgStatus.HAS_BAD_PIX.value

    i0, i1 = np.searchsorted(dat["time"], [start.secs, stop.secs])
    if not exact_interval:
        idx_manvr0 = dat["idx_manvr"][i0]
        i0 = np.searchsorted(dat["idx_manvr"], idx_manvr0, side="left")
        idx_manvr1 = dat["idx_manvr"][i1 - 1]
        i1 = np.searchsorted(dat["idx_manvr"], idx_manvr1, side="right")
    dat = dat[i0:i1]

    if len(dat) > 0:
        # Make idx_manvr start at 0
        dat["idx_manvr"] -= dat["idx_manvr"][0]

        if require_same_row_col:
            # Compute the median of row0 and col0 across all samples and slots
            # then choose only rows with those values.
            median_row0 = np.median(dat["row0"], axis=0)
            median_col0 = np.median(dat["col0"], axis=0)
            ok = np.all(
                (dat["row0"] == median_row0[None, :])
                & (dat["col0"] == median_col0[None, :]),
                axis=1,
            )
            dat = dat[ok]

    # Formatting
    dat["time"].info.format = ".3f"
    dat["img_corr"].info.format = ".0f"
    dat["t_ccd"].info.format = ".2f"

    if filter_constraints:
        kwargs = filter_constraints if isinstance(filter_constraints, dict) else {}
        flags = get_constraint_flags(dat, **kwargs)
        dat = dat[flags == 0]

    return dat


def get_constraint_flags(
    dat,
    ela_limit: float | None = 5.0,
    mla_limit: float | None = 5.0,
    rate_limit: float | None = 40.0,
    dt_recovery: float = 300.0,
):
    """Get flags of images violating attitude / rate constraints.

    This emulates expected operational constraints on times when background scans will
    be commanded.

    The flags are from the ``image_status`` enum:

    - EARTH_VIOLATION: Earth limb angle below ela_limit
    - MOON_VIOLATION: Moon limb angle below mla_limit
    - RATE_VIOLATION: spacecraft rate above rate_limit
    - CCD_RECOVERY: within dt_recovery seconds after exit from an attitude violation

    Parameters
    ----------
    dat : apt.Table
        Table containing monitor window images as returned by read_manvr_mon_images().
    ela_limit : float or None, optional
        Minimum acceptable Earth limb angle in degrees. Images with smaller angles are
        excluded. If None, no filtering on Earth limb angle is applied. Default is 5.0
        degrees.
    mla_limit : float or None, optional
        Minimum acceptable Moon limb angle in degrees. Images with smaller angles are
        excluded. If None, no filtering on Moon limb angle is applied. Default is 5.0
        degrees.
    rate_limit : float or None, optional
        Minimum acceptable rate (arcsec/sec). Images with smaller rates are excluded. If
        None, no filtering on rate is applied. Default is 40.0.
    dt_recovery : float or None, optional
        Time in seconds after an attitude limb violation during which images are also
        excluded. Default is 300 seconds.

    Returns
    -------
    flags : np.ndarray[np.uint8]
        Bit mask flag array indicating which images to exclude (1) based on constraints.
    """

    n_samp = len(dat)
    flags = np.zeros(n_samp, dtype=np.uint8)

    for ss_obj, angle_limit, mask_value in (
        ("earth", ela_limit, ImgStatus.EARTH_VIOLATION.value),
        ("moon", mla_limit, ImgStatus.MOON_VIOLATION.value),
    ):
        if angle_limit is None:
            continue
        viols = get_att_violations(
            dat,
            ss_obj=ss_obj,
            angle_limit=angle_limit,
            dt_recovery=dt_recovery,
        )
        for viol in viols:
            idx_flag = slice(viol["idx_start"], viol["idx_stop"])
            flags[idx_flag] |= mask_value
            idx_flag = slice(viol["idx_stop"], viol["idx_recovery"])
            flags[idx_flag] |= ImgStatus.CCD_RECOVERY.value

    if rate_limit is not None:
        rate_viol = dat["rate"] < rate_limit
        flags[rate_viol] |= ImgStatus.RATE_VIOLATION.value

    return flags


def get_att_violations_one(
    times,
    limb_angles,
    *,
    idx0: int,
    angle_limit: float,
    dt_recovery: float,
) -> apt.Table | None:
    """Get attitude violiations in a single maneuver segment.

    Parameters
    ----------
    times : np.ndarray
        Array of times in CXO seconds.
    limb_angles : np.ndarray
        Array of limb angles in degrees corresponding to times.
    idx0 : int
        Starting index offset for the times in the full dataset.
    angle_limit : float
        Minimum acceptable limb angle in degrees.
    dt_recovery : float
        Time in seconds after violation exit at which CCD is considered recovered.
        Analysis indicates this is nominally around 4 minutes but can vary.

    Returns
    -------
    apt.Table or None
        Table of attitude violation intervals with columns:
        - tstart: start time of violation interval
        - tstop: stop time of violation interval
        - idx_start: starting index of violation interval in full dataset
        - idx_stop: stopping index of violation interval in full dataset
        - idx_recovery: index at which CCD is considered recovered
        - duration: duration of violation interval in seconds
    """
    import cheta.utils

    att_viols = cheta.utils.logical_intervals(times, limb_angles < angle_limit)
    if len(att_viols) == 0:
        return None

    i0 = np.searchsorted(times, att_viols["tstart"])
    i1 = np.searchsorted(times, att_viols["tstop"])
    i_recovery = np.searchsorted(times, att_viols["tstop"] + dt_recovery)

    att_viols["idx_start"] = i0 + idx0
    att_viols["idx_stop"] = i1 + idx0
    att_viols["idx_recovery"] = i_recovery + idx0

    return att_viols


def get_manvr_indices(mon_imgs: apt.Table) -> np.ndarray:
    """Get indices to split mon_imgs into maneuvers.

    Since the table is already sorted by idx_manvr, this is a faster more-lightweight
    way to get maneuver boundaries than using astropy ``group_by("idx_manvr")``.

    Parameters
    ----------
    mon_imgs : apt.Table
        Table of monitor window images as returned by read_manvr_mon_images().

    Returns
    -------
    np.ndarray
        Array of indices into mon_imgs that mark the start of each maneuver, with
        the final index being the length of mon_imgs. This is conveniently used with
        itertools.pairwise().
    """
    indices_ = np.where(np.diff(mon_imgs["idx_manvr"]) != 0)[0]
    indices = np.concatenate([[0], indices_ + 1, [len(mon_imgs)]])
    return indices


def get_att_violations(
    mon_imgs,
    ss_obj="earth+moon",
    angle_limit=5.0,
    dt_recovery=300.0,
):
    """Get attitude violations in a ``mon_imgs`` table for Earth or moon.

    Parameters
    ----------
    mon_imgs : apt.Table
        Table of monitor window images as returned by read_manvr_mon_images().
    ss_obj : str
        Solar system object to check violations for, either 'earth', 'moon', or
        'earth+moon' (default).
    angle_limit : float
        Minimum acceptable limb angle in degrees.
    dt_recovery : float
        Time in seconds after violation exit at which CCD is considered recovered.
        Analysis indicates this is nominally around 4 minutes but can vary.
        Default is 300 seconds.

    Returns
    -------
    apt.Table
        Table of attitude violation intervals with columns:
        - start: start time of violation interval (date string U21)
        - stop: stop time of violation interval (date string U21)
        - tstart: start time of violation interval (float)
        - tstop: stop time of violation interval (float)
        - duration: duration of violation interval in seconds (float)
        - idx_start: starting index of violation interval in full dataset (int)
        - idx_stop: stopping index of violation interval in full dataset (int)
        - idx_recovery: index at which CCD is considered recovered (int)
    """
    att_viols_list = []

    for i0, i1 in itertools.pairwise(get_manvr_indices(mon_imgs)):
        times = mon_imgs["time"][i0:i1]

        limb_angles = (
            np.minimum(
                mon_imgs["earth_limb_angle"][i0:i1],
                mon_imgs["moon_limb_angle"][i0:i1],
            )
            if ss_obj == "earth+moon"
            else mon_imgs[f"{ss_obj}_limb_angle"][i0:i1]
        )

        att_viol = get_att_violations_one(
            times,
            limb_angles,
            idx0=i0,
            angle_limit=angle_limit,
            dt_recovery=dt_recovery,
        )
        if att_viol is not None:
            att_viol["idx_manvr"] = mon_imgs["idx_manvr"][i0]
            att_viols_list.append(att_viol)

    if len(att_viols_list) == 0:
        # Create zero-length table with the expected columns and dtypes from docstring
        att_viols = apt.Table(
            {
                "start": np.array([], dtype="U21"),
                "stop": np.array([], dtype="U21"),
                "tstart": np.array([], dtype=np.float64),
                "tstop": np.array([], dtype=np.float64),
                "duration": np.array([], dtype=np.float64),
                "idx_start": np.array([], dtype=np.int32),
                "idx_stop": np.array([], dtype=np.int32),
                "idx_recovery": np.array([], dtype=np.int32),
                "idx_manvr": np.array([], dtype=np.int32),
            }
        )
    else:
        att_viols = apt.vstack(att_viols_list)

    return att_viols
