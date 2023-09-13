# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

from contextlib import contextmanager
from functools import wraps

import agasc
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import Quaternion
from astropy.table import Table
from cxotime import CxoTime
from Quaternion import Quat
from Ska.quatutil import radec2yagzag

from chandra_aca.planets import get_planet_chandra, get_planet_eci

from .planets import GET_PLANET_ECI_ERRORS, NoEphemerisError, get_planet_angular_sep
from .transform import eci_to_radec, radec_to_yagzag, yagzag_to_pixels

# rc definitions
frontcolor = "black"
backcolor = "white"
rcParams = {}
rcParams["lines.color"] = frontcolor
rcParams["patch.edgecolor"] = frontcolor
rcParams["text.color"] = frontcolor
rcParams["axes.facecolor"] = backcolor
rcParams["axes.edgecolor"] = frontcolor
rcParams["axes.labelcolor"] = frontcolor
rcParams["xtick.color"] = frontcolor
rcParams["ytick.color"] = frontcolor
rcParams["grid.color"] = frontcolor
rcParams["figure.facecolor"] = backcolor
rcParams["figure.edgecolor"] = backcolor
rcParams["savefig.facecolor"] = backcolor
rcParams["savefig.edgecolor"] = backcolor

# Classic grid params https://matplotlib.org/users/dflt_style_changes.html#grid-lines
rcParams["grid.color"] = "k"
rcParams["grid.linestyle"] = ":"
rcParams["grid.linewidth"] = 0.5

BAD_STAR_COLOR = "tomato"
BAD_STAR_ALPHA = 0.75
FAINT_STAR_COLOR = "lightseagreen"
FAINT_STAR_ALPHA = 0.75


__all__ = ["plot_stars", "plot_compass", "bad_acq_stars"]


@contextmanager
def custom_plt():
    orig = {key: plt.rcParams[key] for key in rcParams}
    plt.rcParams.update(rcParams)
    yield
    plt.rcParams.update(orig)


def custom_plt_rcparams(func):
    """
    Decorator to make a function use the custom rcParams plt params
    temporarily.  This uses a context manage to ensure original
    params always get restored.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with custom_plt():
            return func(*args, **kwargs)

    return wrapper


def symsize(mag):
    # map mags to figsizes, defining
    # mag 6 as 40 and mag 11 as 3
    # interp should leave it at the bounding value outside
    # the range
    return np.interp(mag, [6.0, 11.0], [40.0, 3.0])


def _plot_catalog_items(ax, catalog):
    """
    Plot catalog items (guide, acq, bot, mon, fid) in yang and zang on the supplied
    axes object in place.

    Parameters
    ----------
    ax
        matplotlib axes
    catalog : data structure containing starcheck-style columns/attributes
        catalog records.  This can be anything that will work with
        astropy.table.Table(catalog).  A list of dicts is the convention.
    """
    cat = Table(catalog)
    cat["row"], cat["col"] = yagzag_to_pixels(cat["yang"], cat["zang"], allow_bad=True)
    gui_stars = cat[(cat["type"] == "GUI") | (cat["type"] == "BOT")]
    acq_stars = cat[(cat["type"] == "ACQ") | (cat["type"] == "BOT")]
    fids = cat[cat["type"] == "FID"]
    mon_wins = cat[cat["type"] == "MON"]

    for row in cat:
        ax.annotate(
            "%s" % row["idx"],
            xy=(row["row"] + 120 / 5, row["col"] + 60 / 5),
            color="red",
            fontsize=12,
        )
    ax.scatter(
        gui_stars["row"], gui_stars["col"], facecolors="none", edgecolors="green", s=100
    )

    for acq_star in acq_stars:
        box = plt.Rectangle(
            (
                acq_star["row"] - acq_star["halfw"] / 5,
                acq_star["col"] - acq_star["halfw"] / 5,
            ),
            width=acq_star["halfw"] * 2 / 5,
            height=acq_star["halfw"] * 2 / 5,
            color="blue",
            fill=False,
        )
        ax.add_patch(box)

    for mon_box in mon_wins:
        # starcheck convention was to plot monitor boxes at 2X halfw
        box = plt.Rectangle(
            (
                mon_box["row"] - (mon_box["halfw"] * 2 / 5),
                mon_box["col"] - (mon_box["halfw"] * 2 / 5),
            ),
            width=mon_box["halfw"] * 4 / 5,
            height=mon_box["halfw"] * 4 / 5,
            color="orange",
            fill=False,
        )
        ax.add_patch(box)

    ax.scatter(
        fids["row"],
        fids["col"],
        facecolors="none",
        edgecolors="red",
        linewidth=1,
        marker="o",
        s=175,
    )
    ax.scatter(
        fids["row"], fids["col"], facecolors="red", marker="+", linewidth=1, s=175
    )


def _plot_field_stars(ax, stars, attitude, red_mag_lim=None, bad_stars=None):
    """
    Plot plot field stars in yang and zang on the supplied
    axes object in place.

    Parameters
    ----------
    ax
        matplotlib axes
    stars
        astropy.table compatible set of records of agasc entries of stars
    attitude
        Quaternion-compatible attitude
    red_mag_lim
        faint limit
    bad_stars
        boolean mask of stars to be plotted in red
    """
    stars = Table(stars)
    quat = Quaternion.Quat(attitude)

    if bad_stars is None:
        bad_stars = np.zeros(len(stars), dtype=bool)

    if "yang" not in stars.colnames or "zang" not in stars.colnames:
        # Add star Y angle and Z angle in arcsec to the stars table.
        # radec2yagzag returns degrees.
        yags, zags = radec2yagzag(stars["RA_PMCORR"], stars["DEC_PMCORR"], quat)
        stars["yang"] = yags * 3600
        stars["zang"] = zags * 3600

    # Update table to include row/col values corresponding to yag/zag
    rows, cols = yagzag_to_pixels(stars["yang"], stars["zang"], allow_bad=True)
    stars["row"] = rows
    stars["col"] = cols

    # Initialize array of colors for the stars, default is black.  Use 'object'
    # type to not worry in advance about string length and also for Py2/3 compat.
    colors = np.zeros(len(stars), dtype="object")
    colors[:] = "black"

    colors[bad_stars] = BAD_STAR_COLOR

    if red_mag_lim:
        # Mark stars with the FAINT_STAR_COLOR if they have MAG_ACA
        # that is fainter than red_mag_lim but brighter than red_mag_lim
        # plus a rough mag error.  The rough mag error calculation is
        # based on the SAUSAGE acq stage 1 check, which uses nsigma of
        # 3.0, a mag low limit of 1.5, and a random error of 0.26.
        nsigma = 3.0
        mag_error_low_limit = 1.5
        randerr = 0.26
        caterr = stars["MAG_ACA_ERR"] / 100.0
        error = nsigma * np.sqrt(randerr**2 + caterr**2)
        error = error.clip(mag_error_low_limit)
        # Faint and bad stars will keep their BAD_STAR_COLOR
        # Only use the faint mask on stars that are not bad
        colors[
            (stars["MAG_ACA"] >= red_mag_lim)
            & (stars["MAG_ACA"] < red_mag_lim + error)
            & ~bad_stars
        ] = FAINT_STAR_COLOR
        # Don't plot those for which MAG_ACA is fainter than red_mag_lim + error
        # This overrides any that may be 'bad'
        colors[stars["MAG_ACA"] >= red_mag_lim + error] = "none"

    size = symsize(stars["MAG_ACA"])
    # scatter() does not take an array of alphas, and rgba is
    # awkward for color='none', so plot these in a loop.
    for color, alpha in [
        (FAINT_STAR_COLOR, FAINT_STAR_ALPHA),
        (BAD_STAR_COLOR, BAD_STAR_ALPHA),
        ("black", 1.0),
    ]:
        colormatch = colors == color
        ax.scatter(
            stars[colormatch]["row"],
            stars[colormatch]["col"],
            c=color,
            s=size[colormatch],
            edgecolor="none",
            alpha=alpha,
        )


@custom_plt_rcparams
def plot_stars(
    attitude,
    catalog=None,
    stars=None,
    title=None,
    starcat_time=None,
    red_mag_lim=None,
    quad_bound=True,
    grid=True,
    bad_stars=None,
    plot_keepout=False,
    ax=None,
    duration=0,
):
    """
    Plot a catalog, a star field, or both in a matplotlib figure.
    If supplying a star field, an attitude must also be supplied.

    Parameters
    ----------
    attitude
        A Quaternion compatible attitude for the pointing
    catalog : Records describing catalog.  Must be astropy table compatible.
        Required fields are ['idx', 'type', 'yang', 'zang', 'halfw']
    stars : astropy table compatible set of agasc records of stars
        Required fields are ['RA_PMCORR', 'DEC_PMCORR', 'MAG_ACA', 'MAG_ACA_ERR'].
        If bad_acq_stars will be called (bad_stars is None), additional required fields
        ['CLASS', 'ASPQ1', 'ASPQ2', 'ASPQ3', 'VAR', 'POS_ERR']
        If stars is None, stars will be fetched from the AGASC for the
        supplied attitude.
    title
        string to be used as suptitle for the figure
    starcat_time : DateTime-compatible time.  Used in ACASC fetch for proper
        motion correction.  Not used if stars is not None.
    red_mag_lim
        faint limit for field star plotting.
    quad_bound
        boolean, plot inner quadrant boundaries
    grid
        boolean, plot axis grid
    bad_stars : boolean mask on 'stars' of those that don't meet minimum requirements
        to be selected as acq stars.  If None, bad_stars will be set by a call
        to bad_acq_stars().
    plot_keepout
        plot CCD area to be avoided in star selection (default=False)
    ax
        matplotlib axes object to use (optional)
    duration : duration (starting at ``starcat_time``) for plotting planets
        (secs, default=0)

    Returns
    -------
    matplotlib figure
    """
    if stars is None:
        quat = Quaternion.Quat(attitude)
        stars = agasc.get_agasc_cone(quat.ra, quat.dec, radius=1.5, date=starcat_time)

    if bad_stars is None:
        bad_stars = bad_acq_stars(stars)

    if ax is None:
        fig = plt.figure(figsize=(5.325, 5.325))
        fig.subplots_adjust(top=0.95)

        # Make an empty plot in row, col space
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")
    lim0, lim1 = -580, 590
    plt.xlim(lim0, lim1)  # Matches -2900, 2900 arcsec roughly
    plt.ylim(lim0, lim1)

    # plot the box and set the labels
    b1hw = 512
    box1 = plt.Rectangle((b1hw, -b1hw), -2 * b1hw, 2 * b1hw, fill=False)
    ax.add_patch(box1)
    b2w = 520
    box2 = plt.Rectangle((b2w, -b1hw), -4 + -2 * b2w, 2 * b1hw, fill=False)
    ax.add_patch(box2)

    ax.scatter(
        np.array([-2700, -2700, -2700, -2700, -2700]) / -5,
        np.array([2400, 2100, 1800, 1500, 1200]) / 5,
        c="orange",
        edgecolors="none",
        s=symsize(np.array([10.0, 9.0, 8.0, 7.0, 6.0])),
    )

    # Manually set ticks and grid to specified yag/zag values
    yz_ticks = [-2000, -1000, 0, 1000, 2000]
    zeros = [0, 0, 0, 0, 0]
    r, c = yagzag_to_pixels(yz_ticks, zeros)
    ax.set_xticks(r)
    ax.set_xticklabels(yz_ticks)
    r, c = yagzag_to_pixels(zeros, yz_ticks)
    ax.set_yticks(c)
    ax.set_yticklabels(yz_ticks)
    ax.grid()

    ax.set_xlabel("Yag (arcsec)")
    ax.set_ylabel("Zag (arcsec)")
    [label.set_rotation(90) for label in ax.get_yticklabels()]

    if quad_bound:
        ax.plot([-511, 511], [0, 0], color="magenta", alpha=0.4)
        ax.plot([0, 0], [-511, 511], color="magenta", alpha=0.4)

    if plot_keepout:
        # Plot grey area showing effective keep-out zones for stars.  Back off on
        # outer limits by one pixel to improve rendered PNG slightly.
        row_pad = 15
        col_pad = 8
        box = plt.Rectangle(
            (-511, -511),
            1022,
            1022,
            edgecolor="none",
            facecolor="black",
            alpha=0.2,
            zorder=-1000,
        )
        ax.add_patch(box)
        box = plt.Rectangle(
            (-512 + row_pad, -512 + col_pad),
            1024 - row_pad * 2,
            1024 - col_pad * 2,
            edgecolor="none",
            facecolor="white",
            zorder=-999,
        )
        ax.add_patch(box)

    # Plot stars
    _plot_field_stars(
        ax, stars, attitude=attitude, bad_stars=bad_stars, red_mag_lim=red_mag_lim
    )

    # plot starcheck catalog
    if catalog is not None:
        _plot_catalog_items(ax, catalog)

    # Planets
    _plot_planets(ax, attitude, starcat_time, duration, lim0, lim1)

    if title is not None:
        ax.set_title(title, fontsize="small")

    return fig


def _plot_planets(ax, att, date0, duration, lim0, lim1):
    """

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to use
    att : Quat
        Attitude quaternion
    date0 : CxoTime-compatible
        Date of obs start
    duration : float
        Duration of plot (secs)
    lim0 : float
        Lower limit on x, y axis (row)
    lim1 : float
        Upper limit on x, y axis (col)

    Returns
    -------
    boolean
    True if planets were plotted
    """
    if not isinstance(att, Quat):
        att = Quat(att)

    date0 = CxoTime(date0)
    n_times = int(duration / 1000) + 1
    dates = date0 + np.linspace(0, duration, n_times) * u.s

    planets = ("venus", "mars", "jupiter", "saturn")
    has_planet = False

    for planet in planets:
        # First check if planet is within 2 deg of aimpoint using Earth as the
        # reference point (without fetching Chandra ephemeris). These values are
        # accurate to better than 0.25 deg.
        sep = get_planet_angular_sep(
            planet,
            ra=att.ra,
            dec=att.dec,
            time=date0 + ([0, 0.5, 1] * u.s) * duration,
            observer_position="earth",
        )
        if np.all(sep > 2.0):
            continue

        # Compute ACA row, col for planet each ksec (approx) over the duration.
        # This uses get_planet_chandra which is accurate to 4 arcsec for Venus
        # and < 1 arcsec for Jupiter, Saturn.
        try:
            eci = get_planet_chandra(planet, dates)
            from_earth = False
        except NoEphemerisError:
            # Get the position from Earth using built-in DE432
            eci = get_planet_eci(planet, dates)
            from_earth = True

        ra, dec = eci_to_radec(eci)
        yag, zag = radec_to_yagzag(ra, dec, att)
        row, col = yagzag_to_pixels(yag, zag, allow_bad=True)

        # Only plot planet within the image limits
        ok = (row >= lim0) & (row <= lim1) & (col >= lim0) & (col <= lim1)
        if np.any(ok):
            has_planet = True
            row = row[ok]
            col = col[ok]
            # Plot with green at beginning, red at ending
            ax.plot(row, col, ".", color="m", alpha=0.5)
            ax.plot(row[0], col[0], ".", color="g")
            label = planet.capitalize()
            if from_earth:
                err = GET_PLANET_ECI_ERRORS[planet].to(u.arcsec)
                label += f" (from Earth, errors to {err})"

            ax.plot(row[-1], col[-1], ".", color="r", label=label)

    if has_planet:
        ax.legend(loc="upper left", fontsize="small", facecolor="y", edgecolor="k")


def bad_acq_stars(stars):
    """
    Return mask of 'bad' stars, by evaluating AGASC star parameters.

    Parameters
    ----------
    stars : astropy table-compatible set of agasc records of stars. Required fields
        are ['CLASS', 'ASPQ1', 'ASPQ2', 'ASPQ3', 'VAR', 'POS_ERR']

    Returns
    -------
    boolean mask true for 'bad' stars
    """
    return (
        (stars["CLASS"] != 0)
        | (stars["MAG_ACA_ERR"] > 100)
        | (stars["POS_ERR"] > 3000)
        | (stars["ASPQ1"] > 0)
        | (stars["ASPQ2"] > 0)
        | (stars["ASPQ3"] > 999)
        | (stars["VAR"] > -9999)
    )


@custom_plt_rcparams
def plot_compass(roll):
    """
    Make a compass plot.

    Parameters
    ----------
    roll
        Attitude roll for compass plot.

    Returns
    -------
    matplotlib figure
    """
    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(polar=True)
    ax.annotate(
        "", xy=(0, 0), xytext=(0, 1), arrowprops=dict(arrowstyle="<-", color="k")
    )
    ax.annotate(
        "",
        xy=(0, 0),
        xytext=(np.radians(90), 1),
        arrowprops=dict(arrowstyle="<-", color="k"),
    )
    ax.annotate("N", xy=(0, 0), xytext=(0, 1.2))
    ax.annotate("E", xy=(0, 0), xytext=(np.radians(90), 1.2))
    ax.set_theta_offset(np.radians(90 + roll))
    ax.grid(False)
    ax.set_yticklabels([])
    plt.ylim(0, 1.4)
    plt.tight_layout()
    return fig
