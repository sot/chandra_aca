from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
import agasc
import Quaternion
from Ska.quatutil import radec2yagzag

from .transform import pixels_to_yagzag, yagzag_to_pixels

# rc definitions
frontcolor = 'black'
backcolor = 'white'
plt.rcParams['lines.color'] = frontcolor
plt.rcParams['patch.edgecolor'] = frontcolor
plt.rcParams['text.color'] = frontcolor
plt.rcParams['axes.facecolor'] = backcolor
plt.rcParams['axes.edgecolor'] = frontcolor
plt.rcParams['axes.labelcolor'] = frontcolor
plt.rcParams['xtick.color'] = frontcolor
plt.rcParams['ytick.color'] = frontcolor
plt.rcParams['grid.color'] = frontcolor
plt.rcParams['figure.facecolor'] = backcolor
plt.rcParams['figure.edgecolor'] = backcolor
plt.rcParams['savefig.facecolor'] = backcolor
plt.rcParams['savefig.edgecolor'] = backcolor


BAD_STAR_COLOR = 'tomato'
BAD_STAR_ALPHA = .75
FAINT_STAR_COLOR = 'lightseagreen'
FAINT_STAR_ALPHA = .75


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

    :param ax: matplotlib axes
    :param catalog: data structure containing starcheck-style columns/attributes
                    catalog records.  This can be anything that will work with
                    astropy.table.Table(catalog).  A list of dicts is the convention.
    """
    cat = Table(catalog)
    cat['row'], cat['col'] = yagzag_to_pixels(cat['yang'], cat['zang'], allow_bad=True)
    gui_stars = cat[(cat['type'] == 'GUI') | (cat['type'] == 'BOT')]
    acq_stars = cat[(cat['type'] == 'ACQ') | (cat['type'] == 'BOT')]
    fids = cat[cat['type'] == 'FID']
    mon_wins = cat[cat['type'] == 'MON']

    for row in cat:
        ax.annotate("%s" % row['idx'],
                    xy=(row['row'] + 120 / 5, row['col'] + 60 / 5),
                    color='red',
                    fontsize=12)
    ax.scatter(gui_stars['row'], gui_stars['col'],
               facecolors='none',
               edgecolors='green',
               s=100)

    for acq_star in acq_stars:
        box = plt.Rectangle(
            (acq_star['row'] - acq_star['halfw'] / 5,
             acq_star['col'] - acq_star['halfw'] / 5),
            width=acq_star['halfw'] * 2 / 5,
            height=acq_star['halfw'] * 2 / 5,
            color='blue',
            fill=False)
        ax.add_patch(box)

    for mon_box in mon_wins:
        # starcheck convention was to plot monitor boxes at 2X halfw
        box = plt.Rectangle(
            (mon_box['row'] - (mon_box['halfw'] * 2 / 5),
             mon_box['col'] - (mon_box['halfw'] * 2 / 5)),
            width=mon_box['halfw'] * 4 / 5,
            height=mon_box['halfw'] * 4 / 5,
            color='orange',
            fill=False)
        ax.add_patch(box)

    ax.scatter(fids['row'], fids['col'],
               facecolors='none',
               edgecolors='red',
               linewidth=1,
               marker='o',
               s=175)
    ax.scatter(fids['row'], fids['col'],
               facecolors='red',
               marker='+',
               linewidth=1,
               s=175)


def _plot_field_stars(ax, stars, attitude, red_mag_lim=None, bad_stars=None):
    """
    Plot plot field stars in yang and zang on the supplied
    axes object in place.

    :param ax: matplotlib axes
    :param stars: astropy.table compatible set of records of agasc entries of stars
    :param attitude: Quaternion-compatible attitude
    :param red_mag_lim: faint limit
    :param bad_stars: boolean mask of stars to be plotted in red
    """
    stars = Table(stars)
    quat = Quaternion.Quat(attitude)

    if bad_stars is None:
        bad_stars = np.zeros(len(stars), dtype=bool)

    if 'yang' not in stars.colnames or 'zang' not in stars.colnames:
        # Add star Y angle and Z angle in arcsec to the stars table.
        # radec2yagzag returns degrees.
        yags, zags = radec2yagzag(stars['RA_PMCORR'], stars['DEC_PMCORR'], quat)
        stars['yang'] = yags * 3600
        stars['zang'] = zags * 3600

    # Update table to include row/col values corresponding to yag/zag
    rows, cols = yagzag_to_pixels(stars['yang'], stars['zang'], allow_bad=True)
    stars['row'] = rows
    stars['col'] = cols

    # Initialize array of colors for the stars, default is black.  Use 'object'
    # type to not worry in advance about string length and also for Py2/3 compat.
    colors = np.zeros(len(stars), dtype='object')
    colors[:] = 'black'

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
        caterr = stars['MAG_ACA_ERR'] / 100.
        error = nsigma * np.sqrt(randerr**2 + caterr**2)
        error = error.clip(mag_error_low_limit)
        # Faint and bad stars will keep their BAD_STAR_COLOR
        # Only use the faint mask on stars that are not bad
        colors[(stars['MAG_ACA'] >= red_mag_lim)
               & (stars['MAG_ACA'] < red_mag_lim + error)
               & ~bad_stars] = FAINT_STAR_COLOR
        # Don't plot those for which MAG_ACA is fainter than red_mag_lim + error
        # This overrides any that may be 'bad'
        colors[stars['MAG_ACA'] >= red_mag_lim + error] = 'none'

    size = symsize(stars['MAG_ACA'])
    # scatter() does not take an array of alphas, and rgba is
    # awkward for color='none', so plot these in a loop.
    for color, alpha in [(FAINT_STAR_COLOR, FAINT_STAR_ALPHA),
                         (BAD_STAR_COLOR, BAD_STAR_ALPHA),
                         ('black', 1.0)]:
        colormatch = colors == color
        ax.scatter(stars[colormatch]['row'],
                   stars[colormatch]['col'],
                   c=color, s=size[colormatch], edgecolor='none',
                   alpha=alpha)


def plot_stars(attitude, catalog=None, stars=None, title=None, starcat_time=None,
               red_mag_lim=None, quad_bound=True, grid=True, bad_stars=None,
               plot_keepout=False):
    """
    Plot a catalog, a star field, or both in a matplotlib figure.
    If supplying a star field, an attitude must also be supplied.

    :param attitude: A Quaternion compatible attitude for the pointing
    :param catalog: Records describing catalog.  Must be astropy table compatible.
                    Required fields are ['idx', 'type', 'yang', 'zang', 'halfw']
    :param stars: astropy table compatible set of agasc records of stars
          Required fields are ['RA_PMCORR', 'DEC_PMCORR', 'MAG_ACA', 'MAG_ACA_ERR'].
          If bad_acq_stars will be called (bad_stars is None), additional required fields
          ['CLASS', 'ASPQ1', 'ASPQ2', 'ASPQ3', 'VAR', 'POS_ERR']
          If stars is None, stars will be fetched from the AGASC for the
          supplied attitude.
    :param title: string to be used as suptitle for the figure
    :param starcat_time: DateTime-compatible time.  Used in ACASC fetch for proper
                         motion correction.  Not used if stars is not None.
    :param red_mag_lim: faint limit for field star plotting.
    :param quad_bound: boolean, plot inner quadrant boundaries
    :param grid: boolean, plot axis grid
    :param bad_stars: boolean mask on 'stars' of those that don't meet minimum requirements
                      to be selected as acq stars.  If None, bad_stars will be set by a call
                      to bad_acq_stars().
    :param plot_keepout: plot CCD area to be avoided in star selection (default=False)
    :returns: matplotlib figure
    """
    if stars is None:
        quat = Quaternion.Quat(attitude)
        stars = agasc.get_agasc_cone(quat.ra, quat.dec,
                                     radius=1.5,
                                     date=starcat_time)

    if bad_stars is None:
        bad_stars = bad_acq_stars(stars)

    fig = plt.figure(figsize=(5.325, 5.325))
    fig.subplots_adjust(top=0.95)

    # Make an empty plot in row, col space
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.xlim(-580, 590)  # Matches -2900, 2900 arcsec roughly
    plt.ylim(-580, 590)

    # plot the box and set the labels
    b1hw = 512
    box1 = plt.Rectangle((b1hw, -b1hw), -2 * b1hw, 2 * b1hw,
                         fill=False)
    ax.add_patch(box1)
    b2w = 520
    box2 = plt.Rectangle((b2w, -b1hw), -4 + -2 * b2w, 2 * b1hw,
                         fill=False)
    ax.add_patch(box2)

    ax.scatter(np.array([-2700, -2700, -2700, -2700, -2700]) / -5,
               np.array([2400, 2100, 1800, 1500, 1200]) / 5,
               c='orange', edgecolors='none',
               s=symsize(np.array([10.0, 9.0, 8.0, 7.0, 6.0])))

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
        ax.plot([-511, 511], [0, 0], color='magenta', alpha=0.4)
        ax.plot([0, 0], [-511, 511], color='magenta', alpha=0.4)

    if plot_keepout:
        # Plot grey area showing effective keep-out zones for stars.  Back off on
        # outer limits by one pixel to improve rendered PNG slightly.
        row_pad = 15
        col_pad = 8
        box = plt.Rectangle((-511, -511), 1022, 1022, edgecolor='none',
                            facecolor='black', alpha=0.2, zorder=-1000)
        ax.add_patch(box)
        box = plt.Rectangle((-512 + row_pad, -512 + col_pad),
                            1024 - row_pad * 2, 1024 - col_pad * 2,
                            edgecolor='none', facecolor='white', zorder=-999)
        ax.add_patch(box)

    # Plot stars
    _plot_field_stars(ax, stars, attitude=attitude,
                      bad_stars=bad_stars, red_mag_lim=red_mag_lim)
    # plot starcheck catalog
    if catalog is not None:
        _plot_catalog_items(ax, catalog)
    if title is not None:
        fig.suptitle(title, fontsize='small')

    return fig


def bad_acq_stars(stars):
    """
    Return mask of 'bad' stars, by evaluating AGASC star parameters.

    :param stars: astropy table-compatible set of agasc records of stars. Required fields
          are ['CLASS', 'ASPQ1', 'ASPQ2', 'ASPQ3', 'VAR', 'POS_ERR']
    :returns: boolean mask true for 'bad' stars
    """
    return ((stars['CLASS'] != 0) |
            (stars['MAG_ACA_ERR'] > 100) |
            (stars['POS_ERR'] > 3000) |
            (stars['ASPQ1'] > 0) |
            (stars['ASPQ2'] > 0) |
            (stars['ASPQ3'] > 999) |
            (stars['VAR'] > -9999))


def plot_compass(roll):
    """
    Make a compass plot.

    :param roll: Attitude roll for compass plot.
    :returns: matplotlib figure
    """
    fig = plt.figure(figsize=(3, 3))
    ax = plt.subplot(polar=True)
    ax.annotate("", xy=(0, 0), xytext=(0, 1),
                arrowprops=dict(arrowstyle="<-", color="k"))
    ax.annotate("", xy=(0, 0), xytext=(np.radians(90), 1),
                arrowprops=dict(arrowstyle="<-", color="k"))
    ax.annotate("N", xy=(0, 0), xytext=(0, 1.2))
    ax.annotate("E", xy=(0, 0), xytext=(np.radians(90), 1.2))
    ax.set_theta_offset(np.radians(90 + roll))
    ax.grid(False)
    ax.set_yticklabels([])
    plt.ylim(0, 1.4)
    plt.tight_layout()
    return fig
