import numpy as np
import matplotlib.pyplot as plt

from astropy.table import hstack, Table
import agasc
import Quaternion
from Ska.quatutil import radec2yagzag

from .transform import pixels_to_yagzag

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
    gui = cat[(cat['type'] == 'GUI') | (cat['type'] == 'BOT')]
    acq = cat[(cat['type'] == 'ACQ') | (cat['type'] == 'BOT')]
    fid = cat[cat['type'] == 'FID']
    mon = cat[cat['type'] == 'MON']
    for row in cat:
        ax.annotate("%s" % row['idx'],
                    xy=(row['yang'] - 120, row['zang'] + 60),
                    color='red',
                    fontsize=12)
    ax.scatter(gui['yang'], gui['zang'],
               facecolors='none',
               edgecolors='green',
               s=100)
    for acq_star in acq:
        acq_box = plt.Rectangle(
            (acq_star['yang'] - acq_star['halfw'],
             acq_star['zang'] - acq_star['halfw']),
            width=acq_star['halfw'] * 2,
            height=acq_star['halfw'] * 2,
            color='blue',
            fill=False)
        ax.add_patch(acq_box)
    for mon_box in mon:
        # starcheck convention was to plot monitor boxes at 2X halfw
        box = plt.Rectangle(
            (mon_box['yang'] - (mon_box['halfw'] * 2),
             mon_box['zang'] - (mon_box['halfw'] * 2)),
            width=mon_box['halfw'] * 4,
            height=mon_box['halfw'] * 4,
            color='orange',
            fill=False)
        ax.add_patch(box)
    ax.scatter(fid['yang'], fid['zang'],
               facecolors='none',
               edgecolors='red',
               linewidth=.5,
               marker='o',
               s=175)
    ax.scatter(fid['yang'], fid['zang'],
               facecolors='none',
               edgecolors='red',
               marker='+',
               linewidth=.5,
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

    # Add star Y angle and Z angle in arcsec to the stars table
    yagzags = (radec2yagzag(star['RA_PMCORR'], star['DEC_PMCORR'], quat)
               for star in stars)
    yagzags = Table(rows=[(y * 3600, z * 3600) for y, z in yagzags], names=['yang', 'zang'])
    stars = hstack([stars, yagzags])

    # Initialize array of colors for the stars, default is black
    colors = np.zeros(len(stars), dtype='S20')
    colors[:] = 'black'

    colors[bad_stars] = BAD_STAR_COLOR

    if red_mag_lim:
        # Mark stars with the FAINT_STAR_COLOR if they have MAG_ACA
        # that is fainter than red_mag_lim but MAG_ACA - a rough mag error
        # is brighter than that limit.  The rough mag error calculation is
        # based on the SAUSAGE acq stage 1 check, which uses nsigma of
        # 3.0, a mag low limit of 1.5, and a random error of 0.26.
        nsigma = 3.0
        mag_error_low_limit = 1.5
        randerr = 0.26
        caterr = stars['MAG_ACA_ERR'] / 100.
        error = nsigma * np.sqrt(randerr**2 + caterr**2)
        error = np.clip(error, a_min=mag_error_low_limit, a_max=None)
        faint = ((stars['MAG_ACA'] >= red_mag_lim)
                 & ((stars['MAG_ACA'] - error) < red_mag_lim))
        # Faint and bad stars will keep their BAD_STAR_COLOR
        # Only use the faint mask on stars that are not bad
        faint = faint & ~bad_stars
        colors[faint] = FAINT_STAR_COLOR
        # Don't plot those for which MAG_ACA - error is fainter than red_mag_lim
        # This overrides any that may be 'bad'
        too_dim_to_plot = ((stars['MAG_ACA'] >= red_mag_lim)
                            & ((stars['MAG_ACA'] - error) >= red_mag_lim))
        colors[too_dim_to_plot] = 'none'

    size = symsize(stars['MAG_ACA'])
    # scatter() does not take an array of alphas, and rgba is
    # awkward for color='none', so plot these in a loop.
    for color, alpha in [(FAINT_STAR_COLOR, FAINT_STAR_ALPHA),
                         (BAD_STAR_COLOR, BAD_STAR_ALPHA),
                         ('none', 1.0),
                         ('black', 1.0)]:
        colormatch = colors == color
        ax.scatter(stars[colormatch]['yang'],
                   stars[colormatch]['zang'],
                   c=color, s=size[colormatch], edgecolor='none',
                   alpha=alpha)


def star_plot(catalog=None, attitude=None, stars=None, title=None, starcat_time=None,
              red_mag_lim=None, quad_bound=True, grid=True, bad_stars=None):
    """
    Plot a catalog, a star field, or both in a matplotlib figure.
    If supplying a star field, an attitude must also be supplied.

    :param catalog: Records describing catalog.  Must be astropy table compatible.
    :param attitude: A Quaternion compatible attitude for the pointing
    :param stars: astropy table compatible set of agasc records of stars
    :param title: string to be used as suptitle for the figure
    :param red_mag_lim: faint limit for field star plotting.
    :param quad_bound: boolean, plot inner quadrant boundaries
    :param grid: boolean, plot axis grid
    :param bad_stars: boolean mask on 'stars' of those that don't meet minimum requirements
                      to be selected as acq stars
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
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(top=0.95)
    ax.set_aspect('equal')

    # plot the box and set the labels
    plt.xlim(2900, -2900)
    plt.ylim(-2900, 2900)
    b1hw = 2560
    box1 = plt.Rectangle((b1hw, -b1hw), -2 * b1hw, 2 * b1hw,
                         fill=False)
    ax.add_patch(box1)
    b2w = 2600
    box2 = plt.Rectangle((b2w, -b1hw), -4 + -2 * b2w, 2 * b1hw,
                         fill=False)
    ax.add_patch(box2)

    ax.scatter([-2700, -2700, -2700, -2700, -2700],
               [2400, 2100, 1800, 1500, 1200],
               c='orange', edgecolors='none',
               s=symsize(np.array([10.0, 9.0, 8.0, 7.0, 6.0])))

    [l.set_rotation(90) for l in ax.get_yticklabels()]
    ax.grid(grid)
    ax.set_ylabel("Zag (arcsec)")
    ax.set_xlabel("Yag (arcsec)")

    if quad_bound:
        pix_range = np.linspace(-510, 510, 50)
        minus_half_pix = -0.5 * np.ones_like(pix_range)
        # plot the row = -0.5 line
        yag, zag = pixels_to_yagzag(minus_half_pix, pix_range)
        ax.plot(yag, zag, color='magenta', alpha=.4)
        # plot the col = -0.5 line
        yag, zag = pixels_to_yagzag(pix_range, minus_half_pix)
        ax.plot(yag, zag, color='magenta', alpha=.4)

    # plot field if present
    if stars is not None:
        if attitude is None:
            raise ValueError("Must supply attitude to plot field stars")
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
    :param stars: astropy table-compatible set of agasc records of stars
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
