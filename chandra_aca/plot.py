import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
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
    gui_stars = cat[(cat['type'] == 'GUI') | (cat['type'] == 'BOT')]
    acq_stars = cat[(cat['type'] == 'ACQ') | (cat['type'] == 'BOT')]
    fids = cat[cat['type'] == 'FID']
    mon_wins = cat[cat['type'] == 'MON']
    for row in cat:
        ax.annotate("%s" % row['idx'],
                    xy=(row['yang'] - 120, row['zang'] + 60),
                    color='red',
                    fontsize=12)
    g = ax.scatter(gui_stars['yang'], gui_stars['zang'],
                   facecolors='none',
                   edgecolors='green',
                   s=100)
    g.set_zorder(5)
    for acq_star in acq_stars:
        box = plt.Rectangle(
            (acq_star['yang'] - acq_star['halfw'],
             acq_star['zang'] - acq_star['halfw']),
            width=acq_star['halfw'] * 2,
            height=acq_star['halfw'] * 2,
            color='blue',
            fill=False)
        box.set_zorder(5)
        ax.add_patch(box)
    for mon_box in mon_wins:
        # starcheck convention was to plot monitor boxes at 2X halfw
        box = plt.Rectangle(
            (mon_box['yang'] - (mon_box['halfw'] * 2),
             mon_box['zang'] - (mon_box['halfw'] * 2)),
            width=mon_box['halfw'] * 4,
            height=mon_box['halfw'] * 4,
            color='orange',
            fill=False)
        box.set_zorder(5)
        ax.add_patch(box)
    fo = ax.scatter(fids['yang'], fids['zang'],
                    facecolors='none',
                    edgecolors='red',
                    linewidth=1,
                    marker='o',
                    s=175)
    fo.set_zorder(5)
    fp = ax.scatter(fids['yang'], fids['zang'],
                    facecolors='red',
                    marker='+',
                    linewidth=1,
                    s=175)
    fp.set_zorder(5)


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

    # Initialize array of colors for the stars, default is black
    colors = np.zeros(len(stars), dtype='S20')
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
        s = ax.scatter(stars[colormatch]['yang'],
                   stars[colormatch]['zang'],
                   c=color, s=size[colormatch], edgecolor='none',
                   alpha=alpha)
        s.set_zorder(4)


def plot_ccd_edges(ax, row_pad=15, col_pad=8):

    row_min = -512.5
    row_max = 511.5
    col_min = -512.5
    col_max = 511.5

    row_range = np.arange(row_min, row_max + 1)
    col_range = np.arange(col_min, col_max + 1)

    # Add a polygon that represents the edge of the CCD in Y/Z
    # These are plotted as line segments instead of a polygon to make
    # it easier to plot the margins and the readout register

    # Left "line" with double line for readout register
    y1, z1 = pixels_to_yagzag(
        np.repeat(row_min, len(col_range)),
        col_range)
    ax.plot(y1, z1, color='black')
    ax.plot(y1 + 40, z1, color='black')

    # Top "line"
    y2, z2 = pixels_to_yagzag(
        row_range,
        np.repeat(col_max, len(row_range)))
    ax.plot(y2, z2, color='black')

    # Right "line" (flip the ranges on the next two to
    # make the coordinates work when using the points to describe
    # the vertices of the polygon)
    y3, z3 = pixels_to_yagzag(
        np.repeat(row_max, len(col_range)),
        col_range[::-1])
    ax.plot(y3, z3, color='black')
    ax.plot(y3 - 40, z3, color='black')

    # Bottom "line"
    y4, z4 = pixels_to_yagzag(
        row_range[::-1],
        np.repeat(col_min, len(row_range)))
    ax.plot(y4, z4, color='black')
    # end of lines for CCD boundary polygon

    # Grey polygon that starts at the edges
    out_y = np.hstack([y1, y2, y3, y4])
    out_z = np.hstack([z1, z2, z3, z4])
    out_v = np.vstack([out_y, out_z]).transpose()
    grey = plt.Polygon(out_v, fill=True, color='grey')
    ax.add_patch(grey)

    # Make a white polygon to fill the center so we're left with a grey
    # border.  Use "reduced" mins and maxes to describe this
    r_row_max = row_max - row_pad
    r_row_min = row_min + row_pad
    r_col_max = col_max - col_pad
    r_col_min = col_min + col_pad
    r_row_range = np.arange(r_row_min, r_row_max + 1)
    r_col_range = np.arange(r_col_min, r_col_max + 1)
    ry1, rz1 = pixels_to_yagzag(
        np.repeat(r_row_min, len(r_col_range)),
        r_col_range)
    ry2, rz2 = pixels_to_yagzag(
        r_row_range,
        np.repeat(r_col_max, len(r_row_range)))
    ry3, rz3 = pixels_to_yagzag(
        np.repeat(r_row_max, len(r_col_range)),
        r_col_range[::-1])
    ry4, rz4 = pixels_to_yagzag(
        r_row_range[::-1],
        np.repeat(r_col_min, len(r_row_range)))
    r_out_y = np.hstack([ry1, ry2, ry3, ry4])
    r_out_z = np.hstack([rz1, rz2, rz3, rz4])
    r_out_v = np.vstack([r_out_y, r_out_z]).transpose()
    white = plt.Polygon(r_out_v, fill=True, color='white')
    ax.add_patch(white)


def plot_stars(attitude, catalog=None, stars=None, title=None, starcat_time=None,
               red_mag_lim=None, quad_bound=True, grid=True, bad_stars=None):
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

    # plot the CCD edges and the fixed boundaries
    plot_ccd_edges(ax)

    # plot a little mag size mini-legend
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

    # plot stars
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
