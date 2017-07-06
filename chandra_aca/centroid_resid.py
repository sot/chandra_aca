import re
import numpy as np
import warnings

from astropy.table import Table, vstack
from kadi import events
from Ska.Numpy import interpolate
from Quaternion import Quat
import Ska.quatutil
from mica.archive import asp_l1
import mica.starcheck
from Ska.engarchive import fetch
from Chandra.Time import DateTime
import agasc


R2A = 206264.81  # Convert from radians to arcsec


def quat_vtransform(qs):
    """
    Transform a Nx4 matrix of quaternions into a Nx3x3 transform matrix
    :returns: Nx3x3 transform matrix
    :rtype: numpy array
    """
    x, y, z, w = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
    xx2 = x * x * 2.
    yy2 = y * y * 2.
    zz2 = z * z * 2.
    xy2 = x * y * 2.
    wz2 = w * z * 2.
    zx2 = z * x * 2.
    wy2 = w * y * 2.
    yz2 = y * z * 2.
    wx2 = w * x * 2.

    t = np.empty((len(qs), 3, 3), float)
    t[:, 0, 0] = 1. - yy2 - zz2
    t[:, 0, 1] = xy2 - wz2
    t[:, 0, 2] = zx2 + wy2
    t[:, 1, 0] = xy2 + wz2
    t[:, 1, 1] = 1. - xx2 - zz2
    t[:, 1, 2] = yz2 - wx2
    t[:, 2, 0] = zx2 - wy2
    t[:, 2, 1] = yz2 + wx2
    t[:, 2, 2] = 1. - xx2 - yy2

    return t


def quat_vmult(q1, q2):
    # just assume q1 and q2 have same length
    mult = np.zeros((len(q1), 4))
    mult[:,0] =  q1[:,3] * q2[:,0] - q1[:,2] * q2[:,1] + q1[:,1] * q2[:,2] + q1[:,0] * q2[:,3]
    mult[:,1] =  q1[:,2] * q2[:,0] + q1[:,3] * q2[:,1] - q1[:,0] * q2[:,2] + q1[:,1] * q2[:,3]
    mult[:,2] = -q1[:,1] * q2[:,0] + q1[:,0] * q2[:,1] + q1[:,3] * q2[:,2] + q1[:,2] * q2[:,3]
    mult[:,3] = -q1[:,0] * q2[:,0] - q1[:,1] * q2[:,1] - q1[:,2] * q2[:,2] + q1[:,3] * q2[:,3]
    return mult


class CentroidResiduals(object):
    centroid_source = None
    att_source = None
    ra = None
    dec = None
    centroid_dt = None
    _dt_applied = None

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop


    def set_centroids(self, source, slot, alg=8, apply_dt=True):
        """
        Get centroids from ``source``.

        One could also just set yags, yag_times, zags, zag_times
        attributes directly.

        :param source: 'ground' | 'obc'
        :param slot: ACA slot
        """
        self.centroid_source = source
        start = self.start
        stop = self.stop
        # Get centroids from Ska eng archive or mica L1 archive
        # Might want to add filtering for track status here too
        # Also need to include something for time offsets
        # And warn to console if we hit multiple obsids
        if source == 'ground':
            acen_files = sorted(asp_l1.get_files(start=start, stop=stop, content=['ACACENT']))
            acen = vstack([Table.read(f) for f in sorted(acen_files)], metadata_conflicts='silent')
            ok = ((acen['slot'] == slot) & (acen['alg'] == alg) & (acen['status'] == 0)
                  & (acen['time'] >= DateTime(start).secs) & (acen['time'] <= DateTime(stop).secs))
            yags = np.array(acen[ok]['ang_y'] * 3600)
            zags = np.array(acen[ok]['ang_z'] * 3600)
            yag_times = np.array(acen[ok]['time'])
            zag_times = np.array(acen[ok]['time'])
        elif source == 'obc':
            telem = fetch.Msidset(['AOACYAN{}'.format(slot), 'AOACZAN{}'.format(slot)], start, stop)
            # Filter centroids for reasonble-ness
            yok = telem['AOACYAN{}'.format(slot)].vals > -3276
            zok = telem['AOACZAN{}'.format(slot)].vals > -3276
            yags = telem['AOACYAN{}'.format(slot)].vals[yok]
            yag_times = telem['AOACYAN{}'.format(slot)].times[yok]
            zags = telem['AOACZAN{}'.format(slot)].vals[zok]
            zag_times = telem['AOACZAN{}'.format(slot)].times[zok]
        else:
            raise ValueError("centroid_source must be 'obc' or 'ground'")
        self.yags = yags
        self.yag_times = yag_times
        self.zags = zags
        self.zag_times = zag_times
        if apply_dt:
            self.set_offsets()


    def set_atts(self, source):
        """Get attitude solution quaternions from ``source``.

        One could also just set atts and att_times attributes directly.
        """
        self.att_source = source
        tstart = DateTime(self.start).secs
        tstop = DateTime(self.stop).secs
        # Get attitudes and times
        if source == 'obc':
            telem = fetch.Msidset(['aoattqt*'], tstart, tstop)
            atts = np.vstack([telem['aoattqt{}'.format(idx)].vals
                              for idx in [1, 2, 3, 4]]).transpose()
            att_times = telem['aoattqt1'].times
            # Fetch COBSQID at beginning and end of interval, check they match, and defined obsid
            obsid_start = fetch.Msid('COBSRQID', tstart, tstart + 60)
            obsid_stop = fetch.Msid('COBSRQID', tstop - 60, tstop)
            if obsid_start.vals[-1] != obsid_stop.vals[0]:
                raise NotImplementedError("Time range covers more than one obsid; Not supported at this time")
            self.obsid = obsid_start.vals[-1]
        elif source == 'ground':
            atts, att_times, asol_recs = asp_l1.get_atts(start=tstart, stop=tstop)
            obsids = np.unique(np.array([int(rec['OBS_ID']) for rec in asol_recs]))
            if len(obsids) > 1:
                raise NotImplementedError("Time range covers more than one obsid; Not supported at this time")
            self.obsid = obsids[0]
        else:
            raise ValueError("att_source must be 'obc' or 'ground'")
        ok = (att_times >= tstart) & (att_times < tstop)
        self.atts = atts[ok, :]  # (N, 4) numpy array
        self.att_times = att_times[ok]


    def set_atts_from_solfiles(self, asol_files, acal_files, aqual_files):
        atts, att_times, asol_recs = asp_l1.get_atts_from_files(asol_files, acal_files, aqual_files)
        obsids = np.unique(np.array([int(rec['OBS_ID']) for rec in asol_recs]))
        if len(obsids) > 1:
            raise NotImplementedError("Time range covers more than one obsid; Not supported at this time")
        self.atts = atts
        self.att_times = att_times

    def set_star(self, agasc_id=None, obsid=None, slot=None, date=None):
        """
        Set self.ra and dec from either agasc_id *or* obsid + slot.

        This assumes use of star in default agasc miniagasc (no 1.4 or 1.5 or very faint stars)

        Along the way set self.agasc_id.  One can also just set
        self.ra and dec directly.
        """
        if obsid is not None and slot is not None:
            if date is None:
                raise ValueError("Need date to look up star catalog if obsid/slot provided")
            sc = mica.starcheck.get_starcheck_catalog_at_date(date)
            star = sc['cat'][(sc['cat']['slot'] == slot)
                             & ((sc['cat']['type'] == 'GUI') | (sc['cat']['type'] == 'BOT'))]
            if not len(star):
                raise ValueError(
                    "No GUI or BOT in slot {} for obsid {} in dwell".format(slot, obsid))
            agasc_id = star['id'][0]
        # Could also add logic to infer star from loose position and magnitude
        star = agasc.get_star(agasc_id, date=date)
        self.agasc_id = agasc_id
        self.ra = star['RA_PMCORR']
        self.dec = star['DEC_PMCORR']

    @property
    def yags(self):
        return self._yags

    @yags.setter
    def yags(self, vals):
        if isinstance(vals, fetch.MSID):
            self._yags = np.array(vals.vals)
            self._yag_times = vals.times
        else:
            self._yags = np.array(vals)

    @property
    def yag_times(self):
        return self._yag_times

    @yag_times.setter
    def yag_times(self, vals):
        self._yag_times = np.array(vals)

    @property
    def zags(self):
        return self._zags

    @zags.setter
    def zags(self, vals):
        if isinstance(vals, fetch.MSID):
            self._zags = np.array(vals.vals)
            self._zag_times = vals.times
        else:
            self._zags = np.array(vals)

    @property
    def zag_times(self):
        return self._zag_times

    @zag_times.setter
    def zag_times(self, vals):
        self._zag_times = np.array(vals)

    def set_offsets(self):
        """
        Apply time offsets to centroids based on type and source of centroid, obsid
        (suggesting 8x8 or 6x6 data), and aspect solution source.  These time offsets were
        fit.  The offsets determined against OBC aspect solutions are quite variable and
        this code will warn when applying offsets in the case when residuals are to be
        calculated using an OBC solution.
        """

        if self._dt_applied is True:
            raise ValueError("Attempted to apply centroid dt more than once")
        is_or = self.obsid < 38000
        # Offsets calculated using fit scripts in SKA/analysis/centroid_and_sol_time_offsets
        # Also see plots in PR https://github.com/sot/chandra_aca/pull/25
        if self.att_source == 'obc':
            warnings.warn("Applying centroid time offset for {} centroids and obc att source.".format(
                self.centroid_source))
        # For ground centroids we don't need the fetch source
        if self.centroid_source == "ground":
            if is_or:
                if self.att_source == 'ground':
                    self.centroid_dt = 0.125
                if self.att_source == 'obc':
                    self.centroid_dt = 0.665
            else:
                if self.att_source == 'ground':
                    self.centroid_dt = 0.054
                if self.att_source == 'obc':
                    self.centroid_dt = 0.087
        else:
            if len(fetch.data_source.sources()) > 1:
                raise ValueError("Can't set offsets based on fetch data source if multiple data sources set")
            fetch_source = fetch.data_source.sources()[0]
            if fetch_source != 'cxc' and fetch_source != 'maude':
                raise ValueError("Only maude and cxc fetch data sources are supported for offsets")
            if is_or:
                if fetch_source == 'cxc':
                    if self.att_source == 'ground':
                        self.centroid_dt = -2.4
                    if self.att_source == 'obc':
                        self.centroid_dt = -1.8
                if fetch_source == 'maude':
                    if self.att_source == 'ground':
                        self.centroid_dt = -2.8
                    if self.att_source == 'obc':
                        self.centroid_dt = -2.34
            else:
                if fetch_source == 'cxc':
                    if self.att_source == 'ground':
                        self.centroid_dt = -2.44
                    if self.att_source == 'obc':
                        self.centroid_dt = -2.43
                elif fetch_source == 'maude':
                    if self.att_source == 'ground':
                        self.centroid_dt = -2.45
                    if self.att_source == 'obc':
                        self.centroid_dt = -2.45

        self.yag_times = self.yag_times + self.centroid_dt
        self.zag_times = self.zag_times + self.centroid_dt
        self._dt_applied = True

    def calc_residuals(self):
        """
        Calculate residuals based on attitude and ra/dec of star.  Note that the sampling and times
        of yags may be different from zags so these should be done independently.

        Residuals are available in self.dyags and self.dzags.
        Predicted values from attitude and star position in self.pred_yags and self.pred_zags

        """
        eci = Ska.quatutil.radec2eci(self.ra, self.dec)
        # Transform the 3x3 to get the axes to align to have the dot product make sense
        d_aca = np.dot(quat_vtransform(self.atts).transpose(0, 2, 1), eci)
        p_yags = np.arctan2(d_aca[:, 1], d_aca[:, 0]) * R2A
        p_zags = np.arctan2(d_aca[:, 2], d_aca[:, 0]) * R2A
        self.pred_yags = interpolate(p_yags, self.att_times, self.yag_times, sorted=True)
        self.pred_zags = interpolate(p_zags, self.att_times, self.zag_times, sorted=True)
        self.dyags = self.yags - self.pred_yags
        self.dzags = self.zags - self.pred_zags


    @classmethod
    def for_slot(cls, obsid=None, start=None, stop=None,
                 slot=None, att_source='ground', centroid_source='ground'):
        if obsid is not None:
            if start is not None or stop is not None:
                raise ValueError('cannot specify both obsid and start / stop')
            ds = events.dwells.filter(obsid=obsid)
            start = ds[0].start
            stop = ds[len(ds) - 1].stop
        if start is None or stop is None:
            raise ValueError('must specify obsid or start / stop')
        cr = cls(start, stop)
        if obsid is not None:
            cr.obsid = obsid
        cr.set_atts(att_source)
        cr.set_centroids(centroid_source, slot)
        cr.set_star(obsid=obsid, slot=slot, date=start)
        cr.calc_residuals()  # instead of get_residuals
        return cr
