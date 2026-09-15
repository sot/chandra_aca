# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings

import agasc
import mica.starcheck
import numpy as np
from astropy.table import Table, vstack
from Chandra.Time import DateTime
from kadi import events
from mica.archive import asp_l1
from Quaternion import Quat
from Ska.engarchive import fetch
from Ska.Numpy import interpolate

from chandra_aca import transform

R2A = 206264.81  # Convert from radians to arcsec


class CentroidResiduals(object):
    """
    Class to calculate star centroid residuals.

    This class is designed to set up and perform the residual calculations on
    any desired combination of source centroids and source attitudes.  For the common use cases,
    centroids, attitudes, and commanded star positions are retrieved automatically from archived
    sources.

    Based on analysis, time offsets are applied to centroid times by default.  See fit notebooks in:

    http://nbviewer.jupyter.org/url/cxc.harvard.edu/mta/ASPECT/ipynb/centroid_time_offsets/OR.ipynb

    and

    http://nbviewer.jupyter.org/url/cxc.harvard.edu/mta/ASPECT/ipynb/centroid_time_offsets/ER.ipynb

    Users should see the class method ``for_slot`` for a convenient way to get centroid
    residuals on an ``obsid`` for an ACA ``slot`` (aka image number).

    Example usage::

     >>> import numpy as np
     >>> from chandra_aca.centroid_resid import CentroidResiduals
     >>> cr = CentroidResiduals.for_slot(obsid=20001, slot=5)
     >>> np.max(np.abs(cr.dyags))
     0.87602233734844503
     >>> np.max(np.abs(cr.dzags))
     1.2035827855862777
     >>> cr.atts[0]
     array([-0.07933254,  0.87065874, -0.47833673,  0.08278696])
     >>> cr.agasc_id
     649201816

    This example calculates the residuals on slot 5 of obsid 20001 using the ground aspect solution
    and ground centroids.  Here is another example that does the same thing without using the
    ``for_slot`` convenience.

    Example usage::

     >>> import numpy as np
     >>> from chandra_aca.centroid_resid import CentroidResiduals
     >>> cr = CentroidResiduals(start='2017:169:18:54:50.138', stop='2017:170:05:13:58.190')
     >>> cr.set_atts('ground')
     >>> cr.set_centroids('ground', slot=5)
     >>> cr.set_star(agasc_id=649201816)
     >>> cr.calc_residuals()
     >>> np.max(np.abs(cr.dyags))
     0.87602233734844503


    :param start: start time of interval for residuals (DateTime compatible)
    :param stop: stop time of interval for residuals (DateTime compatible)
    :param set_no_track_to_nan: set centroids to NaN where the OBC was not tracking
        instead of dropping those samples (default=False, 'obc' centroid source only)

    By default, OBC centroid samples where the OBC had no star in the slot are dropped
    from the time series entirely, leaving an unmarked gap. Because the residuals on
    either side of the gap are small, anything that draws or interpolates a line across
    it (a plot, or ``np.interp`` onto a uniform grid) produces smooth small values that
    look just like good tracking. With ``set_no_track_to_nan=True`` every sample is
    kept and those with no star are set to NaN instead, so the dropout stays visible in
    ``yags`` / ``zags`` and propagates into ``dyags`` / ``dzags``::

     >>> cr = CentroidResiduals.for_slot(obsid=15175, slot=6, att_source='obc',
     ...                                 centroid_source='obc',
     ...                                 set_no_track_to_nan=True)
     >>> len(cr.dyags), int(np.count_nonzero(np.isnan(cr.dyags)))
     (59363, 482)
     >>> float(np.nanmax(np.abs(cr.dyags)))
     3.9186003402858205

    The 482 samples with no star are kept as NaN here instead of being dropped, and the
    residuals that remain are unchanged. Note that NaN-aware functions
    (``np.nanmedian``, ``np.nanstd`` and friends) are then needed for statistics, since
    ``np.max`` and friends return NaN.

    """

    centroid_source = None
    att_source = None
    ra = None
    dec = None
    centroid_dt = None
    obsid = None

    def __init__(self, start, stop, set_no_track_to_nan=False):
        self.start = start
        self.stop = stop
        self.set_no_track_to_nan = set_no_track_to_nan

    def set_centroids(self, source, slot, alg=8, apply_dt=True):
        """
        Assign centroids.

        Assign centroids from ``source`` and ``slot`` to the objects centroid attributes
        (yag, zag, yag_times, zag_times)

        For the supported sources (ground, obc) the centroids are fetched from the mica L1
        archive or telemetry.

        yag_times and zag_times are always identical.

        yag, zag, yag_times an zag_times can also be set directly without use of this method.

        Parameters
        ----------
        source
            'ground' | 'obc'
        slot
            ACA slot
        alg
            for ground processing, use centroids from this algorithm.
        apply_dt
            apply centroid time offsets via 'set_offsets'
        """
        self.centroid_source = source
        self.centroid_dt = None
        start = self.start
        stop = self.stop
        if self.set_no_track_to_nan and source != "obc":
            raise ValueError(
                "set_no_track_to_nan is only supported for centroid source 'obc', "
                f"got {source!r}. Ground L1 centroids have no per-sample OBC track "
                "status, and the ACACENT rows for a slot are already absent (not "
                "flagged) where there was no star."
            )
        # Get centroids from Ska eng archive or mica L1 archive
        if source == "ground":
            acen_files = sorted(
                asp_l1.get_files(start=start, stop=stop, content=["ACACENT"])
            )
            acen = vstack(
                [Table.read(f) for f in sorted(acen_files)], metadata_conflicts="silent"
            )
            ok = (
                (acen["slot"] == slot)
                & (acen["alg"] == alg)
                & (acen["status"] == 0)
                & (acen["time"] >= DateTime(start).secs)
                & (acen["time"] <= DateTime(stop).secs)
            )
            yags = np.array(acen[ok]["ang_y"] * 3600)
            zags = np.array(acen[ok]["ang_z"] * 3600)
            yag_times = np.array(acen[ok]["time"])
            zag_times = np.array(acen[ok]["time"])
        elif source == "obc":
            msids = [
                f"AOACYAN{slot}",
                f"AOACZAN{slot}",
                f"AOACFCT{slot}",
            ]
            telem = fetch.MSIDset(msids, start, stop)
            # Same content type for all MSIDs so they have exactly the same times and we
            # can interpolate to these times (which are guaranteed to be at 1.025 sec
            # spacing).
            times = telem[f"AOACYAN{slot}"].times
            telem.interpolate(times=times, bad_union=True, filter_bad=False)
            yan = telem[f"AOACYAN{slot}"]
            zan = telem[f"AOACZAN{slot}"]
            fct = telem[f"AOACFCT{slot}"]
            # AOACFCT comes from the same telemetry sampling as AOACYAN / AOACZAN,
            # so the track status lines up with the centroids sample for sample.
            # Find intervals of no track, where any bona fide bad data in telemetry
            # (from extremely rare data loss in dumps) is also considered no track.
            no_track = (fct.vals != "TRAK") | yan.bads | zan.bads
            # Convert from 32-bit native vals to 64-bit for convenience later.
            yags = yan.vals.astype(np.float64)
            zags = zan.vals.astype(np.float64)

            if self.set_no_track_to_nan:
                yags[no_track] = np.nan
                zags[no_track] = np.nan
                yag_times = times
                zag_times = times
            else:
                track = ~no_track
                yags = yags[track]
                zags = zags[track]
                yag_times = times[track]
                zag_times = times[track]
        else:
            raise ValueError("centroid_source must be 'obc' or 'ground'")

        self.yags = yags
        self.yag_times = yag_times
        self.zags = zags
        self.zag_times = zag_times
        if apply_dt is True:
            self.set_offsets()
        else:
            self.centroid_dt = 0

    def set_atts(self, source):
        """Get attitude solution quaternions from ``source``.

        One could also just set atts and att_times attributes directly.
        """
        self.att_source = source
        tstart = DateTime(self.start).secs
        tstop = DateTime(self.stop).secs
        # Get attitudes and times
        if source == "obc":
            telem = fetch.Msidset(["aoattqt*"], tstart, tstop)
            atts = np.vstack(
                [telem[f"aoattqt{idx}"].vals for idx in [1, 2, 3, 4]]
            ).transpose()
            att_times = telem["aoattqt1"].times
            # Fetch COBSQID at beginning and end of interval, check they match, and define obsid
            if self.obsid is None:
                obsid_start = fetch.Msid("COBSRQID", tstart, tstart + 60)
                obsid_stop = fetch.Msid("COBSRQID", tstop - 60, tstop)
                if len(obsid_start.vals) == 0 or len(obsid_stop.vals) == 0:
                    fetch_source = fetch.data_source.sources()[0]
                    raise ValueError(
                        "Error getting COBSRQID telem for "
                        f"tstart:{tstart} tstop:{tstop} "
                        f"from fetch_source:{fetch_source}"
                    )
                self.obsid = obsid_start.vals[-1]
        elif source == "ground":
            atts, att_times, asol_recs = asp_l1.get_atts(start=tstart, stop=tstop)
            obsids = np.unique(np.array([int(rec["OBS_ID"]) for rec in asol_recs]))
            if len(obsids) > 1:
                raise ValueError(
                    "Time range covers more than one obsid; Not supported."
                )
            self.obsid = obsids[0]
        else:
            raise ValueError("att_source must be 'obc' or 'ground'")
        ok = (att_times >= tstart) & (att_times < tstop)
        self.atts = atts[ok, :]  # (N, 4) numpy array
        self.att_times = att_times[ok]

    def set_atts_from_solfiles(self, asol_files, acal_files, aqual_files, filter=True):
        atts, att_times, asol_recs = asp_l1.get_atts_from_files(
            asol_files, acal_files, aqual_files, filter=filter
        )
        obsids = np.unique(np.array([int(rec["OBS_ID"]) for rec in asol_recs]))
        if len(obsids) > 1:
            raise ValueError("Time range covers more than one obsid; Not supported.")
        self.atts = atts
        self.att_times = att_times

    def set_star(self, agasc_id=None, slot=None):
        """
        Set self.ra and dec from either agasc_id *or* slot.

        This assumes use of star in default agasc miniagasc (no 1.4 or 1.5 or very faint stars)
        Lookup by "slot" relies on database of starcheck catalogs.

        This also sets self.agasc_id.
        """
        if agasc_id is not None:
            star = agasc.get_star(agasc_id, date=self.start)
        elif slot is not None:
            sc = mica.starcheck.get_starcheck_catalog_at_date(self.start)
            stars = sc["cat"][
                (sc["cat"]["slot"] == slot)
                & ((sc["cat"]["type"] == "GUI") | (sc["cat"]["type"] == "BOT"))
            ]
            if not len(stars):
                raise ValueError(
                    f"No GUI or BOT in slot {slot} at time "
                    f"{DateTime(self.start).date} in dwell"
                )
            star = agasc.get_star(stars[0]["id"], date=self.start)
        else:
            raise ValueError("Need to supply agasc_id or slot to look up star")

        # Could also add logic to infer star from loose position and magnitude
        self.agasc_id = star["AGASC_ID"]
        self.ra = star["RA_PMCORR"]
        self.dec = star["DEC_PMCORR"]

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
        Apply time offsets to centroids.

        Apply time offsets to centroids based on type and source of centroid, obsid
        (suggesting 8x8 or 6x6 data), telemetry source ('maude' or 'cxc') and aspect solution
        source. These time offsets were fit.  See fit notebooks at:

        http://nbviewer.jupyter.org/url/cxc.harvard.edu/mta/ASPECT/ipynb/centroid_time_offsets/OR.ipynb

        and

        http://nbviewer.jupyter.org/url/cxc.harvard.edu/mta/ASPECT/ipynb/centroid_time_offsets/ER.ipynb

        """
        # If already applied, do nothing
        if self.centroid_dt is not None:
            return
        if self.att_source is None or self.centroid_source is None:
            return
        # Get and check reasonable-ness of fetch data source
        if len(fetch.data_source.sources()) > 1:
            warnings.warn(
                "Can't set offsets based on fetch data "
                "source if multiple data sources set"
            )
            return
        fetch_source = fetch.data_source.sources()[0]
        if fetch_source not in ("cxc", "maude"):
            warnings.warn(
                "Only maude and cxc fetch data sources are supported for offsets. "
                "Not applying offsets."
            )
            return
        obstype = "or" if self.obsid < 38000 else "er"
        if fetch_source == "maude" and obstype == "er":
            warnings.warn(
                "Centroid time offsets not well fit for 'maude' telem source on ERs."
                " Use caution."
            )

        # Offsets calculated using OR and ER notebooks in SKA/analysis/centroid_and_sol_time_offsets
        offsets = {
            # (centroid_source, att_source, fetch_source, obstype):  median offset in time
            ("obc", "obc", "cxc", "or"): -2.45523126997,
            ("obc", "ground", "cxc", "or"): -2.46900481785,
            ("ground", "obc", "cxc", "or"): 0.0366092437236,
            ("ground", "ground", "cxc", "or"): 0.0553306628318,
            ("obc", "obc", "maude", "or"): -2.96746879688,
            ("obc", "ground", "maude", "or"): -2.94076404877,
            ("ground", "obc", "maude", "or"): 0.010515586472,
            ("obc", "obc", "cxc", "er"): -2.53954270068,
            ("obc", "ground", "cxc", "er"): -2.49080106675,
            ("ground", "obc", "cxc", "er"): -0.0322463030744,
            ("ground", "ground", "cxc", "er"): 0.0355677462107,
            ("obc", "obc", "maude", "er"): -2.90454699136,
            ("obc", "ground", "maude", "er"): -3.00151559564,
            ("ground", "obc", "maude", "er"): 0.116096583881,
        }

        self.centroid_dt = offsets[
            (self.centroid_source, self.att_source, fetch_source, obstype)
        ]

        self.yag_times = self.yag_times + self.centroid_dt
        self.zag_times = self.zag_times + self.centroid_dt

    def calc_residuals(self):
        """
        Calculate star residuals.

        Calculate residuals based on attitude and ra/dec of star.  Note that the sampling and times
        of yags may be different from zags so these should be done independently.

        Residuals are available in self.dyags and self.dzags.
        Predicted values from attitude and star position in self.pred_yags and self.pred_zags

        """
        # If time offsets weren't applied because centroids were initialized before atts, try again
        if self.centroid_dt is None:
            self.set_offsets()
        # If still not set, warn
        if self.centroid_dt is None:
            warnings.warn(
                "Residuals calculated on centroids without time offsets applied"
            )
        if len(self.att_times) < 2:
            raise ValueError(
                "Cannot attempt to calculate residuals with fewer than 2 attitude"
                " samples"
            )
        eci = transform.radec_to_eci(self.ra, self.dec)
        # Transform the 3x3 to get the axes to align to have the dot product make sense
        d_aca = np.dot(Quat(q=self.atts).transform.transpose(0, 2, 1), eci)
        p_yags = np.arctan2(d_aca[:, 1], d_aca[:, 0]) * R2A
        p_zags = np.arctan2(d_aca[:, 2], d_aca[:, 0]) * R2A
        self.pred_yags = interpolate(
            p_yags, self.att_times, self.yag_times, sorted=True
        )
        self.pred_zags = interpolate(
            p_zags, self.att_times, self.zag_times, sorted=True
        )
        self.dyags = self.yags - self.pred_yags
        self.dzags = self.zags - self.pred_zags

    @classmethod
    def for_slot(
        cls,
        *,
        obsid=None,
        start=None,
        stop=None,
        slot=None,
        att_source="ground",
        centroid_source="ground",
        set_no_track_to_nan=False,
    ):
        if obsid is not None:
            if start is not None or stop is not None:
                raise ValueError("cannot specify both obsid and start / stop")
            ds = events.dwells.filter(obsid=obsid)
            start = ds[0].start
            stop = ds[len(ds) - 1].stop
        if start is None or stop is None:
            raise ValueError("must specify obsid or start / stop")
        cr = cls(start, stop, set_no_track_to_nan=set_no_track_to_nan)
        if obsid is not None:
            cr.obsid = obsid
        cr.set_atts(att_source)
        cr.set_centroids(centroid_source, slot)
        cr.set_star(slot=slot)
        cr.calc_residuals()  # instead of get_residuals
        return cr
