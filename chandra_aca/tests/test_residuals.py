# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os

import mica.common
import numpy as np
import pytest

from ..centroid_resid import CentroidResiduals

HAS_L1_ARCHIVE = os.path.exists(os.path.join(mica.common.MICA_ARCHIVE, "asp1"))
HAS_STARCHECK_ARCHIVE = os.path.exists(
    os.path.join(mica.common.MICA_ARCHIVE, "starcheck")
)
try:
    from Ska.engarchive import fetch

    fetch.Msidset(["aoattqt*"], "2018:001:00:00:00", "2018:001:00:01:00")
    HAS_QUAT_TELEM = True
except Exception:
    HAS_QUAT_TELEM = False


@pytest.mark.skipif(
    "not HAS_L1_ARCHIVE", reason="No ground solutions without an aspl1 mica archive"
)
@pytest.mark.skipif(
    "not HAS_STARCHECK_ARCHIVE", reason="No for_slot without a starcheck mica archive"
)
def test_multi_ai():
    # obsid 15175 has two aspect intervals
    cr = CentroidResiduals.for_slot(obsid=15175, slot=4)
    assert np.all(np.abs(cr.dyags) < 3)
    assert np.all(np.abs(cr.dzags) < 6)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - 0.0553306628318) < 1e-8


@pytest.mark.skipif(
    "not HAS_L1_ARCHIVE", reason="No ground solutions without an aspl1 mica archive"
)
@pytest.mark.skipif(
    "not HAS_STARCHECK_ARCHIVE", reason="No for_slot without a starcheck mica archive"
)
def test_obc_centroids():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=5, centroid_source="obc")
    assert np.all(np.abs(cr.dyags) < 7)
    assert np.all(np.abs(cr.dzags) < 4)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.46900481785) < 1e-8


@pytest.mark.skipif(
    "not HAS_STARCHECK_ARCHIVE", reason="No for_slot without a starcheck mica archive"
)
@pytest.mark.skipif(
    "not HAS_QUAT_TELEM", reason="No AOATTQT* telemetry in Ska.engarchive"
)
def test_obc():
    cr = CentroidResiduals.for_slot(
        obsid=15175, slot=6, att_source="obc", centroid_source="obc"
    )
    assert np.all(np.abs(cr.dyags) < 4.5)
    assert np.all(np.abs(cr.dzags) < 5.5)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.45523126997) < 1e-8


@pytest.mark.skipif(
    "not HAS_STARCHECK_ARCHIVE", reason="No for_slot without a starcheck mica archive"
)
@pytest.mark.skipif(
    "not HAS_QUAT_TELEM", reason="No AOATTQT* telemetry in Ska.engarchive"
)
def test_er():
    cr = CentroidResiduals.for_slot(
        obsid=57635, slot=6, att_source="obc", centroid_source="obc"
    )
    assert np.all(np.abs(cr.dyags) < 4)
    assert np.all(np.abs(cr.dzags) < 3)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.53954270068) < 1e-8


def test_or_manual():
    # This test should run with either maude or cxc as the fetch source and without
    # access to mica archive of L1 products or mica starcheck archive
    with fetch.data_source("cxc" if HAS_QUAT_TELEM else "maude"):
        # Smaller time interval for MAUDE
        stop = "2017:170:05:13:58.190" if HAS_QUAT_TELEM else "2017:169:19:30:00.00"
        cr = CentroidResiduals(start="2017:169:18:54:50.138", stop=stop)
        cr.set_atts("obc")
        cr.set_centroids("obc", slot=5)
        cr.set_star(agasc_id=649201816)
        cr.calc_residuals()
