import os
import numpy as np
import pytest
import mica.common

from ..centroid_resid import CentroidResiduals

HAS_L1_ARCHIVE = os.path.exists(os.path.join(mica.common.MICA_ARCHIVE, 'asp1'))
HAS_STARCHECK_ARCHIVE = os.path.exists(os.path.join(mica.common.MICA_ARCHIVE, 'starcheck'))

@pytest.mark.skipif('not HAS_L1_ARCHIVE', reason='No ground solutions without an aspl1 mica archive')
@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_multi_ai():
    # obsid 15175 has two aspect intervals
    cr = CentroidResiduals.for_slot(obsid=15175, slot=4)
    assert np.all(np.abs(cr.dyags) < 3)
    assert np.all(np.abs(cr.dzags) < 6)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - 0.0553306628318) < 1e-8

@pytest.mark.skipif('not HAS_L1_ARCHIVE', reason='No ground solutions without an aspl1 mica archive')
@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_obc_centroids():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=5, centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 6)
    assert np.all(np.abs(cr.dzags) < 4)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.46900481785) < 1e-8

@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_obc():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=6, att_source='obc', centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 4.5)
    assert np.all(np.abs(cr.dzags) < 5.5)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.45523126997) < 1e-8

@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_er():
    cr = CentroidResiduals.for_slot(obsid=57635, slot=6, att_source='obc', centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 4)
    assert np.all(np.abs(cr.dzags) < 3)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.53954270068) < 1e-8

@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_er():
    cr = CentroidResiduals.for_slot(obsid=57635, slot=6, att_source='obc', centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 4)
    assert np.all(np.abs(cr.dzags) < 3)
    # check that the right offset was applied from the table in set_offsets
    assert abs(cr.centroid_dt - -2.53954270068) < 1e-8

def test_or_manual():
    # This test should run with either maude or cxc as the fetch source and without
    # access to mica archive of L1 products or mica starcheck archive
    cr = CentroidResiduals(start='2017:169:18:54:50.138', stop='2017:170:05:13:58.190')
    cr.set_atts('obc')
    cr.set_centroids('obc', slot=5)
    cr.set_star(agasc_id=649201816)
    cr.calc_residuals()
