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

@pytest.mark.skipif('not HAS_L1_ARCHIVE', reason='No ground solutions without an aspl1 mica archive')
@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_obc_centroids():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=5, centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 6)
    assert np.all(np.abs(cr.dzags) < 4)

@pytest.mark.skipif('not HAS_STARCHECK_ARCHIVE', reason='No for_slot without a starcheck mica archive')
def test_obc():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=6, att_source='obc', centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 4.5)
    assert np.all(np.abs(cr.dzags) < 5.5)
