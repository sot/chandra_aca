import numpy as np
from ..centroid_resid import CentroidResiduals

def test_multi_ai():
    # obsid 15175 has two aspect intervals
    cr = CentroidResiduals.for_slot(obsid=15175, slot=4)
    assert np.all(np.abs(cr.dyags) < 3)
    assert np.all(np.abs(cr.dzags) < 6)

def test_obc_centroids():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=5, centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 6)
    assert np.all(np.abs(cr.dzags) < 4)

def test_obc():
    cr = CentroidResiduals.for_slot(obsid=15175, slot=6, att_source='obc', centroid_source='obc')
    assert np.all(np.abs(cr.dyags) < 4.5)
    assert np.all(np.abs(cr.dzags) < 5.5)
