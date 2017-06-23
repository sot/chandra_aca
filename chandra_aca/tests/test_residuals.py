import numpy as np
from ..centroid_resid import get_obs_slot_residuals

def test_multi_ai():
    # obsid 15175 has two aspect intervals
    dyags, yt, dzags, zt = get_obs_slot_residuals(15175, 4)
    assert np.all(np.abs(dyags) < 3)
    assert np.all(np.abs(dzags) < 6)

def test_obc_centroids():
    dyags, yt, dzags, zt = get_obs_slot_residuals(15175, 5, centroid_source='obc')
    assert np.all(np.abs(dyags) < 6)
    assert np.all(np.abs(dzags) < 3)

def test_obc():
    dyags, yt, dzags, zt = get_obs_slot_residuals(15175, 6, att_source='obc', centroid_source='obc')
    assert np.all(np.abs(dyags) < 4.5)
    assert np.all(np.abs(dzags) < 5.5)
