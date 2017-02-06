import numpy as np
from ..dark_model import dark_temp_scale

def test_dark_temp_scale():
    scale = dark_temp_scale(-10., -14)
    assert np.allclose(scale, 0.70)

    scale = dark_temp_scale(-10., -14, scale_4c=2.0)
    assert scale == 0.5  # Should be an exact match
