import numpy as np
from ..aca_image import AcaPsfLibrary

ap = AcaPsfLibrary()


def test_basic():
    assert len(ap.psfs) == 12 ** 2
    assert np.isclose(ap.drc, 0.1, rtol=0, atol=1e-8)  # Current PSF library has 0.1 pixel gridding
    psf = ap.get_psf_image(0, 0)
    assert psf.__class__.__name__ == 'ACAImage'


def test_centroids():
    """
    Test that the actual centroid of interpolated PSF is close enough to
    desired centroid.  Currently there is a problem near the edge of the
    sub-pixel grid where centroids can be off by as much as 0.12 arcsec.
    "The edge" means outside of 0.16 to 0.84.
    """
    r0s = [0, 150]
    c0s = [-300, 0]
    nt = 30
    outr = np.zeros((nt, nt))
    outc = np.zeros((nt, nt))
    for r0 in r0s:
        for c0 in c0s:
            for ir, dr in enumerate(np.linspace(0.0, 1.0, nt)):
                for ic, dc in enumerate(np.linspace(0.0, 1.0, nt)):
                    r = r0 + dr
                    c = c0 + dc
                    psf = ap.get_psf_image(r, c)

                    # Get the centroid in ACA coordinates
                    rc, cc, norm = psf.aca.centroid_fm()

                    # PSF centroids are off from expectation near the edges.
                    # Doing an imshow(outr) or imshow(outc) demonstrates this.
                    if abs(0.5 - dr) > 0.33 or abs(0.5 - dc) > 0.33:
                        atol = 0.032  # 0.16 arcsec (needs work)
                    else:
                        atol = 0.008  # 0.04 arcsec

                    outr[ir, ic] = r - rc
                    outc[ir, ic] = c - cc
                    assert np.isclose(r, rc, rtol=0, atol=atol)
                    assert np.isclose(c, cc, rtol=0, atol=atol)

    return outr, outc
