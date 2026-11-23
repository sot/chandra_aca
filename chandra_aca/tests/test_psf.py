import os

import numpy as np
import pytest

from chandra_aca.aca_image import ACAImage, AcaPsfLibrary

APS = {version: AcaPsfLibrary(version=version) for version in AcaPsfLibrary.VERSIONS}


@pytest.fixture(params=AcaPsfLibrary.VERSIONS)
def ap(request):
    return APS[request.param]


def test_basic(ap):
    assert len(ap.psfs) == 12**2
    assert np.isclose(
        ap.drc, 0.1, rtol=0, atol=1e-8
    )  # Current PSF library has 0.1 pixel gridding
    psf = ap.get_psf_image(0, 0)
    assert isinstance(psf, ACAImage)

    psf, row0, col0 = ap.get_psf_image(
        104.1, 203.9, aca_image=False, pix_zero_loc="edge"
    )
    assert not isinstance(psf, ACAImage)
    assert isinstance(psf, np.ndarray)
    assert row0 == 100
    assert col0 == 200


def test_centroids(ap):
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
                        # 2011: 0.16 arcsec, 2026: 0.21 arcsec (needs work)
                        atol = {"2011": 0.032, "2026": 0.042}[ap.version]
                    else:
                        atol = 0.008  # 0.04 arcsec

                    outr[ir, ic] = r - rc
                    outc[ir, ic] = c - cc
                    assert np.isclose(r, rc, rtol=0, atol=atol)
                    assert np.isclose(c, cc, rtol=0, atol=atol)


def test_psf_at_index_location(ap):
    """Test that requesting a PSF image at exactly an existing location in the
    library gives the expected result"""
    # Generate row/col position at exactly a bin center for ap.psfs[2, 3]
    dat = ap.dat
    ii = 2
    jj = 3
    ok = (dat["row_bin_idx"] == ii) & (dat["col_bin_idx"] == jj)
    i22 = np.flatnonzero(ok)[0]
    row = dat[i22]

    # Row/col of center of bin
    rc = (row["row_bin_left_edge"] + row["row_bin_right_edge"]) / 2.0
    cc = (row["col_bin_left_edge"] + row["col_bin_right_edge"]) / 2.0

    psf_direct = ap.psfs[ii, jj]

    psf_interp = ap.get_psf_image(rc, cc, pix_zero_loc="edge")
    assert np.allclose(psf_direct, psf_interp, rtol=0, atol=1e-5)

    # Test with interpolation=nearest
    psf_interp = ap.get_psf_image(rc, cc, pix_zero_loc="edge", interpolation="nearest")
    assert np.allclose(psf_direct, psf_interp, rtol=0, atol=1e-5)

    # Test with interpolation=nearest, slightly offset but still within grid subpixel
    psf_interp = ap.get_psf_image(
        rc + ap.drc * 0.45,
        cc - ap.drc * 0.45,
        pix_zero_loc="edge",
        interpolation="nearest",
    )
    assert np.allclose(psf_direct, psf_interp, rtol=0, atol=1e-5)


def test_default_version():
    ap = AcaPsfLibrary()
    assert ap.version == "2026"
    assert os.path.basename(ap.filename) == "aca_psf_lib_2026.dat"
    assert np.all(ap.dat == APS["2026"].dat)


@pytest.mark.parametrize("version", ["2026", "2011"])
def test_version(version):
    ap = AcaPsfLibrary(version=version)
    assert ap.version == version
    assert os.path.basename(ap.filename) == f"aca_psf_lib_{version}.dat"


def test_versions_differ():
    psf_2026 = APS["2026"].get_psf_image(10.3, 20.7, norm=1000, aca_image=False)[0]
    psf_2011 = APS["2011"].get_psf_image(10.3, 20.7, norm=1000, aca_image=False)[0]
    assert not np.allclose(psf_2026, psf_2011)


def test_filename():
    filename = APS["2011"].filename
    ap = AcaPsfLibrary(filename=filename)
    assert ap.version is None
    assert ap.filename == filename
    assert np.all(ap.dat == APS["2011"].dat)


@pytest.mark.parametrize("version", ["2020", 2026, 2011])
def test_bad_version(version):
    with pytest.raises(ValueError, match="version must be one of"):
        AcaPsfLibrary(version=version)


def test_filename_and_version():
    with pytest.raises(ValueError, match="cannot specify both filename and version"):
        AcaPsfLibrary(filename=APS["2011"].filename, version="2011")
