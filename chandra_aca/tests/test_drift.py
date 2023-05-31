# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.table as tbl
import numpy as np
import pytest
import ska_helpers.paths

from chandra_aca import drift


def test_get_aca_offsets_legacy():
    """
    Legacy test that ACA offsets are reasonable, and regression test particular values
    corresponding to cycle 17 zero-offset aimpoints used below for chip_x, chip_y inputs.

    The output reference values here have been validated as being "reasonable" for the
    given inputs.
    """
    offsets = drift.get_aca_offsets(
        "ACIS-I", 3, 930.2, 1009.6, "2016:180:12:00:00", -15.0
    )
    assert np.allclose(offsets, (11.45, 2.34), atol=0.1, rtol=0)

    offsets = drift.get_aca_offsets(
        "ACIS-S", 7, 200.7, 476.9, "2016:180:12:00:00", -15.0
    )
    assert np.allclose(offsets, (12.98, 3.52), atol=0.1, rtol=0)

    offsets = drift.get_aca_offsets("HRC-I", 0, 7591, 7936, "2016:180:12:00:00", -15.0)
    assert np.allclose(offsets, (14.35, 0.45), atol=0.1, rtol=0)

    offsets = drift.get_aca_offsets("HRC-S", 2, 2041, 9062, "2016:180:12:00:00", -15.0)
    assert np.allclose(offsets, (16.89, 3.10), atol=0.1, rtol=0)


# Generate test cases for get_aca_offsets(). This file was generated by the notebook
# fit_aimpoint_drift-2022-11.ipynb in the aimpoint_mon repo.
filename = Path(__file__).parent / "data" / "aimpoint_regression_data.ecsv"
dat = tbl.Table.read(filename)
kwargs_list = []
for row in dat:
    kwargs = dict(
        detector=row["detector"],
        chip_id=row["chip_id"],
        chipx=row["chipx"],
        chipy=row["chipy"],
        time=row["mean_date"],
        t_ccd=row["mean_t_ccd"],
        aca_offset_y=row["aca_offset_y"],
        aca_offset_z=row["aca_offset_z"],
    )
    kwargs_list.append(kwargs)


@pytest.mark.parametrize("kwargs", kwargs_list)
@pytest.mark.parametrize("env_override", [None, str(Path(__file__).parent / "data")])
def test_get_aca_offsets(kwargs, env_override, monkeypatch):
    """Regression test that ACA offsets match the original flight values from 2022-11
    analysis to expected precision."""
    monkeypatch.setenv("CHANDRA_MODELS_DEFAULT_VERSION", "3.48")
    kwargs = kwargs.copy()
    aca_offset_y = kwargs.pop("aca_offset_y")
    aca_offset_z = kwargs.pop("aca_offset_z")
    offsets = drift.get_aca_offsets(**kwargs)
    dy = offsets[0] - aca_offset_y
    dz = offsets[1] - aca_offset_z
    # Stored precision of t_ccd is 0.01, so error from actual could be 0.005 C *
    # 3.9 arcsec/C. Also up to 0.005 arcsec error in the stored aca_offset_y/z values.
    assert abs(dy) < 0.03
    assert abs(dz) < 0.02
