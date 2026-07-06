"""Companion tests for chandra_aca.swats.

Two kinds of checks:

* Pure checks on the reader / dict builder.
* Cross-checks that decommutate the SWATS ASP_TLM dump and compare against the
  commanded values in the SWATS ACA_CMDS file.

Both bench files live in the test data directory and are committed to the repo.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from chandra_aca import maude_decom
from chandra_aca.swats import build_raw_aca_packets, read_aca_packets

DATA = Path(__file__).resolve().parent / "data"
ASP_TLM = DATA / "swats_asp_tlm.txt"
ACA_CMDS = DATA / "swats_aca_cmds.txt"

INTEG_UNIT = 0.016  # ACA integration-time LSB [s]


def _commanded_star_locations(text):
    """Return {slot: (row, col)} from the first test case in ACA_CMDS.txt."""
    star = {}
    for m in re.finditer(r"Star #(\d+) Location\s*=\s*([-\d.]+)\s+([-\d.]+)", text):
        star.setdefault(int(m.group(1)), (float(m.group(2)), float(m.group(3))))
    return star


@pytest.fixture(scope="module")
def images():
    """Decommutate the whole aspect telemetry dump file once for the cross-check tests."""

    raw = read_aca_packets(ASP_TLM)
    start, stop = raw["TIME"][0], raw["TIME"][-1] + 1.025
    return maude_decom.get_aca_images(start, stop, raw_aca_packets=raw)


def test_build_raw_aca_packets_increments():
    # TIME steps by the 1.025 s ACA readout period; VCDUCTR by 4; MNF/MJF derived.
    raw = build_raw_aca_packets([bytes(224)] * 5)

    # Default construction starts every counter/time at zero.
    assert raw["TIME"][0] == 0.0
    assert raw["VCDUCTR"][0] == 0
    assert raw["MJF"][0] == 0
    assert raw["MNF"][0] == 0

    assert np.allclose(np.diff(raw["TIME"]), 1.025)
    assert np.all(np.diff(raw["VCDUCTR"]) == 4)
    assert np.array_equal(raw["MNF"], raw["VCDUCTR"] % 128)
    assert np.array_equal(raw["MJF"], raw["VCDUCTR"] // 128)

    # The VCDU counter is a 24-bit field: VCDUCTR and the derived MJF roll back to 0 at
    # 2**24. Start 4 short of the rollover so the 2nd packet lands exactly on it.
    two24 = 1 << 24  # 16777216
    wrap = build_raw_aca_packets([bytes(224)] * 3, vcdu0=two24 - 4)
    assert wrap["VCDUCTR"].tolist() == [two24 - 4, 0, 4]
    assert wrap["MJF"][0] == 131071  # 2**17 - 1: major counter at its max ...
    assert wrap["MJF"][1] == 0 and wrap["MNF"][1] == 0  # ... then wraps to zero


def test_integ_matches_commanded_int_time(images):
    # This is a sanity check that verifies that the commanded integration time in ACA_CMDS.txt
    # matches the integration time in the decommutated images.
    commanded = float(
        re.search(r"Int Time\s*=\s*([\d.]+)", ACA_CMDS.read_text()).group(1)
    )
    quantized = round(commanded / INTEG_UNIT) * INTEG_UNIT  # 1.696 s

    integ = np.asarray(images["INTEG"], dtype=float)
    assert np.any(np.isclose(integ, quantized, atol=1e-3))


def test_star_locations_match_tracking_windows(images):
    # The commanded Star #N Location (row, col) must show up as an 8x8 tracking
    # window center (IMGROW0_8X8 + 4, IMGCOL0_8X8 + 4). This is an acquisition run
    # (search boxes), so not every slot converges -- require at least 6 of 8 within 2 px.
    star = _commanded_star_locations(ACA_CMDS.read_text())
    num = np.asarray(images["IMGNUM"])
    cen_row = np.asarray(images["IMGROW0_8X8"], dtype=float) + 4.0
    cen_col = np.asarray(images["IMGCOL0_8X8"], dtype=float) + 4.0

    residuals = {}
    for slot, (row, col) in star.items():
        sel = num == slot
        residuals[slot] = np.hypot(cen_row[sel] - row, cen_col[sel] - col).min()

    matched = [slot for slot, r in residuals.items() if r < 2.0]
    assert len(matched) >= 6, (
        f"only {len(matched)}/8 slots matched: residuals={residuals}"
    )
