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
from chandra_aca.swats import (
    _fix_pixel_bit_order,
    build_raw_aca_packets,
    read_aca_packets,
)

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


def test_fix_pixel_bit_order():
    # Build a packet whose slot-0 pixel field encodes known 10-bit values in the
    # PEA-dump order ([2 LSBs][8 MSBs] per pixel). After _fix_pixel_bit_order, the
    # standard maude_decom decoder must return the original values.
    values = np.arange(16) * 61 + 42  # 16 distinct 10-bit values, up to 957
    dump_groups = np.zeros((16, 10), dtype=np.uint8)
    for j, v in enumerate(values):
        lsb2 = [(v >> 1) & 1, v & 1]
        msb8 = [(v >> k) & 1 for k in range(9, 1, -1)]
        dump_groups[j] = lsb2 + msb8
    packet = bytearray(224)
    packet[8 + 7 : 8 + 27] = np.packbits(dump_groups.reshape(-1)).tobytes()

    fixed = _fix_pixel_bit_order(bytes(packet))
    slots = maude_decom.unpack_aca_telemetry(fixed)
    assert np.array_equal(slots[0]["pixels"], values)

    # The re-packing must not touch anything outside the pixel fields.
    assert fixed[:15] == bytes(packet[:15])


def test_images_contain_stars():
    # Each complete 8x8 tracking image must look like a star: most of the
    # background-subtracted flux concentrated in the 3x3 box around the peak, and the
    # peak within 2 px of the window center. This catches pixel bit-order regressions
    # (a mis-packed pixel field decodes to spatially uncorrelated noise).
    # The pixel bit-order fix is opt-in; enable it while reading the dump so the
    # decoded images contain actual star data (see _fix_pixel_bit_order).
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CHANDRA_ACA_FIX_PIXEL_BIT_ORDER", "1")
        raw = read_aca_packets(ASP_TLM)
    start, stop = raw["TIME"][0], raw["TIME"][-1] + 1.025
    images = maude_decom.get_aca_images(start, stop, raw_aca_packets=raw)

    sel = images[images["IMGTYPE"] == 4]
    img = np.asarray([r["IMG"] for r in sel], dtype=float)
    complete = ~np.isnan(img).any(axis=(1, 2))
    assert complete.sum() > 100
    img = img[complete]

    bkg = np.median(img, axis=(1, 2), keepdims=True)
    flux = np.clip(img - bkg, 0, None)
    peak = flux.reshape(len(flux), -1).argmax(axis=1)
    prow, pcol = np.unravel_index(peak, (8, 8))

    # peak concentration: sum of the 3x3 around the peak over the total
    rows = np.clip(prow[:, None] + np.array([-1, 0, 1]), 0, 7)
    cols = np.clip(pcol[:, None] + np.array([-1, 0, 1]), 0, 7)
    idx = np.arange(len(flux))[:, None, None]
    core = flux[idx, rows[:, :, None], cols[:, None, :]].sum(axis=(1, 2))
    concentration = core / flux.sum(axis=(1, 2))
    assert np.mean(concentration) > 0.5

    # Tracked stars sit at the window center (rows/cols 3-4). This is an acquisition
    # run, so allow for stretches where a slot has not converged yet.
    centered = (np.abs(prow - 3.5) <= 2) & (np.abs(pcol - 3.5) <= 2)
    assert np.mean(centered) > 0.85


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
