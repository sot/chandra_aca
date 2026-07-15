"""Companion tests for chandra_aca.swats.

Two kinds of checks:

* Pure checks on the readers / dict builder / packet decoders.
* Cross-checks that decommutate the SWATS ASP_TLM and OBC_TLM dumps and compare
  against the commanded values in the SWATS ACA_CMDS file.

All bench files live in the test data directory and are committed to the repo.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from chandra_aca import maude_decom
from chandra_aca.swats import (
    build_raw_aca_packets,
    read_aca_packets,
    read_obc_packets,
)

DATA = Path(__file__).resolve().parent / "data"
ASP_TLM = DATA / "swats_asp_tlm.txt"
ACA_CMDS = DATA / "swats_aca_cmds.txt"
OBC_TLM = DATA / "swats_obc_tlm.txt"

INTEG_UNIT = 0.016  # ACA integration-time LSB [s]


def _commanded_star_locations(text):
    """Return {slot: (row, col)} from the first test case in ACA_CMDS.txt."""
    star = {}
    for m in re.finditer(r"Star #(\d+) Location\s*=\s*([-\d.]+)\s+([-\d.]+)", text):
        star.setdefault(int(m.group(1)), (float(m.group(2)), float(m.group(3))))
    return star


def _commanded_yang_zang(text):
    """Return the 8 commanded (Yang, Zang) [arcsec] from the first test case in ACA_CMDS.txt."""
    yang = [float(v) for v in re.search(r"Yang\s*=([^\n#]+)", text).group(1).split()]
    zang = [float(v) for v in re.search(r"Zang\s*=([^\n#]+)", text).group(1).split()]
    return list(zip(yang, zang, strict=True))


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


def test_images_contain_stars():
    # Each complete 8x8 tracking image must look like a star: most of the
    # background-subtracted flux concentrated in the 3x3 box around the peak, and the
    # peak within 2 px of the window center. This catches pixel bit-order regressions
    # (a mis-packed pixel field decodes to spatially uncorrelated noise). SWATS dumps
    # once had the 10-bit pixel groups rotated by two bits; this guards against that
    # coming back.
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


def _bits(value, n):
    """Return the n-bit big-endian bit list of (the two's complement of) value."""
    return [(value >> (n - 1 - k)) & 1 for k in range(n)]


def test_unpack_obc_telemetry():
    # Build a 60-byte OBC packet bit by bit from the format description and check the
    # decoded engineering values. Slot 0 is tracking (IMGFUNC=1) with known angles;
    # the other slots are searching (IMGFUNC=2) with the not-tracking sentinels
    # (YAG/ZAG=$20000, MAG=$FF).
    bits = []
    bits += _bits(106, 16)  # INTEG: 106 x 16 ms = 1.696 s
    bits += _bits(0b10000001, 8)  # global status: HIGH_BGD and SYNTAX_ERROR
    bits += _bits(61, 6) + _bits(2, 2)  # COMMPROG=61, COMMPROG_REPEAT=2

    # slot 0: fid=0, num=0, func=1 (tracking), sat=1, def=0, quad=1, common=0,
    # multi=0, ion=1, ZAG=-117 arcsec, YAG=+1352 arcsec, MAG=10.0
    bits += [0] + _bits(0, 3) + _bits(1, 2) + [1, 0]
    bits += [1, 0, 0, 1]
    bits += _bits(-4680, 18)  # -117 / 0.025
    bits += _bits(54080, 18)  # 1352 / 0.025
    bits += _bits(192, 8)  # (10.0 + 2.0) / 0.0625

    for num in range(1, 8):  # slots 1-7: fiducial, searching, sentinel values
        bits += [1] + _bits(num, 3) + _bits(2, 2) + [0, 0]
        bits += [0, 0, 0, 0]
        bits += _bits(0x20000, 18) + _bits(0x20000, 18) + _bits(0xFF, 8)

    packet = np.packbits(bits).tobytes()
    assert len(packet) == 60

    result = maude_decom.unpack_obc_telemetry(packet)

    assert np.isclose(result["INTEG"], 1.696)
    assert result["GLBSTAT"] == 0b10000001
    for flag in ["HIGH_BGD", "SYNTAX_ERROR"]:
        assert result[flag] is True
    for flag in [
        "RAM_FAIL",
        "ROM_FAIL",
        "POWER_FAIL",
        "CAL_FAIL",
        "COMM_CHECKSUM_FAIL",
        "RESET",
    ]:
        assert result[flag] is False
    assert result["COMMPROG"] == 61
    assert result["COMMPROG_REPEAT"] == 2

    assert len(result["image_data"]) == 8
    slot0 = result["image_data"][0]
    assert slot0["IMGFID"] is False
    assert slot0["IMGNUM"] == 0
    assert slot0["IMGFUNC"] == 1
    assert slot0["SAT_PIXEL"] is True
    assert slot0["DEF_PIXEL"] is False
    assert slot0["QUAD_BOUND"] is True
    assert slot0["COMMON_COL"] is False
    assert slot0["MULTI_STAR"] is False
    assert slot0["ION_RAD"] is True
    assert np.isclose(slot0["ZAG"], -117.0)
    assert np.isclose(slot0["YAG"], 1352.0)
    assert np.isclose(slot0["MAG"], 10.0)

    for num, slot in enumerate(result["image_data"][1:], start=1):
        assert slot["IMGFID"] is True
        assert slot["IMGNUM"] == num
        assert slot["IMGFUNC"] == 2
        # the $20000 sentinel decodes to the most negative 18-bit angle
        assert np.isclose(slot["ZAG"], -131072 * 0.025)
        assert np.isclose(slot["YAG"], -131072 * 0.025)
        assert np.isclose(slot["MAG"], -2.0 + 255 * 0.0625)


def test_unpack_obc_telemetry_bad_length():
    with pytest.raises(ValueError, match="60-byte packet"):
        maude_decom.unpack_obc_telemetry(bytes(59))


def test_obc_matches_commanded():
    # Decode the whole OBC telemetry dump and cross-check against the ACA_CMDS file:
    # the commanded integration time must show up in INTEG, and each tracked slot's
    # (YAG, ZAG) centroid must approach the commanded (Yang, Zang) star position.
    # The commanded positions are for the target attitude while the OBC tracks at the
    # estimated flight attitude, so allow a generous tolerance. This is an acquisition
    # run, so not every slot converges -- require at least 6 of 8.
    telemetry = read_obc_packets(OBC_TLM)
    assert len(telemetry) > 100

    commanded = float(
        re.search(r"Int Time\s*=\s*([\d.]+)", ACA_CMDS.read_text()).group(1)
    )
    quantized = round(commanded / INTEG_UNIT) * INTEG_UNIT  # 1.696 s
    integ = np.array([t["INTEG"] for t in telemetry])
    assert np.any(np.isclose(integ, quantized, atol=1e-3))

    yang_zang = _commanded_yang_zang(ACA_CMDS.read_text())
    residuals = {}
    for t in telemetry:
        for slot in t["image_data"]:
            if slot["IMGFUNC"] != 1:
                continue
            yang, zang = yang_zang[slot["IMGNUM"]]
            r = np.hypot(slot["YAG"] - yang, slot["ZAG"] - zang)
            residuals[slot["IMGNUM"]] = min(r, residuals.get(slot["IMGNUM"], np.inf))

    assert residuals, "no tracked slots found in the OBC dump"
    matched = [slot for slot, r in residuals.items() if r < 25.0]
    assert len(matched) >= 6, (
        f"only {len(matched)}/8 slots matched: residuals={residuals}"
    )
