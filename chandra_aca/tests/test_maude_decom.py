#!/usr/bin/env python3

import copy
import os
import pickle

import astropy.units as u
import maude
import numpy as np
import pytest
from cxotime import CxoTime

from chandra_aca import maude_decom

test_data = {}

with open(
    os.path.join(os.path.dirname(__file__), "data", "maude_decom.pkl"), "rb"
) as f:
    test_data.update(pickle.load(f))


def compare_tables(t1, t2, keys=None, exclude=()):
    assert len(t1) == len(t2)
    assert sorted(t1.colnames) == sorted(t2.colnames)
    _compare_common_columns(t1, t2, keys, exclude)


def _compare_common_columns(t1, t2, keys=None, exclude=()):
    from astropy import table

    if keys is None:
        keys = [k for k in t1.colnames if k in t2.colnames]
    for key in exclude:
        keys.remove(key)
    errors = []
    for name in keys:
        col_1, col_2 = t1[name], t2[name]
        ok = type(col_1) is type(col_2)
        if type(col_1) is table.MaskedColumn and type(col_1) is table.MaskedColumn:
            ok &= np.all(col_1.mask == col_2.mask)
        if np.issubdtype(col_1.dtype, np.inexact) or np.issubdtype(
            col_1.dtype.base, np.inexact
        ):
            c = np.isclose(col_1, col_2) | (np.isnan(col_1) & np.isnan(col_1))
        else:
            c = col_1 == col_2
        if type(col_1) is table.MaskedColumn:
            if np.any(~c.mask):
                ok &= np.all(c[~c.mask])
        else:
            ok &= np.all(c)
        if not ok:
            errors.append([name, str(col_1.data), str(col_2.data)])
    if errors:
        msg = "The following columns do not match:\n\n"
        for name, e1, e2 in errors:
            msg += f"  - {name}\n"
            msg += f"    t1: {e1}\n"
            msg += f"    t2: {e2}\n\n"
        raise Exception(msg)


def test_vcdu_0_raw():
    data = maude_decom.get_raw_aca_packets(176267186, 176267186)
    assert data["flags"] == 0
    for key in ["TIME", "MJF", "MNF", "packets"]:
        assert len(data[key]) == 0

    ref_data = test_data["686111007-686111017"]["raw"]
    data = maude_decom.get_raw_aca_packets(686111007, 686111017)
    assert np.all(np.isclose(data["TIME"], ref_data["TIME"], rtol=0, atol=1e-3))
    for key in ["MJF", "MNF"]:
        assert np.all(data[key] == ref_data[key])
    assert data["packets"] == ref_data["packets"]
    assert data["flags"] == ref_data["flags"]


def test_vcdu_0_raw_2():
    # this test gets the raw ACA packets (225 byte packets that come in 4 VCDU frames)
    # for two time ranges, where the second time range is a subset of the first.

    t1 = 387039066.184
    t2 = 387039216.184
    t3 = 387039370.034

    frames = maude.get_frames(t1, t3, channel="FLIGHT")

    raw_aca_packets_1 = maude_decom.get_raw_aca_packets(
        t1, t3, maude_result=frames, channel="FLIGHT"
    )

    # This time we request the packets for a time range that is a subset of the previous one.
    # However, we pass _all_ the frames from the previous query, so the times do not match.
    raw_aca_packets_2 = maude_decom.get_raw_aca_packets(
        t2, t3, maude_result=frames, channel="FLIGHT"
    )

    n_frames = raw_aca_packets_2["TIME"].shape[0]
    assert np.all(raw_aca_packets_2["TIME"] == raw_aca_packets_1["TIME"][-n_frames:])
    assert np.all(raw_aca_packets_2["MJF"] == raw_aca_packets_1["MJF"][-n_frames:])
    assert np.all(raw_aca_packets_2["MNF"] == raw_aca_packets_1["MNF"][-n_frames:])
    assert np.all(
        [
            raw_aca_packets_2["packets"][i]
            == raw_aca_packets_1["packets"][-n_frames:][i]
            for i in range(n_frames)
        ]
    )


def test_blob_0_raw():
    blobs = maude_decom.get_raw_aca_blobs(176267186, 176267186)
    t = maude.blobs_to_table(**blobs)[["TIME", "CVCMJCTR", "CVCMNCTR"]]
    assert len(t) == 0

    ref_data = test_data["686111007-686111017"]["raw"]
    blobs = maude_decom.get_raw_aca_blobs(686111007, 686111017)
    t = maude.blobs_to_table(**blobs)[["TIME", "CVCMJCTR", "CVCMNCTR"]]
    assert np.all(np.isclose(t["TIME"], ref_data["TIME"], rtol=0, atol=1e-3))
    assert np.all(t["CVCMJCTR"] == ref_data["MJF"])
    assert np.all(t["CVCMNCTR"] == ref_data["MNF"])


def test_scale():
    raw = test_data["686111007-686111017"]["raw"]
    table = maude_decom._get_aca_packets(
        raw, 686111007, 686111017, combine=True, calibrate=False
    )
    img_ref = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 60.00, 76.00, 85.00, 82.00, np.nan, np.nan],
        [np.nan, 76.00, 109.00, 217.00, 203.00, 109.00, 95.00, np.nan],
        [np.nan, 120.00, 302.00, 863.00, 914.00, 475.00, 253.00, np.nan],
        [np.nan, 117.00, 685.00, 638.00, 599.00, 742.00, 227.00, np.nan],
        [np.nan, 83.00, 277.00, 372.00, 645.00, 392.00, 98.00, np.nan],
        [np.nan, np.nan, 98.00, 142.00, 351.00, 88.00, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    assert np.all(
        (table[table["IMGNUM"] == 4]["IMG"][0].data == img_ref) + (np.isnan(img_ref))
    )

    table = maude_decom._get_aca_packets(
        raw, 686111007, 686111017, combine=True, calibrate=True
    )
    img_ref = [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, 10.00, 26.00, 35.00, 32.00, np.nan, np.nan],
        [np.nan, 26.00, 59.00, 167.00, 153.00, 59.00, 45.00, np.nan],
        [np.nan, 70.00, 252.00, 813.00, 864.00, 425.00, 203.00, np.nan],
        [np.nan, 67.00, 635.00, 588.00, 549.00, 692.00, 177.00, np.nan],
        [np.nan, 33.00, 227.00, 322.00, 595.00, 342.00, 48.00, np.nan],
        [np.nan, np.nan, 48.00, 92.00, 301.00, 38.00, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ]
    assert np.all(
        (table[table["IMGNUM"] == 4]["IMG"][0].data == img_ref) + (np.isnan(img_ref))
    )


def test_partial_images():
    mask = {
        "4X41": np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        "6X61": np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        "6X62": np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, False, False, False, False, True, True],
                [True, False, True, True, True, True, False, True],
                [True, False, True, True, True, True, False, True],
                [True, False, True, True, True, True, False, True],
                [True, False, True, True, True, True, False, True],
                [True, True, False, False, False, False, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        "8X81": np.array(
            [
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        "8X82": np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        "8X83": np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        "8X84": np.array(
            [
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
            ]
        ),
    }
    mask.update(
        {
            0: mask["4X41"],
            1: mask["6X61"],
            2: mask["6X62"],
            4: mask["8X81"],
            5: mask["8X82"],
            6: mask["8X83"],
            7: mask["8X84"],
        }
    )
    aca_packets = test_data["686111007-686111017"]["packets"]
    non_combined_aca_packets = [
        maude_decom._combine_aca_packets([row]) for slot in aca_packets for row in slot
    ]

    for packet in non_combined_aca_packets:
        assert "IMG" in packet
        assert packet["IMG"].shape == (8, 8)
        assert np.all(packet["IMG"].mask == mask[packet["IMGTYPE"]])

    raw = test_data["686111007-686111017"]["raw"]
    table = maude_decom._get_aca_packets(raw, 686111007, 686111017, combine=False)

    for i in range(len(table)):
        assert np.all(table[i]["IMG"].mask == mask[table[i]["IMGTYPE"]])


def test_aca_images_chunks_1():
    """Test that a fetch longer than the maude step limit works"""
    single_fetch_limit = maude_decom.MAUDE_SINGLE_FETCH_LIMIT
    maude_decom.MAUDE_SINGLE_FETCH_LIMIT = 1 * u.min
    start = "2023:001:00:00:01.000"
    stop = "2023:001:00:05:01.000"
    tstart = CxoTime(start).secs
    tstop = CxoTime(stop).secs
    try:
        # Include the "times" metadata in the returned images table for testing
        imgs = maude_decom.get_aca_images(start, stop, set_times_metadata=True)
    finally:
        maude_decom.MAUDE_SINGLE_FETCH_LIMIT = single_fetch_limit

    imgs.sort(["TIME", "IMGNUM"])

    for slot in range(8):
        # Confirm that the data is basically contiguous
        ok_slot = imgs["IMGNUM"] == slot
        # Confirm no duplicates by VCDUCTR
        assert len(np.unique(imgs[ok_slot]["VCDUCTR"])) == len(imgs[ok_slot])
        # Confirm for these 8x8 data that there's no gap > 5 seconds
        assert np.max(np.diff(imgs[ok_slot]["TIME"])) < 5

    # Confirm that the returned data times are within the start stop
    assert imgs["TIME"][0] >= tstart
    assert imgs["TIME"][-1] <= tstop

    # Confirm that the beginning and end match the expected values
    assert abs(imgs[0]["TIME"] - tstart) < 2
    assert abs(imgs[-1]["TIME"] - tstop) < 2

    imgs_start = maude_decom.get_aca_images(start, CxoTime(start) + 60 * u.s)
    imgs_stop = maude_decom.get_aca_images(CxoTime(stop) - 60 * u.s, stop)
    imgs_start.sort(["TIME", "IMGNUM"])
    imgs_stop.sort(["TIME", "IMGNUM"])
    assert np.all(imgs[0] == imgs_start[0])
    assert np.all(imgs[-1] == imgs_stop[-1])

    # Check the linspace times in the returned data
    used_step_max = 1 * u.min - 1 * u.s
    deltas_lte = [t <= used_step_max for t in np.diff(imgs.meta["times"])]
    assert np.all(deltas_lte)


def test_aca_images_chunks_2():
    """Test that a fetch longer than MAUDE_FETCH_LIMIT throws a ValueError"""
    start = CxoTime("2023:001:00:00:01.000")
    stop = start + (maude_decom.MAUDE_FETCH_LIMIT * 1.1)
    with pytest.raises(ValueError, match="stop - start cannot be greater than"):
        maude_decom.get_aca_images(start, stop)


def test_vcdu_vs_level0():
    from astropy.table import Table

    start, stop = (686111020, 686111030)

    table = maude_decom.get_aca_packets(start, stop, level0=True)

    table2 = maude_decom.get_aca_images(start, stop)
    for col, col2 in zip(table.itercols(), table2.itercols(), strict=False):
        assert np.all(col == col2)

    raw = test_data["686111010-686111040"]["raw"]
    table2 = maude_decom._get_aca_packets(
        raw, start, stop, combine=True, adjust_time=True, calibrate=True
    )
    for col, col2 in zip(table.itercols(), table2.itercols(), strict=False):
        if col.name in ["TIME", "END_INTEG_TIME"]:
            assert np.all(np.isclose(col2, col, rtol=0, atol=1e-3))
        else:
            assert np.all(col == col2)

    names = [
        "TIME",
        "MJF",
        "MNF",
        "END_INTEG_TIME",
        "INTEG",
        "GLBSTAT",
        "COMMCNT",
        "COMMPROG",
        "IMGROW0",
        "IMGCOL0",
        "IMGSCALE",
        "BGDAVG",
        "BGDRMS",
        "TEMPCCD",
        "TEMPHOUS",
        "TEMPPRIM",
        "TEMPSEC",
        "BGDSTAT",
    ]

    for slot in range(8):
        l0_test_data = Table.read(
            os.path.join(
                os.path.dirname(__file__), "data", f"test_level0_{slot}.fits.gz"
            ),
            # Undocumented way to stop UnitsWarning 'DN' did not parse warning. See
            # https://github.com/astropy/astropy/issues/14330#issuecomment-1406538512
            unit_parse_strict="silent",
        )
        td = l0_test_data[
            (l0_test_data["TIME"] <= stop) * (l0_test_data["TIME"] >= start)
        ]

        tt = table[table["IMGNUM"] == slot]

        assert len(tt) == len(td)
        n = (
            tt["IMG"].shape[1] - td["IMGRAW"].shape[1]
        ) // 2  # the padding to fit in an 8x8 image
        imgraw = (
            np.pad(td["IMGRAW"], n, "constant")[n:-n] if n else td["IMGRAW"]
        )  # padded image
        t = np.all(np.isclose(tt["IMG"], imgraw))
        assert t
        for name in names:
            t = np.all(np.isclose(tt[name], td[name]))
            assert t


def test_vcdu_packet_combination():
    import copy

    # 8x8
    test_packets_groups = [
        {"IMGTYPE": 7, "MJF": 9800, "MNF": 84},
        {"IMGTYPE": 4, "MJF": 9800, "MNF": 88},
        {"IMGTYPE": 5, "MJF": 9800, "MNF": 92},
        {"IMGTYPE": 6, "MJF": 9800, "MNF": 96},
        {"IMGTYPE": 7, "MJF": 9800, "MNF": 100},
        {"IMGTYPE": 4, "MJF": 9800, "MNF": 104},
        {"IMGTYPE": 5, "MJF": 9800, "MNF": 108},
        {"IMGTYPE": 6, "MJF": 9800, "MNF": 112},
        {"IMGTYPE": 7, "MJF": 9800, "MNF": 116},
        {"IMGTYPE": 4, "MJF": 9800, "MNF": 120},
        {"IMGTYPE": 5, "MJF": 9800, "MNF": 124},
        {"IMGTYPE": 6, "MJF": 9801, "MNF": 0},
    ]

    packets = copy.deepcopy(test_packets_groups)
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 84)],
        [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
        [(9800, 104), (9800, 108), (9800, 112), (9800, 116)],
        [(9800, 120), (9800, 124), (9801, 0)],
    ], "All 8X8"

    packets = copy.deepcopy(test_packets_groups)
    del packets[4]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 84)],
        [(9800, 88), (9800, 92), (9800, 96)],
        [(9800, 104), (9800, 108), (9800, 112), (9800, 116)],
        [(9800, 120), (9800, 124), (9801, 0)],
    ], "Missing 8X81"

    packets = copy.deepcopy(test_packets_groups)
    del packets[5]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, True)
    ]
    assert p == [
        [(9800, 88), (9800, 92), (9800, 96), (9800, 100)]
    ], "Whole, missing 8X81"

    packets = copy.deepcopy(test_packets_groups)
    del packets[5]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 84)],
        [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
        [(9800, 108), (9800, 112), (9800, 116)],
        [(9800, 120), (9800, 124), (9801, 0)],
    ], "Missing 8X81"

    packets = copy.deepcopy(test_packets_groups)
    del packets[6]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 84)],
        [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
        [(9800, 104), (9800, 112), (9800, 116)],
        [(9800, 120), (9800, 124), (9801, 0)],
    ], "Missing 8X82"

    packets = copy.deepcopy(test_packets_groups)
    del packets[7]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 84)],
        [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
        [(9800, 104), (9800, 108), (9800, 116)],
        [(9800, 120), (9800, 124), (9801, 0)],
    ], "Missing 8X83"

    packets = copy.deepcopy(test_packets_groups)
    del packets[8]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 84)],
        [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
        [(9800, 104), (9800, 108), (9800, 112)],
        [(9800, 120), (9800, 124), (9801, 0)],
    ], "Missing 8X84"

    # 6x6
    test_packets_groups = [
        {"IMGTYPE": 2, "MJF": 9800, "MNF": 120},
        {"IMGTYPE": 1, "MJF": 9800, "MNF": 124},
        {"IMGTYPE": 2, "MJF": 9801, "MNF": 0},
        {"IMGTYPE": 1, "MJF": 9801, "MNF": 4},
        {"IMGTYPE": 2, "MJF": 9801, "MNF": 8},
        {"IMGTYPE": 1, "MJF": 9801, "MNF": 12},
        {"IMGTYPE": 2, "MJF": 9801, "MNF": 16},
        {"IMGTYPE": 1, "MJF": 9801, "MNF": 20},
    ]

    packets = copy.deepcopy(test_packets_groups)
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 120)],
        [(9800, 124), (9801, 0)],
        [(9801, 4), (9801, 8)],
        [(9801, 12), (9801, 16)],
        [(9801, 20)],
    ], "All 6X61"

    packets = copy.deepcopy(test_packets_groups)
    del packets[2]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 120)],
        [(9800, 124)],
        [(9801, 4), (9801, 8)],
        [(9801, 12), (9801, 16)],
        [(9801, 20)],
    ], "Missing 6X62"

    packets = copy.deepcopy(test_packets_groups)
    del packets[1]
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]

    assert p == [
        [(9800, 120)],
        [(9801, 0)],
        [(9801, 4), (9801, 8)],
        [(9801, 12), (9801, 16)],
        [(9801, 20)],
    ], "Missing 6X61"

    packets = copy.deepcopy(test_packets_groups)
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, True)
    ]
    assert p == [
        [(9800, 124), (9801, 0)],
        [(9801, 4), (9801, 8)],
        [(9801, 12), (9801, 16)],
    ]

    # 4x4
    test_packets_groups = [
        {"IMGTYPE": 0, "MJF": 9800, "MNF": 120},
        {"IMGTYPE": 0, "MJF": 9800, "MNF": 124},
        {"IMGTYPE": 0, "MJF": 9801, "MNF": 0},
        {"IMGTYPE": 0, "MJF": 9801, "MNF": 4},
        {"IMGTYPE": 0, "MJF": 9801, "MNF": 8},
        {"IMGTYPE": 0, "MJF": 9801, "MNF": 12},
        {"IMGTYPE": 0, "MJF": 9801, "MNF": 16},
        {"IMGTYPE": 0, "MJF": 9801, "MNF": 20},
    ]

    packets = copy.deepcopy(test_packets_groups)
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, False)
    ]
    assert p == [
        [(9800, 120)],
        [(9800, 124)],
        [(9801, 0)],
        [(9801, 4)],
        [(9801, 8)],
        [(9801, 12)],
        [(9801, 16)],
        [(9801, 20)],
    ]

    packets = copy.deepcopy(test_packets_groups)
    p = [
        [(q["MJF"], q["MNF"]) for q in p]
        for p in maude_decom._group_packets(packets, True)
    ]
    assert p == [
        [(9800, 120)],
        [(9800, 124)],
        [(9801, 0)],
        [(9801, 4)],
        [(9801, 8)],
        [(9801, 12)],
        [(9801, 16)],
        [(9801, 20)],
    ]

    [None for _ in maude_decom._group_packets(packets[:0], False)]
    [None for _ in maude_decom._group_packets(packets[:0], True)]
    [None for _ in maude_decom._group_packets(packets[:1], False)]
    [None for _ in maude_decom._group_packets(packets[:1], True)]

    # VCDU counter rolls over
    pkts = [
        {"MJF": 131071, "MNF": 68, "IMGTYPE": 7},
        {"MJF": 131071, "MNF": 72, "IMGTYPE": 4},
        {"MJF": 131071, "MNF": 76, "IMGTYPE": 5},
        {"MJF": 131071, "MNF": 80, "IMGTYPE": 6},
        {"MJF": 131071, "MNF": 84, "IMGTYPE": 7},
        {"MJF": 131071, "MNF": 88, "IMGTYPE": 4},
        {"MJF": 131071, "MNF": 92, "IMGTYPE": 5},
        {"MJF": 131071, "MNF": 96, "IMGTYPE": 6},
        {"MJF": 131071, "MNF": 100, "IMGTYPE": 7},
        {"MJF": 131071, "MNF": 104, "IMGTYPE": 4},
        {"MJF": 131071, "MNF": 108, "IMGTYPE": 5},
        {"MJF": 131071, "MNF": 112, "IMGTYPE": 6},
        {"MJF": 131071, "MNF": 116, "IMGTYPE": 7},
        {"MJF": 131071, "MNF": 120, "IMGTYPE": 4},
        {"MJF": 131071, "MNF": 124, "IMGTYPE": 5},
        {"MJF": 0, "MNF": 0, "IMGTYPE": 6},
        {"MJF": 0, "MNF": 4, "IMGTYPE": 7},
        {"MJF": 0, "MNF": 8, "IMGTYPE": 4},
        {"MJF": 0, "MNF": 12, "IMGTYPE": 5},
        {"MJF": 0, "MNF": 16, "IMGTYPE": 6},
        {"MJF": 0, "MNF": 20, "IMGTYPE": 7},
        {"MJF": 0, "MNF": 24, "IMGTYPE": 4},
        {"MJF": 0, "MNF": 28, "IMGTYPE": 5},
    ]

    ref_g_pkts = [
        [
            {"MJF": 131071, "MNF": 72, "IMGTYPE": 4},
            {"MJF": 131071, "MNF": 76, "IMGTYPE": 5},
            {"MJF": 131071, "MNF": 80, "IMGTYPE": 6},
            {"MJF": 131071, "MNF": 84, "IMGTYPE": 7},
        ],
        [
            {"MJF": 131071, "MNF": 88, "IMGTYPE": 4},
            {"MJF": 131071, "MNF": 92, "IMGTYPE": 5},
            {"MJF": 131071, "MNF": 96, "IMGTYPE": 6},
            {"MJF": 131071, "MNF": 100, "IMGTYPE": 7},
        ],
        [
            {"MJF": 131071, "MNF": 104, "IMGTYPE": 4},
            {"MJF": 131071, "MNF": 108, "IMGTYPE": 5},
            {"MJF": 131071, "MNF": 112, "IMGTYPE": 6},
            {"MJF": 131071, "MNF": 116, "IMGTYPE": 7},
        ],
        [
            {"MJF": 131071, "MNF": 120, "IMGTYPE": 4},
            {"MJF": 131071, "MNF": 124, "IMGTYPE": 5},
            {"MJF": 0, "MNF": 0, "IMGTYPE": 6},
            {"MJF": 0, "MNF": 4, "IMGTYPE": 7},
        ],
        [
            {"MJF": 0, "MNF": 8, "IMGTYPE": 4},
            {"MJF": 0, "MNF": 12, "IMGTYPE": 5},
            {"MJF": 0, "MNF": 16, "IMGTYPE": 6},
            {"MJF": 0, "MNF": 20, "IMGTYPE": 7},
        ],
    ]
    g_pkts = list(maude_decom._group_packets(pkts))
    assert g_pkts == ref_g_pkts


def test_row_col():
    # this tests consistency between different row/col references
    # absolute values are tested elsewhere
    start, stop = (686111020, 686111030)
    raw = test_data[f"{start}-{stop}"]["raw"]
    table = maude_decom._get_aca_packets(raw, start, stop, combine=False)

    assert np.all(
        table[table["IMGTYPE"] == 0]["IMGCOL0"] - 2
        == table[table["IMGTYPE"] == 0]["IMGCOL0_8X8"]
    )
    assert np.all(
        table[table["IMGTYPE"] == 0]["IMGROW0"] - 2
        == table[table["IMGTYPE"] == 0]["IMGROW0_8X8"]
    )

    assert np.all(
        table[table["IMGTYPE"] == 4]["IMGCOL0"]
        == table[table["IMGTYPE"] == 4]["IMGCOL0_8X8"]
    )
    assert np.all(
        table[table["IMGTYPE"] == 4]["IMGROW0"]
        == table[table["IMGTYPE"] == 4]["IMGROW0_8X8"]
    )

    assert np.all(
        table[table["IMGTYPE"] == 1]["IMGCOL0"] - 1
        == table[table["IMGTYPE"] == 1]["IMGCOL0_8X8"]
    )
    assert np.all(
        table[table["IMGTYPE"] == 1]["IMGROW0"] - 1
        == table[table["IMGTYPE"] == 1]["IMGROW0_8X8"]
    )


def test_start_stop():
    # check some conventions on start/stop times
    start, stop = (686111020, 686111028.893)

    raw = test_data["686111010-686111040"]["raw"]
    table = maude_decom._get_aca_packets(
        raw, start, stop, combine=True, adjust_time=True, calibrate=True
    )
    n1 = len(table)
    # query is done in the closed/open interval [start, stop)
    start, stop = table["TIME"].min(), table["TIME"].max()
    table = maude_decom._get_aca_packets(
        raw, start, stop, combine=True, adjust_time=True, calibrate=True
    )
    n2 = len(table)
    assert start == table["TIME"].min()
    assert stop > table["TIME"].max()
    assert n2 == n1 - 8

    # it should be ok to query a few kiloseconds
    start, stop = (686111020, 686112020)
    _ = maude_decom.get_aca_packets(start, stop)

    # it should raise an exception if the interval is too large
    with pytest.raises(ValueError, match="Maximum allowed"):
        start, stop = (686111020, 686219020)
        _ = maude_decom.get_aca_packets(start, stop)


def test_dynbgd_decom():
    """
    This test looks at telemetry data around times when either AABGDTYP or AAPIXTLM change.
    It checks two things: that the values are properly decommuted and that the proper values of
    AABGDTYP and AAPIXTLM are set when packets are combined.

    AABGDTYP and AAPIXTLM are the same for all slots within an ACA packet, but they might not
    correspond to actual pixel data. The values that correspond to the pixel data being sent are
    the values in the first packet of the image (image types 0, 1 and 4).

    For example, this series of ACA packets shows a change in AAPIXTLM and AABGDTYP at frame 39736:
         TIME     VCDUCTR IMGTYPE AABGDTYP AAPIXTLM
    ------------- ------- ------- ------ ------
    694916092.438   39732       4      1      2
    694916093.464   39736       5      0      0
    694916094.483   39740       6      0      0
    694916095.509   39744       7      0      0

    however, frames 39732 to 39744 form an 8x8 image, and the value of AAPIXTLM in the first packet
    of the image is 2. Therefore, the packets must be combined to give
         TIME     VCDUCTR AABGDTYP AAPIXTLM
    ------------- ------- ------ ------
    694916092.438   39732      1      2
    """

    with open(
        os.path.join(os.path.dirname(__file__), "data", "dynbgd.pkl"), "rb"
    ) as out:
        raw_frames, partial_packets, grouped_packets = pickle.load(out)
        for key in raw_frames:
            start, stop = key
            partial_packets_2 = maude_decom._get_aca_packets(
                raw_frames[key], start=start, stop=stop
            )
            grouped_packets_2 = maude_decom._get_aca_packets(
                raw_frames[key], combine=True, start=start, stop=stop
            )
            for slot in range(8):
                assert np.all(
                    partial_packets[key][
                        "TIME", "VCDUCTR", "IMGTYPE", "AABGDTYP", "AAPIXTLM"
                    ]
                    == partial_packets_2[
                        "TIME", "VCDUCTR", "IMGTYPE", "AABGDTYP", "AAPIXTLM"
                    ]
                )
                assert np.all(
                    grouped_packets[key][
                        "TIME", "VCDUCTR", "IMGTYPE", "AABGDTYP", "AAPIXTLM"
                    ]
                    == grouped_packets_2[
                        "TIME", "VCDUCTR", "IMGTYPE", "AABGDTYP", "AAPIXTLM"
                    ]
                )


def test_get_aca_blobs():
    t0 = maude_decom.get_aca_packets(686111007, 686111017, frames=True)
    t1 = maude_decom.get_aca_packets(686111007, 686111017, blobs=True)
    compare_tables(t0, t1, exclude=["COMMPROG_REPEAT"])


def test_blob_frame_consistency():
    slot = 6
    start, stop = (686111012.0, 686111212.0)
    slot_data = maude_decom.get_aca_packets(
        start,
        stop,
        level0=True,
        blobs=False,
    )
    slot_data_2 = maude_decom.get_aca_packets(
        start,
        stop,
        level0=True,
        blobs=True,
    )
    slot_data = slot_data[slot_data["IMGNUM"] == slot]
    slot_data_2 = slot_data_2[slot_data_2["IMGNUM"] == slot]
    assert len(slot_data) == len(slot_data_2)
    compare_tables(slot_data, slot_data_2, exclude=["COMMPROG_REPEAT"])

    start, stop = 686105735.6057751, 686105740.7507753
    slot_data = maude_decom.get_aca_packets(start, stop, blobs=False)
    slot_data_blobs = maude_decom.get_aca_packets(start, stop, blobs=True)
    compare_tables(slot_data, slot_data_blobs, exclude=["COMMPROG_REPEAT"])


def test_get_aca_packets_blobs():
    # this test indirectly checks that the blobs argument in get_aca_packets is used
    # because the blobs correspond to a shorter interval and therefore the should match
    # the result from the shorter interval
    slot = 6
    start, stop = (686111012.0, 686111111.0)
    slot_data = maude_decom.get_aca_packets(
        start,
        stop,
        level0=True,
        blobs=True,
    )
    blobs = maude.get_blobs(
        start=start,
        stop=stop + 4,
        channel="FLIGHT",
    )
    start, stop = (686111012.0, 686111212.0)
    slot_data_2 = maude_decom.get_aca_packets(
        start,
        stop,
        level0=True,
        blobs=blobs,
    )
    slot_data = slot_data[slot_data["IMGNUM"] == slot]
    slot_data_2 = slot_data_2[slot_data_2["IMGNUM"] == slot]
    assert len(slot_data) == len(slot_data_2)

    # calling the function again with the same blobs (they should not have been modified)
    ref_blobs = copy.deepcopy(blobs)
    slot_data_2 = maude_decom.get_aca_packets(
        start,
        stop,
        level0=True,
        blobs=blobs,
    )
    assert blobs == ref_blobs


@pytest.mark.parametrize("source", ["blobs", "frames"])
def test_imgtype_dnld(source):
    """
    Tests for the case when IMGTYPE is DNLD.
    """
    from cxotime import CxoTime

    start = CxoTime("2023:047:02:57")
    stop = CxoTime("2023:047:02:58")
    if source == "blobs":
        maude_result = maude.get_blobs(start, stop, channel="FLIGHT")
        args = {"blobs": maude_result}
    else:
        maude_result = maude.get_frames(start, stop, channel="FLIGHT")
        args = {"frames": maude_result}

    img = maude_decom.get_aca_packets(
        start, stop, combine=False, adjust_time=False, calibrate=False, **args
    )

    all_masked = np.array([np.all(row["IMG"].mask) for row in img])
    img_dnld = img["IMGTYPE"] == 3
    assert img_dnld.shape[0] > 0
    assert np.all(all_masked == img_dnld)


def test_vcdu_rollover():
    start = "2023:296:22:20:26"
    stop = "2023:296:22:20:50"
    middle = "2023:296:22:20:35.250"
    middle_2 = "2023:296:22:20:41.750"
    images_1 = maude_decom.get_aca_images(start, middle)
    images_2 = maude_decom.get_aca_images(middle_2, stop)
    images = maude_decom.get_aca_images(start, stop)

    assert images["TIME"].min() == images_1["TIME"].min()
    assert images["TIME"].max() != images_1["TIME"].max()
    assert images["TIME"].max() == images_2["TIME"].max()


def test_repeated_frames():
    import pickle

    with open(
        os.path.join(os.path.dirname(__file__), "data", "repeat_frames.pkl"), "rb"
    ) as fh:
        frames = pickle.load(fh)
    # this has repeated frames
    packets = maude_decom.get_aca_packets(
        start="2023:297:12:00:00", stop="2023:297:12:02:00", frames=frames
    )
    assert len(packets) == 40

    # and removing one VCDU frame results in one ACA packet missing (eight slots)
    frames["data"]["frames"] = (
        frames["data"]["frames"][:7] + frames["data"]["frames"][7 + 1 :]
    )
    packets = maude_decom.get_aca_packets(
        start="2023:297:12:00:00", stop="2023:297:12:02:00", frames=frames
    )
    assert len(packets) == 32


def test_filter_vcdu_jumps():
    filter_vcdu_test_cases = [
        {
            "inp": np.array([1, 2, 3, 4, 5, 6, 6, 7, 8, 9]),
            "exp": np.array(
                [False, False, False, True, True, True, False, True, False, False]
            ),
        },
        {
            "inp": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "exp": np.array(
                [False, False, False, True, True, True, True, False, False]
            ),
        },
        {
            "inp": np.array([0, 2, 3, 4, 5, 6, 7, 8, 9]),
            "exp": np.array(
                [False, False, False, True, True, True, True, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 3, 4, 5, 6, 7, 8, 9]),
            "exp": np.array(
                [False, False, False, True, True, True, True, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 2, 4, 5, 6, 7, 8, 9]),
            "exp": np.array(
                [False, False, False, True, True, True, True, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 2, 3, 5, 6, 7, 8, 9]),
            "exp": np.array(
                [True, True, True, True, False, False, False, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 6, 7, 8, 9]),
            "exp": np.array(
                [True, True, True, True, False, False, False, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 7, 8, 9]),
            "exp": np.array(
                [True, True, True, True, False, False, False, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 6, 8, 9]),
            "exp": np.array(
                [True, True, True, True, False, False, False, False, False]
            ),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 6, 7, 9]),
            "exp": np.array([True, True, True, True, True, True, True, True, False]),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            "exp": np.array([True, True, True, True, True, True, True, True, False]),
        },
        {
            "inp": np.array([2, 3, 4, 5, 6, 7, 8, 9]),
            "exp": np.array([False, False, True, True, True, True, False, False]),
        },
        {
            "inp": np.array([0, 3, 4, 5, 6, 7, 8, 9]),
            "exp": np.array([False, False, True, True, True, True, False, False]),
        },
        {
            "inp": np.array([0, 1, 4, 5, 6, 7, 8, 9]),
            "exp": np.array([False, False, True, True, True, True, False, False]),
        },
        {
            "inp": np.array([0, 1, 2, 5, 6, 7, 8, 9]),
            "exp": np.array([False, False, False, False, False, False, False, False]),
        },
        {
            "inp": np.array([0, 1, 2, 3, 6, 7, 8, 9]),
            "exp": np.array([True, True, True, True, False, False, False, False]),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 7, 8, 9]),
            "exp": np.array([True, True, True, True, False, False, False, False]),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 8, 9]),
            "exp": np.array([True, True, True, True, False, False, False, False]),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 6, 9]),
            "exp": np.array([True, True, True, True, False, False, False, False]),
        },
        {
            "inp": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "exp": np.array([True, True, True, True, True, True, True, True]),
        },
        {
            "inp": np.array(
                [
                    16777204,
                    16777205,
                    16777206,
                    16777207,
                    16777208,
                    16777209,
                    16777210,
                    16777211,
                    16777212,
                    16777213,
                    16777214,
                    16777215,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ]
            ),
            "exp": np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        },
        {
            "inp": np.array(
                [
                    16777204,
                    16777205,
                    16777206,
                    16777207,
                    16777208,
                    16777209,
                    16777210,
                    16777211,
                    16777212,
                    16777213,
                    16777214,
                    16777215,
                    14400,
                    14401,
                    14402,
                    14403,
                    14404,
                    14405,
                    14406,
                    14407,
                ]
            ),
            "exp": np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        },
        {
            "inp": np.array(
                [
                    16762804,
                    16762805,
                    16762806,
                    16762807,
                    16762808,
                    16762809,
                    16762810,
                    16762811,
                    16762812,
                    16762813,
                    16762814,
                    16762815,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                ]
            ),
            "exp": np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        },
        {
            "inp": np.array(
                [
                    16770004,
                    16770005,
                    16770006,
                    16770007,
                    16770008,
                    16770009,
                    16770010,
                    16770011,
                    16770012,
                    16770013,
                    16770014,
                    16770015,
                    7200,
                    7201,
                    7202,
                    7203,
                    7204,
                    7205,
                    7206,
                    7207,
                ]
            ),
            "exp": np.array(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
        },
    ]

    for case in filter_vcdu_test_cases:
        idx = maude_decom.filter_vcdu_jumps(case["inp"])
        assert np.all(idx == case["exp"])


@pytest.mark.parametrize("combine", [True, False])
def test_end_integ_time(combine):
    # this test verifies that the relations between END_INTEG_TIME and the two definitions of TIME
    # (VCDU time and CXC level0) are as specified in the specs.
    start_md, stop_md = (686111022.556, 686111030.776)
    table_maude = maude_decom.get_aca_packets(
        start_md, stop_md, combine=combine, adjust_time=False
    )

    # if adjust_time==True, times are shifted to the center of the integration window
    # and the result is filtered based on this shifted time, so to guarantee we get the same
    # VCDU frames, we need to give shifted times as input
    start_l0, stop_l0 = start_md - 1.696 / 2 - 1.025, stop_md - 1.696 / 2 - 1.025
    table_l0 = maude_decom.get_aca_packets(
        start_l0, stop_l0, combine=combine, adjust_time=True
    )

    # sanity check making sure we get the same VCDU frames
    assert len(table_l0) == len(table_maude)
    assert np.all(table_l0["VCDUCTR"] == table_maude["VCDUCTR"])

    # Test that END_INTEG_TIME is the same in both cases
    assert np.all(table_maude["END_INTEG_TIME"] == table_l0["END_INTEG_TIME"])

    # Test the relation between END_INTEG_TIME and TIME found in the ACA L0 ICD
    # this is true for all sub-images, because the time is set to that of the first sub-image
    # and it is the same for all sub-images in the same full image.
    assert np.all(
        np.isclose(
            table_l0["END_INTEG_TIME"],
            table_l0["TIME"] + table_l0["INTEG"] / 2,
            rtol=0,
        )
    )

    # Test the relation between END_INTEG_TIME and VCDU TIME (maude) found in the ACA L0 ICD
    # this is true only for the first sub-image, subsequent sub-images have a different VCDU time
    # whereas END_INTEG_TIME does not change.
    first_sub_image = np.in1d(table_maude["IMGTYPE"], [0, 1, 4])
    assert np.all(
        np.isclose(
            table_maude["END_INTEG_TIME"][first_sub_image],
            table_maude["TIME"][first_sub_image] - 1.025,
            rtol=0,
        )
    )


def test_get_aca_image_masked_columns_1():
    """Get images over a time covering end of maneuver and start of dwell

    This has fids (8x8), guides (6x6) and null images (4x4).
    """
    start = "2018:024:23:05:00"
    stop = "2018:024:23:10:00"
    imgs = maude_decom.get_aca_images(start, stop)
    # 4x4, 6x6, and 8x8 images
    assert set(imgs["IMGTYPE"]) == {0, 1, 4}
    # Null, tracking, and search
    assert set(imgs["IMGFUNC"]) == {0, 1, 3}
    cols_masked = {col.info.name for col in imgs.itercols() if hasattr(col, "mask")}
    assert cols_masked == {
        "BGDRMS",
        "TEMPCCD",
        "TEMPHOUS",
        "TEMPPRIM",
        "TEMPSEC",
        "BGDSTAT",
        "IMG",
    }


def test_get_aca_image_masked_columns_2():
    """Only 6x6 and 8x8 so only IMG is masked"""
    start = "2018:024:23:15:00"
    stop = "2018:024:23:20:00"
    imgs = maude_decom.get_aca_images(start, stop)
    assert set(imgs["IMGTYPE"]) == {1, 4}
    cols_masked = {col.info.name for col in imgs.itercols() if hasattr(col, "mask")}
    assert cols_masked == {"IMG"}


def test_get_aca_image_masked_columns_3():
    """Only 8x8 so only IMG masked"""
    start = "2024:001:00:00:00"
    stop = "2024:001:00:01:00"
    imgs = maude_decom.get_aca_images(start, stop)
    assert set(imgs["IMGTYPE"]) == {4}
    cols_masked = {col.info.name for col in imgs.itercols() if hasattr(col, "mask")}
    assert cols_masked == {"IMG"}
