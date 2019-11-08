#!/usr/bin/env python3

import os
import pickle
import numpy as np
from astropy.table import vstack

from chandra_aca import maude_decom


with open(os.path.join(os.path.dirname(__file__), 'data', 'maude_decom.pcl'), 'rb') as f:
    test_data = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'data', 'maude_decom2.pkl'), 'rb') as f:
    test_data.update(pickle.load(f))

def test_vcdu_0_raw():
    data = maude_decom.get_raw_aca_packets(176267186, 176267186)
    assert data['flags'] == 0
    for key in ['TIME', 'MJF', 'MNF', 'packets']:
        assert len(data[key]) == 0

    ref_data = test_data['686111007-686111017']['raw']
    data = maude_decom.get_raw_aca_packets(686111007, 686111017)
    for key in ['TIME', 'MJF', 'MNF']:
        assert np.all(data[key] == ref_data[key])
    assert data['packets'] == ref_data['packets']
    assert data['flags'] == ref_data['flags']


def test_vcdu_vs_level0():
    from astropy.table import Table

    start, stop = (686111020, 686111030)
    table = maude_decom.get_aca_packets(start, stop, level0=True)

    raw = test_data[f'686111010-686111040']['raw']
    table2 = maude_decom._get_aca_packets(raw, start, stop,
                                          combine=True, adjust_time=True, calibrate_pixels=True,
                                          adjust_corner=True, calibrate_temperatures=True)
    assert np.all(table == table2)

    names = ['TIME', 'MJF', 'MNF', 'END_INTEG_TIME', 'INTEG', 'GLBSTAT', 'COMMCNT',
             'COMMPROG', 'IMGROW0', 'IMGCOL0', 'IMGSCALE', 'BGDAVG', 'BGDRMS',
             'TEMPCCD', 'TEMPHOUS', 'TEMPPRIM', 'TEMPSEC', 'BGDSTAT']
    for slot in range(8):
        l0_test_data = Table.read(os.path.join(os.path.dirname(__file__), 'data',
                                            f'acaf686111014N001_{slot}_img0.fits.gz'))
        td = l0_test_data[(l0_test_data['TIME'] <= stop) * (l0_test_data['TIME'] >= start)]

        tt = table[table['IMGNUM'] == slot]

        assert len(tt) == len(td)
        n = (tt['IMG'].shape[1] - td['IMGRAW'].shape[1]) // 2
        imgraw = np.pad(td['IMGRAW'], n, 'constant')[n:-n] if n else td['IMGRAW']
        n = 'IMG'
        t = np.all(np.isclose(tt[n], imgraw))
        assert t
        for n in names:
            t = np.all(np.isclose(tt[n], td[n]))
            assert t


def test_vcdu_packet_combination():
    import copy

    # 8x8
    test_packets_groups = [
        {'IMGTYPE': 7, 'MJF': 9800, 'MNF': 84},
        {'IMGTYPE': 4, 'MJF': 9800, 'MNF': 88},
        {'IMGTYPE': 5, 'MJF': 9800, 'MNF': 92},
        {'IMGTYPE': 6, 'MJF': 9800, 'MNF': 96},
        {'IMGTYPE': 7, 'MJF': 9800, 'MNF': 100},
        {'IMGTYPE': 4, 'MJF': 9800, 'MNF': 104},
        {'IMGTYPE': 5, 'MJF': 9800, 'MNF': 108},
        {'IMGTYPE': 6, 'MJF': 9800, 'MNF': 112},
        {'IMGTYPE': 7, 'MJF': 9800, 'MNF': 116},
        {'IMGTYPE': 4, 'MJF': 9800, 'MNF': 120},
        {'IMGTYPE': 5, 'MJF': 9800, 'MNF': 124},
        {'IMGTYPE': 6, 'MJF': 9801, 'MNF': 0}]

    packets = copy.deepcopy(test_packets_groups)
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 84)],
                 [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
                 [(9800, 104), (9800, 108), (9800, 112), (9800, 116)],
                 [(9800, 120), (9800, 124), (9801, 0)]], 'All 8X8'

    packets = copy.deepcopy(test_packets_groups)
    del packets[4]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 84)],
                 [(9800, 88), (9800, 92), (9800, 96)],
                 [(9800, 104), (9800, 108), (9800, 112), (9800, 116)],
                 [(9800, 120), (9800, 124), (9801, 0)]], 'Missing 8X81'

    packets = copy.deepcopy(test_packets_groups)
    del packets[5]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, True)]
    assert p == [[(9800, 88), (9800, 92), (9800, 96), (9800, 100)]], 'Whole, missing 8X81'

    packets = copy.deepcopy(test_packets_groups)
    del packets[5]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 84)],
                 [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
                 [(9800, 108), (9800, 112), (9800, 116)],
                 [(9800, 120), (9800, 124), (9801, 0)]], 'Missing 8X81'

    packets = copy.deepcopy(test_packets_groups)
    del packets[6]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 84)],
                 [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
                 [(9800, 104), (9800, 112), (9800, 116)],
                 [(9800, 120), (9800, 124), (9801, 0)]], 'Missing 8X82'

    packets = copy.deepcopy(test_packets_groups)
    del packets[7]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 84)],
                 [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
                 [(9800, 104), (9800, 108), (9800, 116)],
                 [(9800, 120), (9800, 124), (9801, 0)]], 'Missing 8X83'

    packets = copy.deepcopy(test_packets_groups)
    del packets[8]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 84)],
                 [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
                 [(9800, 104), (9800, 108), (9800, 112)],
                 [(9800, 120), (9800, 124), (9801, 0)]], 'Missing 8X84'

    # 6x6
    test_packets_groups = [
        {'IMGTYPE': 2, 'MJF': 9800, 'MNF': 120},
        {'IMGTYPE': 1, 'MJF': 9800, 'MNF': 124},
        {'IMGTYPE': 2, 'MJF': 9801, 'MNF': 0},
        {'IMGTYPE': 1, 'MJF': 9801, 'MNF': 4},
        {'IMGTYPE': 2, 'MJF': 9801, 'MNF': 8},
        {'IMGTYPE': 1, 'MJF': 9801, 'MNF': 12},
        {'IMGTYPE': 2, 'MJF': 9801, 'MNF': 16},
        {'IMGTYPE': 1, 'MJF': 9801, 'MNF': 20}]

    packets = copy.deepcopy(test_packets_groups)
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 120)],
                 [(9800, 124), (9801, 0)],
                 [(9801, 4), (9801, 8)],
                 [(9801, 12), (9801, 16)],
                 [(9801, 20)]], 'All 6X61'

    packets = copy.deepcopy(test_packets_groups)
    del packets[2]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 120)],
                 [(9800, 124)],
                 [(9801, 4), (9801, 8)],
                 [(9801, 12), (9801, 16)],
                 [(9801, 20)]], 'Missing 6X62'

    packets = copy.deepcopy(test_packets_groups)
    del packets[1]
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]

    assert p == [[(9800, 120)],
                 [(9801, 0)],
                 [(9801, 4), (9801, 8)],
                 [(9801, 12), (9801, 16)],
                 [(9801, 20)]], 'Missing 6X61'

    packets = copy.deepcopy(test_packets_groups)
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, True)]
    assert p == [[(9800, 124), (9801, 0)], [(9801, 4), (9801, 8)], [(9801, 12), (9801, 16)]]

    # 4x4
    test_packets_groups = [
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 120},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 124},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 0},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 4},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 8},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 12},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 16},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 20}]

    packets = copy.deepcopy(test_packets_groups)
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, False)]
    assert p == [[(9800, 120)],
                 [(9800, 124)],
                 [(9801, 0)],
                 [(9801, 4)],
                 [(9801, 8)],
                 [(9801, 12)],
                 [(9801, 16)],
                 [(9801, 20)]]

    packets = copy.deepcopy(test_packets_groups)
    p = [[(q['MJF'], q['MNF']) for q in p] for p in maude_decom._group_packets(packets, True)]
    assert p == [[(9800, 120)],
                 [(9800, 124)],
                 [(9801, 0)],
                 [(9801, 4)],
                 [(9801, 8)],
                 [(9801, 12)],
                 [(9801, 16)],
                 [(9801, 20)]]

    [None for _ in maude_decom._group_packets(packets[:0], False)]
    [None for _ in maude_decom._group_packets(packets[:0], True)]
    [None for _ in maude_decom._group_packets(packets[:1], False)]
    [None for _ in maude_decom._group_packets(packets[:1], True)]
