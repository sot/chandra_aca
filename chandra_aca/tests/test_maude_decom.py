#!/usr/bin/env python3

import os
import pickle
import numpy as np
from astropy.table import vstack

from chandra_aca import maude_decom


with open(os.path.join(os.path.dirname(__file__), 'data', 'maude_decom.pcl'), 'rb') as f:
    test_data = pickle.load(f)


def test_empty():
    pea = 1
    data = {e['msid']: e for e in test_data['176267186-176267186']}
    table = maude_decom._assemble_img(1, pea, data, full=True, calibrate=False)
    assert len(table) == 0


def test_assembly():
    pea = 1

    data = {e['msid']: e for e in test_data['686111007-686111017']}
    r_full = vstack([maude_decom._assemble_img(i, pea, data, full=True) for i in range(8)])
    assert [len(r_full[r_full['IMGNUM'] == i]) for i in range(8)] == [1, 1, 1, 4, 4, 4, 4, 4]
    assert np.all(r_full[r_full['IMGNUM'] == 3]['IMGSIZE'].data == ['6X6', '6X6', '6X6', '6X6'])

    r = vstack([maude_decom._assemble_img(i, pea, data) for i in range(8)])
    assert [len(r[r['IMGNUM'] == i]) for i in range(8)] == [10, 10, 10, 10, 10, 10, 10, 10]
    assert np.all(r[r['IMGNUM'] == 3]['IMGSIZE'].data ==
                  ['6X62', '6X61', '6X62', '6X61', '6X62', '6X61', '6X62', '6X61', '6X62', '6X61'])

    eq = (r_full[r_full['IMGNUM'] == 0]['IMG'].data ==
          np.nansum(r[r['IMGNUM'] == 0]['IMG'].data[3:7].reshape((-1, 4, 8, 8)), axis=1))
    assert np.all(eq)

    eq = (r_full[r_full['IMGNUM'] == 3]['IMG'].data ==
          np.nansum(r[r['IMGNUM'] == 3]['IMG'].data[1:-1].reshape((-1, 2, 8, 8)), axis=1))
    e, m = np.broadcast_arrays(eq, maude_decom.PIXEL_MASK['6x6'])
    assert np.all(e[~m])


def test_scale():
    pea = 1

    data = {e['msid']: e for e in test_data['686111007-686111017']}
    table = maude_decom._assemble_img(4, pea, data, full=True, calibrate=False)
    img_ref = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, 60.00, 76.00, 85.00, 82.00, np.nan, np.nan],
               [np.nan, 76.00, 109.00, 217.00, 203.00, 109.00, 95.00, np.nan],
               [np.nan, 120.00, 302.00, 863.00, 914.00, 475.00, 253.00, np.nan],
               [np.nan, 117.00, 685.00, 638.00, 599.00, 742.00, 227.00, np.nan],
               [np.nan, 83.00, 277.00, 372.00, 645.00, 392.00, 98.00, np.nan],
               [np.nan, np.nan, 98.00, 142.00, 351.00, 88.00, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    assert np.all((table[0]['IMG'] == img_ref) + (np.isnan(img_ref)))

    table = maude_decom._assemble_img(4, pea, data, full=True, calibrate=True)
    img_ref = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, 10.00, 26.00, 35.00, 32.00, np.nan, np.nan],
               [np.nan, 26.00, 59.00, 167.00, 153.00, 59.00, 45.00, np.nan],
               [np.nan, 70.00, 252.00, 813.00, 864.00, 425.00, 203.00, np.nan],
               [np.nan, 67.00, 635.00, 588.00, 549.00, 692.00, 177.00, np.nan],
               [np.nan, 33.00, 227.00, 322.00, 595.00, 342.00, 48.00, np.nan],
               [np.nan, np.nan, 48.00, 92.00, 301.00, 38.00, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    assert np.all((table[0]['IMG'] == img_ref) + (np.isnan(img_ref)))


def test_fetch():
    start, stop = 686111007, 686111017

    result = maude_decom.fetch(start, stop)
    assert np.all(result['TIME'] >= start)
    assert np.all(result['TIME'] <= stop)
    assert len(result) == 80

    result = maude_decom.fetch(start, stop, full=True)
    assert np.all(result['TIME'] >= start)
    assert np.all(result['TIME'] <= stop)
    assert len(result) == 31

    result = maude_decom.fetch(start, stop, calibrate=True)
    assert np.all(result['TIME'] >= start)
    assert np.all(result['TIME'] <= stop)
    assert len(result) == 80

    result = maude_decom.fetch(start, stop, adjust_time=True)
    assert np.all(result['TIME'] >= start)
    assert np.all(result['TIME'] <= stop)
    assert len(result) == 64


def test_partial_images():
    mask = {
        '4X41': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, True, True, True, True, True, True]]),
        '6X61': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, True, True, True, True, True, True]]),
        '6X62': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, False, False, False, False, True, True],
                          [True, False, True, True, True, True, False, True],
                          [True, False, True, True, True, True, False, True],
                          [True, False, True, True, True, True, False, True],
                          [True, False, True, True, True, True, False, True],
                          [True, True, False, False, False, False, True, True],
                          [True, True, True, True, True, True, True, True]]),
        '8X81': np.array([[False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True]]),
        '8X82': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True]]),
        '8X83': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True]]),
        '8X84': np.array([[True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [True, True, True, True, True, True, True, True],
                          [False, False, False, False, False, False, False, False],
                          [False, False, False, False, False, False, False, False]]),
    }

    pea = 1
    data = {e['msid']: e for e in test_data['686111007-686111017']}
    r = vstack([maude_decom._assemble_img(i, pea, data) for i in range(8)])
    for i in range(len(r[r['IMGNUM'] == 0]['IMG'])):
        img = r[r['IMGNUM'] == 0]['IMG'][i]
        imgsize = r[r['IMGNUM'] == 0]['IMGSIZE'][i]
        assert np.all(np.isnan(img[mask[imgsize]]))
        assert np.all(~np.isnan(img[~mask[imgsize]]))


def test_vcdu_vs_level0():
    from astropy.table import Table

    start, stop = (686111020, 686111030)
    table = maude_decom.get_aca_packets(start, stop, level0=True)
    names = ['TIME', 'MJF', 'MNF', 'END_INTEG_TIME', 'INTEG', 'GLBSTAT', 'COMMCNT',
             'COMMPROG', 'IMGROW0', 'IMGCOL0', 'IMGSCALE', 'BGDAVG', 'BGDRMS',
             'TEMPCCD', 'TEMPHOUS', 'TEMPPRIM', 'TEMPSEC', 'BGDSTAT']
    for slot in range(8):
        test_data = Table.read(
            f'/Users/javierg/SAO/git/aca_view/aca_view/tests/data/acaf686111014N001_{slot}_img0.fits.gz')
        td = test_data[(test_data['TIME'] <= stop) * (test_data['TIME'] >= start)]

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
    assert [[(q['MJF'], q['MNF']) for q in p]
            for p in maude_decom.group_packets(packets, False)] == \
           [[(9800, 84)],
            [(9800, 88), (9800, 92), (9800, 96), (9800, 100)],
            [(9800, 104), (9800, 108), (9800, 112), (9800, 116)],
            [(9800, 120), (9800, 124), (9801, 0)]]

    packets = copy.deepcopy(test_packets_groups)
    del packets[4]
    assert [[(q['MJF'], q['MNF']) for q in p]
            for p in maude_decom.group_packets(packets, False)] == \
           [[(9800, 84)],
            [(9800, 88), (9800, 92), (9800, 96)],
            [(9800, 104), (9800, 108), (9800, 112), (9800, 116)],
            [(9800, 120), (9800, 124), (9801, 0)]]

    packets = copy.deepcopy(test_packets_groups)
    del packets[4]
    assert [[(q['MJF'], q['MNF']) for q in p]
            for p in maude_decom.group_packets(packets, True)] == \
           [[(9800, 104), (9800, 108), (9800, 112), (9800, 116)]]

    packets = copy.deepcopy(test_packets_groups)

    del packets[5]
    assert [[(q['MJF'], q['MNF']) for q in p]
            for p in maude_decom.group_packets(packets, True)] == \
           [[(9800, 88), (9800, 92), (9800, 96), (9800, 100)]]

    # 6x6
    test_packets_groups = [
        {'IMGTYPE': 2, 'MJF': 9800, 'MNF': 120},
        {'IMGTYPE': 1, 'MJF': 9800, 'MNF': 124},
        {'IMGTYPE': 2, 'MJF': 9801, 'MNF': 0},
        {'IMGTYPE': 1, 'MJF': 9800, 'MNF': 4},
        {'IMGTYPE': 2, 'MJF': 9800, 'MNF': 8},
        {'IMGTYPE': 1, 'MJF': 9800, 'MNF': 12},
        {'IMGTYPE': 2, 'MJF': 9800, 'MNF': 16},
        {'IMGTYPE': 1, 'MJF': 9800, 'MNF': 20}]

    packets = copy.deepcopy(test_packets_groups)
    assert [[(q['MJF'], q['MNF']) for q in p] for p in
            maude_decom.group_packets(packets, False)] == [[(9800, 120)],
                                                           [(9800, 124), (9801, 0)],
                                                           [(9800, 4), (9800, 8)],
                                                           [(9800, 12), (9800, 16)],
                                                           [(9800, 20)]]

    packets = copy.deepcopy(test_packets_groups)
    assert [[(q['MJF'], q['MNF']) for q in p]
            for p in maude_decom.group_packets(packets, True)] == \
           [[(9800, 124), (9801, 0)], [(9800, 4), (9800, 8)], [(9800, 12), (9800, 16)]]

    # 4x4
    test_packets_groups = [
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 120},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 124},
        {'IMGTYPE': 0, 'MJF': 9801, 'MNF': 0},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 4},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 8},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 12},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 16},
        {'IMGTYPE': 0, 'MJF': 9800, 'MNF': 20}]

    packets = copy.deepcopy(test_packets_groups)
    assert [[(q['MJF'], q['MNF']) for q in p] for p in
            maude_decom.group_packets(packets, False)] == [[(9800, 120)],
                                                           [(9800, 124)],
                                                           [(9801, 0)],
                                                           [(9800, 4)],
                                                           [(9800, 8)],
                                                           [(9800, 12)],
                                                           [(9800, 16)],
                                                           [(9800, 20)]]

    packets = copy.deepcopy(test_packets_groups)
    assert [[(q['MJF'], q['MNF']) for q in p] for p in
            maude_decom.group_packets(packets, True)] == [[(9800, 120)],
                                                          [(9800, 124)],
                                                          [(9801, 0)],
                                                          [(9800, 4)],
                                                          [(9800, 8)],
                                                          [(9800, 12)],
                                                          [(9800, 16)],
                                                          [(9800, 20)]]
