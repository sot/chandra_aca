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
    assert [len(r_full[r_full['imgnum'] == i]) for i in range(8)] == [1, 1, 1, 4, 4, 4, 4, 4]
    assert np.all(r_full[r_full['imgnum'] == 3]['size'].data == ['6X6', '6X6', '6X6', '6X6'])

    r = vstack([maude_decom._assemble_img(i, pea, data) for i in range(8)])
    assert [len(r[r['imgnum'] == i]) for i in range(8)] == [10, 10, 10, 10, 10, 10, 10, 10]
    assert np.all(r[r['imgnum'] == 3]['size'].data ==
                  ['6X62', '6X61', '6X62', '6X61', '6X62', '6X61', '6X62', '6X61', '6X62', '6X61'])

    eq = (r_full[r_full['imgnum'] == 0]['img'].data ==
          np.nansum(r[r['imgnum'] == 0]['img'].data[3:7].reshape((-1, 4, 8, 8)), axis=1))
    assert np.all(eq)

    eq = (r_full[r_full['imgnum'] == 3]['img'].data ==
          np.nansum(r[r['imgnum'] == 3]['img'].data[1:-1].reshape((-1, 2, 8, 8)), axis=1))
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
    assert np.all((table[0]['img'] == img_ref) + (np.isnan(img_ref)))

    table = maude_decom._assemble_img(4, pea, data, full=True, calibrate=True)
    img_ref = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, 10.00, 26.00, 35.00, 32.00, np.nan, np.nan],
               [np.nan, 26.00, 59.00, 167.00, 153.00, 59.00, 45.00, np.nan],
               [np.nan, 70.00, 252.00, 813.00, 864.00, 425.00, 203.00, np.nan],
               [np.nan, 67.00, 635.00, 588.00, 549.00, 692.00, 177.00, np.nan],
               [np.nan, 33.00, 227.00, 322.00, 595.00, 342.00, 48.00, np.nan],
               [np.nan, np.nan, 48.00, 92.00, 301.00, 38.00, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    assert np.all((table[0]['img'] == img_ref) + (np.isnan(img_ref)))


def test_fetch():
    pea = 1
    start, stop = 686111007, 686111017

    result = maude_decom.fetch(start, stop)
    assert np.all(result['time'] >= start)
    assert np.all(result['time'] <= stop)
    assert len(result) == 80

    result = maude_decom.fetch(start, stop, full=True)
    assert np.all(result['time'] >= start)
    assert np.all(result['time'] <= stop)
    assert len(result) == 31

    result = maude_decom.fetch(start, stop, calibrate=True)
    assert np.all(result['time'] >= start)
    assert np.all(result['time'] <= stop)
    assert len(result) == 80

    result = maude_decom.fetch(start, stop, adjust_time=True)
    assert np.all(result['time'] >= start)
    assert np.all(result['time'] <= stop)
    assert len(result) == 64
