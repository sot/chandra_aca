#!/usr/bin/env python3

import os
import pickle
import numpy as np

from chandra_aca import maude_decom


with open(os.path.join(os.path.dirname(__file__), 'data', 'maude_decom.pcl'), 'rb') as f:
    test_data = pickle.load(f)


def test_msids():
    _ = maude_decom.AcaTelemetryMsidList(1)


def test_empty():
    msids = maude_decom.AcaTelemetryMsidList(1)
    data = test_data['176267186-176267186']
    table = maude_decom.assemble(msids, data, full=True, calibrate=False)
    assert len(table) == 0


def test_assembly():
    msids = maude_decom.AcaTelemetryMsidList(1)

    data = test_data['686111007-686111017']
    r_full = maude_decom.assemble(msids, data, full=True)
    assert [len(r_full[r_full['imgnum'] == i]) for i in range(8)] == [1, 1, 1, 4, 4, 4, 4, 4]
    assert np.all(r_full[r_full['imgnum'] == 3]['size'].data == ['6X6', '6X6', '6X6', '6X6'])

    r = maude_decom.assemble(msids, data)
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
    msids = maude_decom.AcaTelemetryMsidList(1)
    data = test_data['686111007-686111017']
    table = maude_decom.assemble(msids, data, full=True, calibrate=False)
    table = table[table['imgnum'] == 4]
    img_ref = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, 60.00, 76.00, 85.00, 82.00, np.nan, np.nan],
               [np.nan, 76.00, 109.00, 217.00, 203.00, 109.00, 95.00, np.nan],
               [np.nan, 120.00, 302.00, 863.00, 914.00, 475.00, 253.00, np.nan],
               [np.nan, 117.00, 685.00, 638.00, 599.00, 742.00, 227.00, np.nan],
               [np.nan, 83.00, 277.00, 372.00, 645.00, 392.00, 98.00, np.nan],
               [np.nan, np.nan, 98.00, 142.00, 351.00, 88.00, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    assert np.all((table[0]['img'] == img_ref) + (np.isnan(img_ref)))

    table = maude_decom.assemble(msids, data, full=True, calibrate=True)
    table = table[table['imgnum'] == 4]
    img_ref = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
               [np.nan, np.nan, 10.00, 26.00, 35.00, 32.00, np.nan, np.nan],
               [np.nan, 26.00, 59.00, 167.00, 153.00, 59.00, 45.00, np.nan],
               [np.nan, 70.00, 252.00, 813.00, 864.00, 425.00, 203.00, np.nan],
               [np.nan, 67.00, 635.00, 588.00, 549.00, 692.00, 177.00, np.nan],
               [np.nan, 33.00, 227.00, 322.00, 595.00, 342.00, 48.00, np.nan],
               [np.nan, np.nan, 48.00, 92.00, 301.00, 38.00, np.nan, np.nan],
               [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    assert np.all((table[0]['img'] == img_ref) + (np.isnan(img_ref)))


if __name__ == '__main__':
    test_msids()
