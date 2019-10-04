#!/usr/bin/env python3

import os
import pickle
import numpy as np

from chandra_aca import maude_decom


with open(os.path.join(os.path.dirname(__file__), 'data', 'maude_decom.pcl'), 'rb') as f:
    test_data = pickle.load(f)


def test_msids():
    msids = maude_decom.AcaTelemetryMsidList(1)

    data = test_data['686111007-686111017']['data']
    r_full = maude_decom.assemble(msids, data, full=True)
    assert [len(r_full[r_full['imgnum'] == i]) for i in range(8)] == [1, 1, 1, 4, 4, 4, 4, 4]
    assert np.all(r_full[r_full['imgnum'] == 3]['size'].data == ['6X61', '6X61', '6X61', '6X61'])

    r = maude_decom.assemble(msids, data)
    assert [len(r[r['imgnum'] == i]) for i in range(8)] == [10, 10, 10, 10, 10, 10, 10, 10]
    assert np.all(r[r['imgnum'] == 3]['size'].data ==
                  ['6X62', '6X61', '6X62', '6X61', '6X62', '6X61', '6X62', '6X61', '6X62', '6X61'])

    eq = (r_full[r_full['imgnum'] == 0]['img'].data ==
          np.nansum(r[r['imgnum'] == 0]['img'].data[3:7].reshape((-1, 4, 8, 8)), axis=1))
    assert np.all(eq)

    eq = (r_full[r_full['imgnum'] == 3]['img'].data ==
          np.nansum(r[r['imgnum'] == 3]['img'].data[1:-1].reshape((-1, 2, 8, 8)), axis=1))
    assert np.all(eq.flatten()[~maude_decom.PIXEL_MASK['6x6'].flatten()])


if __name__ == '__main__':
    test_msids()
