"""
Classes and functions to help fetching ACA telemetry data using Maude.
"""

import numpy as np

from astropy.table import Table, vstack
import maude

from Chandra.Time import DateTime

PIXEL_MAP = {
    '4x4': np.array([['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
                     ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
                     ['  ', '  ', 'A1', 'B1', 'C1', 'D1', '  ', '  '],
                     ['  ', '  ', 'E1', 'F1', 'G1', 'H1', '  ', '  '],
                     ['  ', '  ', 'I1', 'J1', 'K1', 'L1', '  ', '  '],
                     ['  ', '  ', 'M1', 'N1', 'O1', 'P1', '  ', '  '],
                     ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
                     ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ']]),
    '6x6': np.array([['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  '],
                     ['  ', '  ', 'A2', 'B2', 'C2', 'D2', '  ', '  '],
                     ['  ', 'P2', 'A1', 'B1', 'C1', 'D1', 'E2', '  '],
                     ['  ', 'O2', 'E1', 'F1', 'G1', 'H1', 'F2', '  '],
                     ['  ', 'N2', 'I1', 'J1', 'K1', 'L1', 'G2', '  '],
                     ['  ', 'M2', 'M1', 'N1', 'O1', 'P1', 'H2', '  '],
                     ['  ', '  ', 'L2', 'K2', 'J2', 'I2', '  ', '  '],
                     ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ']]),
    '8x8': np.array([['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1'],
                     ['I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1'],
                     ['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2'],
                     ['I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'O2', 'P2'],
                     ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3'],
                     ['I3', 'J3', 'K3', 'L3', 'M3', 'N3', 'O3', 'P3'],
                     ['A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4'],
                     ['I4', 'J4', 'K4', 'L4', 'M4', 'N4', 'O4', 'P4']])
}

PIXEL_MASK = {k: PIXEL_MAP[k] == '  ' for k in PIXEL_MAP}
ROWS, COLS = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')

PIXEL_MAP_INV = {k: {p: (i, j) for i, j, p in zip(ROWS[PIXEL_MAP[k] != '  '],
                                                  COLS[PIXEL_MAP[k] != '  '],
                                                  PIXEL_MAP[k][PIXEL_MAP[k] != '  '])}
                 for k in ['6x6', '4x4', '8x8']}


_msid_prefix = {
    1: 'A',
    2: 'R'
}


def _aca_msid_list(pea):
    return {
        'command_count': f'{_msid_prefix[pea]}CCMDS',
        'integration_time': f'{_msid_prefix[pea]}ACAINT0'
    }


def _aca_image_msid_list(pea):
    msid_prefix = _msid_prefix[pea]

    px_ids = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'}
    px_nums = [str(n) for n in range(1, 5)]
    px_img_nums = [str(n) for n in range(8)]

    pixels = [[f'{msid_prefix}CIMG{px_img_num}{px_id}{px_num}' for px_num in px_nums for px_id in px_ids]
              for px_img_num in px_img_nums]

    res = {
        'pixels': pixels,
        'sizes': [f'{msid_prefix}CA00040',   # Size of image 0
                  f'{msid_prefix}CA00043',   # Size of image 1
                  f'{msid_prefix}CA00046',   # Size of image 2
                  f'{msid_prefix}CA00049',   # Size of image 3
                  f'{msid_prefix}CA00052',   # Size of image 4
                  f'{msid_prefix}CA00055',   # Size of image 5
                  f'{msid_prefix}CA00058',   # Size of image 6
                  f'{msid_prefix}CA00061'],   # Size of image 7

        'rows': [f'{msid_prefix}CA00076',   # Row of pixel A1 of image 0
                 f'{msid_prefix}CA00292',   # Row of pixel A1 of image 1
                 f'{msid_prefix}CA00508',   # Row of pixel A1 of image 2
                 f'{msid_prefix}CA00724',   # Row of pixel A1 of image 3
                 f'{msid_prefix}CA00940',   # Row of pixel A1 of image 4
                 f'{msid_prefix}CA01156',   # Row of pixel A1 of image 5
                 f'{msid_prefix}CA01372',   # Row of pixel A1 of image 6
                 f'{msid_prefix}CA01588'],   # Row of pixel A1 of image 7

        'cols': [f'{msid_prefix}CA00086',   # Column of pixel A1 of image 0
                 f'{msid_prefix}CA00302',   # Column of pixel A1 of image 1
                 f'{msid_prefix}CA00518',   # Column of pixel A1 of image 2
                 f'{msid_prefix}CA00734',   # Column of pixel A1 of image 3
                 f'{msid_prefix}CA00950',   # Column of pixel A1 of image 4
                 f'{msid_prefix}CA01166',   # Column of pixel A1 of image 5
                 f'{msid_prefix}CA01382',   # Column of pixel A1 of image 6
                 f'{msid_prefix}CA01598'],  # Column of pixel A1 of image 7

        'scale_factor': [f'{msid_prefix}CA00096',   # Scale factor of image 0
                         f'{msid_prefix}CA00312',   # Scale factor of image 1
                         f'{msid_prefix}CA00528',   # Scale factor of image 2
                         f'{msid_prefix}CA00744',   # Scale factor of image 3
                         f'{msid_prefix}CA00960',   # Scale factor of image 4
                         f'{msid_prefix}CA01176',   # Scale factor of image 5
                         f'{msid_prefix}CA01392',   # Scale factor of image 6
                         f'{msid_prefix}CA01608'],  # Scale factor of image 7

        'background_rms': [f'{msid_prefix}CRMSBG{i}' for i in range(8)],
        'background_avg': [f'{msid_prefix}CA00110', f'{msid_prefix}CA00326',
                           f'{msid_prefix}CA00542', f'{msid_prefix}CA00758',
                           f'{msid_prefix}CA00974', f'{msid_prefix}CA01190',
                           f'{msid_prefix}CA01406', f'{msid_prefix}CA01622'],
        'housing_temperature': [f'{msid_prefix}ACH1T{i}2' for i in range(8)],  # AC HOUSING TEMPERATURE
        'ccd_temperature': [f'{msid_prefix}CCDPT{i}2' for i in range(8)],  # CCD TEMPERATURE
        'primary_temperature': [f'{msid_prefix}QTAPMT{i}' for i in range(8)],  # PRIMARY MIRROR/LENS CELL TEMP
        'secondary_temperature': [f'{msid_prefix}QTH2MT{i}' for i in range(8)]  # AC SECONDARY MIRROR TEMPERATURE
    }
    return [{k: res[k][i] for k in res.keys()} for i in range(8)]


ACA_MSID_LIST = {i+1: _aca_msid_list(i+1) for i in range(2)}
ACA_SLOT_MSID_LIST = {i+1: _aca_image_msid_list(i+1) for i in range(2)}


def assemble_image(pixel_data, img_size):
    """
    Assemble ACA images from a collection of MSID data.

    This function takes pixel data values/times in the form of arrays, one for each MSID.
    It returns an array of shape (8,8,len(img_size)).

    Pixel MSIDs are mapped to array antries depending on the image size::

      - Size 4X41:

        -----------------------------------------
        | -- | -- | -- | -- | -- | -- | -- | -- |
        -----------------------------------------
        | -- | -- | -- | -- | -- | -- | -- | -- |
        -----------------------------------------
        | -- | -- | D1 | H1 | L1 | P1 | -- | -- |
        -----------------------------------------
        | -- | -- | C1 | G1 | K1 | O1 | -- | -- |
        -----------------------------------------
        | -- | -- | B1 | F1 | J1 | N1 | -- | -- |
        -----------------------------------------
        | -- | -- | A1 | E1 | I1 | M1 | -- | -- |
        -----------------------------------------
        | -- | -- | -- | -- | -- | -- | -- | -- |
        -----------------------------------------
        | -- | -- | -- | -- | -- | -- | -- | -- |
        -----------------------------------------

      - Size 6X61 or 6X62:

        -----------------------------------------
        | -- | -- | -- | -- | -- | -- | -- | -- |
        -----------------------------------------
        | -- | -- | E2 | F2 | G2 | H2 | -- | -- |
        -----------------------------------------
        | -- | D2 | D1 | H1 | L1 | P1 | I2 | -- |
        -----------------------------------------
        | -- | C2 | C1 | G1 | K1 | O1 | J2 | -- |
        -----------------------------------------
        | -- | B2 | B1 | F1 | J1 | N1 | K2 | -- |
        -----------------------------------------
        | -- | A2 | A1 | E1 | I1 | M1 | L2 | -- |
        -----------------------------------------
        | -- | -- | P2 | O2 | N2 | M2 | -- | -- |
        -----------------------------------------
        | -- | -- | -- | -- | -- | -- | -- | -- |
        -----------------------------------------


      - Size 8X81, 8X82, 8X83 or 8X84:

        -----------------------------------------
        | H1 | P1 | H2 | P2 | H3 | P3 | H4 | P4 |
        -----------------------------------------
        | G1 | O1 | G2 | O2 | G3 | O3 | G4 | O4 |
        -----------------------------------------
        | F1 | N1 | F2 | N2 | F3 | N3 | F4 | N4 |
        -----------------------------------------
        | E1 | M1 | E2 | M2 | E3 | M3 | E4 | M4 |
        -----------------------------------------
        | D1 | L1 | D2 | L2 | D3 | L3 | D4 | L4 |
        -----------------------------------------
        | C1 | K1 | C2 | K2 | C3 | K3 | C4 | K4 |
        -----------------------------------------
        | B1 | J1 | B2 | J2 | B3 | J3 | B4 | J4 |
        -----------------------------------------
        | A1 | I1 | A2 | I2 | A3 | I3 | A4 | I4 |
        -----------------------------------------

    NOTE: in the previous tables, the rows are numbered in _ascending_ order.
    If one prints the corresponding np.array they are printed in _descending_ order,
    while if you draw them using plt.pcolor they will be in ascending order.

    :param pixel_data: dictionary containing MSID values and times.

    The keys of ``pizel_data`` must be MSIDs.
    The values must be dictionaries with keys ['times', 'values']

    :param img_size: an array of image size specifications.

    Each entry in ``img_size`` must be one of: 4X41, 6X61, 6X62, 8X81, 8X82, 8X83 or 8X84.
    The size of ``img_size`` must be equal to the size of the last axis of ``pixel_data``.

    """
    if list(pixel_data.values())[0]['values'].shape[-1] != len(img_size):
        s1, s2 = list(pixel_data.values())[0]['values'].shape[-1], len(img_size)
        raise Exception(
            f'Pixel data shape ({s1},) and image size array shape ({s2},) do not agree.')

    img = np.ones((len(img_size), 8, 8)) * np.nan

    msid_img = list(set([k[:-2] for k in pixel_data.keys()]))
    assert len(msid_img) == 1
    msid_img = msid_img[0]

    entries = {
        '4x4': (img_size == '4X41'),
        '6x6': (img_size == '6X61') + (img_size == '6X62'),
        '8x8': ((img_size == '8X81') + (img_size == '8X82') +
                (img_size == '8X83') + (img_size == '8X84'))
    }

    for k, m in PIXEL_MAP_INV.items():
        for p, (i, j) in m.items():
            img[entries[k], i, j] = pixel_data[f'{msid_img}{p}']['values'][entries[k]]

    return img


def _subsets(l, n):
    # consecutive subsets of a list, each with at most n elements
    for i in range(0, len(l) + n, n):
        if l[i:i + n]:
            yield l[i:i + n]


def _reshape_values(data, tref):
    """
    This stores a data field coming from a maude query into a different data structure.

    The most important thing this does is to reshape each MSID values array so the number
    of samples is the same as the number of sample times in tref, with NAN values at
    times when there is no data for the MSID.
    """
    t = data['times']
    if t.shape[0] == tref.shape[0]:
        # image size values pass through here, and they are strings, not floats
        return {'times': t, 'values': np.array(data['values'])}

    v = np.ones(tref.shape) * np.nan
    if t.shape[0] != 0:
        dt = (t[np.newaxis] - tref[:, np.newaxis])
        dt = np.where(dt > 0, dt, np.inf)
        i = np.argmin(dt, axis=0)
        v[i] = data['values']

    return {'times': tref, 'values': v}


def combine_sub_images(table):
    """
    Take a table as input and combine consecutive image segments.
    Partial images are discarded.

    :param table: astropy.Table with ACA image telemetry data
    :return:
    """
    subimage = table['subimage']
    tref = table['time']
    # What follows is not trivial and needs attention.
    # If requesting full images, identify complete entries first.
    # We will return complete images at the time of the first partial image and discard the rest.
    # take all 4x41 images:
    ok_4x4 = (table['size'] == '4X41')
    # take all 6x61 images (subimage == 1) if the next image has subimage == 2:
    ok_6x6 = (table['size'] == '6X61') * \
        np.concatenate([subimage[1:] - subimage[:-1] == 1, [False]])
    # take all 8x81 images (subimage == 1) if the 3rd image after this has subimage == 4:
    ok_8x8 = (table['size'] == '8X81') * \
        np.concatenate([subimage[3:] - subimage[:-3] == 3, [False] * 3])

    # now add the partial images (with subimage > 1) to the first partial image
    # for 8x8:
    i = np.arange(len(tref))[ok_8x8]
    table['img'][i] = np.nansum([table['img'][i], table['img'][i + 1],
                                 table['img'][i + 2], table['img'][i + 3]],
                                axis=0)
    # and for some reason I also had to do this:
    for k in ['row0', 'col0', 'scale_factor']:
        table[k][i] = np.nansum([table[k][i], table[k][i + 1], table[k][i + 2], table[k][i + 3]],
                                axis=0)

    # for 6x6:
    i = np.arange(len(tref))[ok_6x6]
    tmp = np.nansum([table['img'][i], table['img'][i + 1]], axis=0)
    tmp[:, PIXEL_MASK['6x6']] = np.nan
    table['img'][i] = tmp
    # and for some reason I also had to do this:
    for k in ['row0', 'col0', 'scale_factor']:
        table[k][i] = np.nansum([table[k][i], table[k][i + 1]], axis=0)

    # now actually discard partial images
    # (only if discarded in all slots)
    ok = ok_4x4 + ok_6x6 + ok_8x8
    table = {k: v[ok] for k, v in table.items()}

    table['size'][table['size'] == '4X41'] = '4X4'
    table['size'][table['size'] == '6X61'] = '6X6'
    table['size'][table['size'] == '8X81'] = '8X8'

    return table


def _assemble_img(slot, pea, data, full=False,
                  calibrate=False, adjust_time=False, adjust_corner=False):
    """
    This method assembles an astropy.Table for a given PEA and slot.

    :param slot: integer in range(8)
    :param pea_choice: integer 1 or 2
    :param data: dictionary with maude data.

    Each entry in the dictionary is a values returned by maude.get_msids.
    Something like this:

        >>> import maude
        >>> start, stop = 686111007, 686111017
        >>> data = {e['msid']:e for e in maude.get_msids([...], start=start, stop=stop)['data']}

    :param full: bool. Combine partial image segments into full images
    :param calibrate: bool. Scale image values.
    :param adjust_time: bool. Correct times the way it is done in level 0.
    :param adjust_corner: bool. Shift col0 and row0 the way it is done in level 0.
    """

    msids = ACA_MSID_LIST[pea]
    slot_msids = ACA_SLOT_MSID_LIST[pea][slot]

    tref = data[slot_msids['sizes']]['times']

    # reshape all values using the times from an MSID we know will be there at all sample times:
    data = {k: _reshape_values(data[k], tref) for k in data}

    if len(tref) == 0:
        names = ['time', 'imgnum', 'size', 'row0', 'col0', 'scale_factor', 'integ', 'img']
        dtype = ['<f8', '<i8', '<U4', '<f8', '<f8', '<f8', '<f8', ('<f8', (8, 8))]
        result = {n: np.array([], dtype=t) for n, t in zip(names, dtype)}
    else:
        pixel_data = {k: data[k] for k in slot_msids['pixels']}
        img_size = data[slot_msids['sizes']]['values']
        image = assemble_image(pixel_data, img_size)
        # there must be an MSID to fetch this, but this works
        subimage = np.char.replace(np.char.replace(np.char.replace(
            img_size, '8X8', ''), '6X6', ''), '4X4', '').astype(int) - 1

        result = {
            'time': data[slot_msids['sizes']]['times'],
            'imgnum': np.ones(len(tref), dtype='<i8') * slot,
            'subimage': subimage,
            'size': data[slot_msids['sizes']]['values'],
            'row0': data[slot_msids['rows']]['values'],
            'col0': data[slot_msids['cols']]['values'],
            'scale_factor': data[slot_msids['scale_factor']]['values'],
            'integ': 0.016 * data[msids['integration_time']]['values'],
            'img': image
        }

        if adjust_time:
            result['time'] -= (result['integ'] / 2 + 1.025)

        if adjust_corner:
            result['row0'][result['size'] == '6X61'] -= 1
            result['col0'][result['size'] == '6X61'] -= 1

        if calibrate:
            # scale specified in ACA L0 ICD, section D.2.2 (scale_factor is already divided by 32)
            #
            # for an incomplete images the result can look like this:
            #     size scale_factor
            #     str4   float64
            #     ---- ------------
            #     8X84          nan
            #     8X81          1.0
            #     8X82          nan
            #     8X83          nan
            #     8X84          nan
            #     8X81          1.0
            #     8X82          nan
            #     8X83          nan
            #     8X84          nan
            #     8X81          1.0
            #
            # so one has to set the proper scale for images 6X62, 8X82, 8X83 and 8X84.
            # the data should cover a larger interval than requested to avoid edge effects.
            # I also do not want to change the original array
            # this assumes that image sizes ALWAYS alternate
            # is there a better way to do this?
            scale = np.array(result['scale_factor'])
            for name, n in [('6X61', 2), ('8X81', 4)]:
                s1 = (result['size'] == name)
                for i in range(1,n):
                    s2 = np.roll(s1, i)  # roll and drop the ones that go over the edge
                    s2[:i] = False
                    scale[s2] = scale[s1][:sum(s2)]
            result['img'] *= scale[:, np.newaxis, np.newaxis]
            result['img'] -= 50

        if full:
            result = combine_sub_images(result)
            del result['subimage']


    return Table(result)


def fetch(start, stop, slots=range(8), pea=1, full=False,
          calibrate=False, adjust_time=False, adjust_corner=False):
    """
    This is an example of fetching and assembling data using maude.

    Example usage::

      >>> from chandra_aca import maude_decom
      >>> data = maude_decom.fetch(start, stop, 1)

    It will be changed once we know::

      - what other telemetry to include
      - what structure should the data be in the viewer

    :param start: timestamp
    :param stop: timestamp
    :param pea_choice: integer 1 or 2

    Each entry in the dictionary is a values returned by maude.get_msids.
    Something like this:

        >>> import maude
        >>> start, stop = 686111007, 686111017
        >>> data = {e['msid']:e for e in maude.get_msids([...], start=start, stop=stop)['data']}

    :param full: bool. Combine partial image segments into full images
    :param calibrate: bool. Scale image values.
    :param adjust_time: bool. Correct times the way it is done in level 0.
    :param adjust_corner: bool. Shift col0 and row0 the way it is done in level 0.
    """

    start, stop = DateTime(start), DateTime(stop)
    if calibrate or adjust_time:
        start -= 6 / 86400  # padding at the beginning in case of time/scale adjustments
    if full:
        stop += 6 / 86400  # padding at the end in case of trailing partial images

    tables = []
    for slot in slots:
        msids = ['sizes', 'rows', 'cols', 'scale_factor']  # the MSIDs we fetch (plus IMG pixels)
        msids = (ACA_SLOT_MSID_LIST[pea][slot]['pixels'] +
                 [ACA_SLOT_MSID_LIST[pea][slot][k] for k in msids] +
                 [ACA_MSID_LIST[pea]['integration_time']])
        res = {e['msid']: e for e in maude.get_msids(msids, start=start, stop=stop)['data']}
        tables.append(_assemble_img(slot, pea, res, full=full, calibrate=calibrate,
                                    adjust_time=adjust_time, adjust_corner=adjust_corner))
    result = vstack(tables)
    # and chop the padding we added above
    result = result[(result['time'] >= start.secs)*(result['time'] <= stop.secs)]
    return result
