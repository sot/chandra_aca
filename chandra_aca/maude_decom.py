"""
Classes and functions to help fetching ACA telemetry data using Maude.
"""

from struct import unpack, Struct
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
    # helper method to make a dictionary with all global (non-slot) MSIDs used here
    return {
        'status': 'AOACSTAT',  # ASPECT CAMERA DATA PROCESSING OVERALL STATUS FLAG
        'command_count': f'{_msid_prefix[pea]}CCMDS',
        'integration_time': f'{_msid_prefix[pea]}ACAINT0',
        'major_frame': 'CVCMJCTR',
        'minor_frame': 'CVCMNCTR',
        'cmd_count': f'{_msid_prefix[pea]}CCMDS',
        'cmd_progress': f'{_msid_prefix[pea]}AROW2GO'  # NUMBER OF ROWS TO GO COMMAND PROGRESS

    }


def _aca_image_msid_list(pea):
    # helper method to make a list of dictionaries with MSIDs for each slot in a given PEA.
    msid_prefix = _msid_prefix[pea]

    px_ids = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'}
    px_nums = [str(n) for n in range(1, 5)]
    px_img_nums = [str(n) for n in range(8)]

    pixels = [[f'{msid_prefix}CIMG{px_img_num}{px_id}{px_num}'
               for px_num in px_nums for px_id in px_ids]
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

        'image_status': [f'AOIMAGE{i}' for i in range(8)],    # IMAGE 0 STATUS FLAG
        'fiducial_flag': [f'AOACFID0{i}' for i in range(8)],  # FIDUCIAL LIGHT FLAG (OBC)
        'image_function': [f'AOACFCT{i}' for i in range(8)],  # IMAGE FUNCTION (OBC)
        # this one exists also as FUNCTION2/3/4
        # 'image_function_pea':
        #     [f'{msid_prefix}AIMGF{i}1' for i in range(8)],  # IMAGE FUNCTION1 (PEA)

        'background_rms': [f'{msid_prefix}CRMSBG{i}' for i in range(8)],
        'background_avg': [f'{msid_prefix}CA00110', f'{msid_prefix}CA00326',
                           f'{msid_prefix}CA00542', f'{msid_prefix}CA00758',
                           f'{msid_prefix}CA00974', f'{msid_prefix}CA01190',
                           f'{msid_prefix}CA01406', f'{msid_prefix}CA01622'],
        'housing_temperature':
            [f'{msid_prefix}ACH1T{i}2' for i in range(8)],                 # AC HOUSING TEMPERATURE
        'ccd_temperature': [f'{msid_prefix}CCDPT{i}2' for i in range(8)],  # CCD TEMPERATURE
        'primary_temperature':
            [f'{msid_prefix}QTAPMT{i}' for i in range(8)],  # PRIMARY MIRROR/LENS CELL TEMP
        'secondary_temperature':
            [f'{msid_prefix}QTH2MT{i}' for i in range(8)],  # AC SECONDARY MIRROR TEMPERATURE

        'magnitude': [f'AOACMAG{i}' for i in range(8)],       # STAR OR FIDUCIAL MAGNITUDE (OBC)
        'centroid_ang_y': [f'AOACYAN{i}' for i in range(8)],  # YAG CENTROID Y ANGLE (OBC)
        'centroid_ang_z': [f'AOACZAN{i}' for i in range(8)],  # ZAG CENTROID Z ANGLE (OBC)
    }
    return [{k: res[k][i] for k in res.keys()} for i in range(8)]


ACA_MSID_LIST = {i + 1: _aca_msid_list(i + 1) for i in range(2)}
ACA_SLOT_MSID_LIST = {i + 1: _aca_image_msid_list(i + 1) for i in range(2)}


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

    The keys of ``pixel_data`` must be MSIDs.
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
        dt = np.where(dt >= 0, dt, np.inf)
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
    tref = table['TIME']
    # What follows is not trivial and needs attention.
    # If requesting full images, identify complete entries first.
    # We will return complete images at the time of the first partial image and discard the rest.
    # take all 4x41 images:
    ok_4x4 = (table['IMGSIZE'] == '4X41')
    # take all 6x61 images (subimage == 1) if the next image has subimage == 2:
    ok_6x6 = (table['IMGSIZE'] == '6X61') * \
        np.concatenate([subimage[1:] - subimage[:-1] == 1, [False]])
    # take all 8x81 images (subimage == 1) if the 3rd image after this has subimage == 4:
    ok_8x8 = (table['IMGSIZE'] == '8X81') * \
        np.concatenate([subimage[3:] - subimage[:-3] == 3, [False] * 3])

    # now add the partial images (with subimage > 1) to the first partial image
    # for 8x8:
    i = np.arange(len(tref))[ok_8x8]
    table['IMG'][i] = np.nansum([table['IMG'][i], table['IMG'][i + 1],
                                 table['IMG'][i + 2], table['IMG'][i + 3]],
                                axis=0)
    # and for some reason I also had to do this:
    for k in ['IMGROW0', 'IMGCOL0', 'SCALE_FACTOR']:
        table[k][i] = np.nansum([table[k][i], table[k][i + 1], table[k][i + 2], table[k][i + 3]],
                                axis=0)

    # for 6x6:
    i = np.arange(len(tref))[ok_6x6]
    tmp = np.nansum([table['IMG'][i], table['IMG'][i + 1]], axis=0)
    tmp[:, PIXEL_MASK['6x6']] = np.nan
    table['IMG'][i] = tmp
    # and for some reason I also had to do this:
    for k in ['IMGROW0', 'IMGCOL0', 'SCALE_FACTOR']:
        table[k][i] = np.nansum([table[k][i], table[k][i + 1]], axis=0)

    # now actually discard partial images
    # (only if discarded in all slots)
    ok = ok_4x4 + ok_6x6 + ok_8x8
    table = {k: v[ok] for k, v in table.items()}

    table['IMGSIZE'][table['IMGSIZE'] == '4X41'] = '4X4'
    table['IMGSIZE'][table['IMGSIZE'] == '6X61'] = '6X6'
    table['IMGSIZE'][table['IMGSIZE'] == '8X81'] = '8X8'

    return table


def _assemble_img(slot, pea, data, full=False,
                  calibrate=False, adjust_time=False, adjust_corner=False):
    """
    This method assembles an astropy.Table for a given PEA and slot.

    :param slot: integer in range(8)
    :param pea: integer 1 or 2
    :param data: dictionary with maude data.

    Each entry in the dictionary is a value returned by maude.get_msids.
    Something like this:

        >>> import maude
        >>> start, stop = 686111007, 686111017
        >>> data = {e['msid']:e for e in maude.get_msids([...], start=start, stop=stop)['data']}

    :param full: bool. Combine partial image segments into full images
    :param calibrate: bool. Scale image values (ignored if full=False).
    :param adjust_time: bool. Correct times the way it is done in level 0.
    :param adjust_corner: bool. Shift IMGCOL0 and IMGROW0 the way it is done in level 0.
    """

    msids = ACA_MSID_LIST[pea]
    slot_msids = ACA_SLOT_MSID_LIST[pea][slot]

    tref = data[slot_msids['sizes']]['times']

    # reshape all values using the times from an MSID we know will be there at all sample times:
    data = {k: _reshape_values(data[k], tref) for k in data}

    if len(tref) == 0:
        names = ['TIME', 'IMGNUM', 'IMGSIZE', 'IMGROW0', 'IMGCOL0', 'SCALE_FACTOR', 'INTEG', 'IMG']
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
            'TIME': data[slot_msids['sizes']]['times'],
            'IMGNUM': np.ones(len(tref), dtype='<i8') * slot,
            'subimage': subimage,
            'IMGSIZE': data[slot_msids['sizes']]['values'],
            'IMGROW0': data[slot_msids['rows']]['values'],
            'IMGCOL0': data[slot_msids['cols']]['values'],
            'SCALE_FACTOR': data[slot_msids['scale_factor']]['values'],
            'INTEG': 0.016 * data[msids['integration_time']]['values'],
            'IMG': image
        }

        if adjust_time:
            result['TIME'] -= (result['INTEG'] / 2 + 1.025)
            # result['END_INTEG_TIME'] = result['TIME'] + result['INTEG']

        if adjust_corner:
            result['IMGROW0'][result['IMGSIZE'] == '6X61'] -= 1
            result['IMGCOL0'][result['IMGSIZE'] == '6X61'] -= 1

        if calibrate:
            # scale specified in ACA L0 ICD, section D.2.2 (scale_factor is already divided by 32)
            #
            # for an incomplete images the result can look like this:
            #     IMGSIZE SCALE_FACTOR
            #      str4     float64
            #     ------- ------------
            #      8X84       nan
            #      8X81       1.0
            #      8X82       nan
            #      8X83       nan
            #      8X84       nan
            #      8X81       1.0
            #      8X82       nan
            #      8X83       nan
            #      8X84       nan
            #      8X81       1.0
            #
            # so one has to set the proper scale for images 6X62, 8X82, 8X83 and 8X84.
            # the data should cover a larger interval than requested to avoid edge effects.
            # I also do not want to change the original array
            # this assumes that image sizes ALWAYS alternate
            # is there a better way to do this?
            scale = np.array(result['SCALE_FACTOR'])
            for name, n in [('6X61', 2), ('8X81', 4)]:
                s1 = (result['IMGSIZE'] == name)
                for i in range(1, n):
                    s2 = np.roll(s1, i)  # roll and drop the ones that go over the edge
                    s2[:i] = False
                    scale[s2] = scale[s1][:sum(s2)]
            result['IMG'] *= scale[:, np.newaxis, np.newaxis]
            result['IMG'] -= 50

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
      >>> data = maude_decom.fetch(start, stop, pea=1)

    :param start: timestamp interpreted as a Chandra.Time.DateTime
    :param stop: timestamp interpreted as a Chandra.Time.DateTime
    :param slots: iterable of ints. Default: range(8)
    :param pea: integer 1 or 2. Default: 1
    :param full: bool. Combine partial image segments into full images.
    :param calibrate: bool. Scale image values.
    :param adjust_time: bool. Correct times the way it is done in level 0.
    :param adjust_corner: bool. Shift IMGCOL0 and IMGROW0 the way it is done in level 0.
    """

    start, stop = DateTime(start), DateTime(stop)
    start_pad = 0
    stop_pad = 0
    if calibrate or adjust_time:
        start_pad = 6 / 86400  # padding at the beginning in case of time/scale adjustments
    if full:
        stop_pad = 6 / 86400  # padding at the end in case of trailing partial images

    tables = []
    for slot in slots:
        msids = ['sizes', 'rows', 'cols', 'scale_factor']  # the MSIDs we fetch (plus IMG pixels)
        msids = (ACA_SLOT_MSID_LIST[pea][slot]['pixels'] +
                 [ACA_SLOT_MSID_LIST[pea][slot][k] for k in msids] +
                 [ACA_MSID_LIST[pea]['integration_time']])
        res = {e['msid']: e for e in
               maude.get_msids(msids, start=start - start_pad, stop=stop + stop_pad)['data']}
        tables.append(_assemble_img(slot, pea, res, full=full, calibrate=calibrate,
                                    adjust_time=adjust_time, adjust_corner=adjust_corner))
    result = vstack(tables)
    # and chop the padding we added above
    result = result[(result['TIME'] >= start.secs) * (result['TIME'] <= stop.secs)]
    return result


_a2p = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
indices = [
    np.array([PIXEL_MAP_INV['4x4'][f'{k}1'] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV['6x6'][f'{k}1'] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV['6x6'][f'{k}2'] for k in _a2p]).T,
    [],
    np.array([PIXEL_MAP_INV['8x8'][f'{k}1'] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV['8x8'][f'{k}2'] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV['8x8'][f'{k}3'] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV['8x8'][f'{k}4'] for k in _a2p]).T
]

######################
# VCDU-based functions
######################

# these are used multiple times
_aca_front_fmt = Struct('>HBBBBBB')
_size_bits = np.zeros((8, 8), dtype=np.uint8)
_pixel_bits = np.zeros((16, 16), dtype=np.uint16)
_bits = np.array([1 << i for i in range(64)], dtype=np.uint64)[::-1]


# I'm sure there is a better way...
def _packbits(a, unsigned=True):
    # take something like this: [1,0,1,1,0] and return 2^4 + 2^2 + 2
    # This handles integer types only
    n = len(a)
    if not unsigned and a[0]:
        return np.int64(np.sum(a * _bits[-n:]) - (1 << n))
    return np.sum(a * _bits[-n:])


def _aca_header_1(bits):
    """
    Unpack ACA header 1 (ACA User Manual 5.3.2.2.1).

    :param bits: bytes-like object of length 7
    :return: dict
    """
    bits = np.unpackbits(np.array(unpack('BBBbbBB', bits), dtype=np.uint8))
    return {
        'fid': bool(bits[0]),
        'IMGNUM': _packbits(bits[1:4]),
        'IMGFUNC': _packbits(bits[4:6]),
        'sat_pixel': bool(bits[6]),
        'def_pixel': bool(bits[7]),
        'IMGROW0': _packbits(bits[12:22], unsigned=False),
        'IMGCOL0': _packbits(bits[22:32], unsigned=False),
        'IMGSCALE': _packbits(bits[32:46]),
        'BGDAVG': _packbits(bits[46:56])
    }


def _aca_header_2(bits):
    """
    Unpack ACA header 2 (ACA User Manual 5.3.2.2.2).

    :param bits: bytes-like object of length 7
    :return: dict
    """
    bits = unpack('BbbbbBB', bits)
    c = np.unpackbits(np.array(bits[:2], dtype=np.uint8))
    return {
        'BGDRMS': _packbits(c[6:16]),
        'TEMPCCD': bits[2],
        'TEMPHOUS': bits[3],
        'TEMPPRIM': bits[4],
        'TEMPSEC': bits[5],
        'BGDSTAT': bits[6],
        'bkg_pixel_status': np.unpackbits(np.array(bits[:2], dtype=np.uint8)[-1:])
    }


def _aca_header_3(bits):
    """
    Unpack ACA header 3 (ACA User Manual 5.3.2.2.3).

    :param bits: bytes-like object of length 7
    :return: dict
    """
    return {
        f'DIAGNOSTIC': unpack('BBBBBB', bits[1:])
    }


# all headers for each kind of image
ACA_HEADER = [
    _aca_header_1,
    _aca_header_1,
    _aca_header_2,
    None,
    _aca_header_1,
    _aca_header_2,
    _aca_header_3,
    _aca_header_3
]


def unpack_aca_telemetry(a):
    """
    Unpack ACA telemetry encoded in 225-byte packets.

    :param a:
    :return: list of list of dict

    A list of frames, each with a list of slots, each being a dictionary.
    """
    integ, glbstat, commcnt, commprog, s1, s2, s3 = _aca_front_fmt.unpack(a[:8])
    _size_bits[:, -3:] = np.unpackbits(np.array([[s1, s2, s3]], dtype=np.uint8).T, axis=1).reshape(
        (8, -1))
    img_types = np.packbits(_size_bits, axis=1).T[0]
    slots = []
    for img_num, i in enumerate(range(8, len(a), 27)):
        img_header = ACA_HEADER[img_types[img_num]](a[i:i + 7])
        img_pixels = unpack('B' * 20, a[i + 7:i + 27])
        _pixel_bits[:, -10:] = np.unpackbits(np.array([img_pixels], dtype=np.uint8).T,
                                             axis=1).reshape((-1, 10))
        img_pixels = np.sum(np.packbits(_pixel_bits, axis=1) * [[2 ** 8, 1]], axis=1)
        img_header['pixels'] = img_pixels
        slots.append(img_header)
    res = {'INTEG': integ, 'GLBSTAT': glbstat, 'COMMCNT': commcnt, 'COMMPROG': commprog}
    for i, s in enumerate(slots):
        s.update(res)
        s['IMGTYPE'] = img_types[i]
    return slots


def combine_packets(aca_packets):
    """
    Combine a list of ACA packets into a single record.

    This is intended to combine the two 6X6 packets or the four 8X8 packets.

    :param aca_packets: list of dict
    :return: dict
    """
    # note that they are reverse-sorted so the first frame overwrites the others if they collide
    aca_packets = sorted(aca_packets, key=lambda p: p['TIME'], reverse=True)
    res = {}
    pixels = np.ma.masked_all((8, 8))
    pixels.data[:] = np.nan
    for f in aca_packets:
        pixels[indices[f['IMGTYPE']][0], indices[f['IMGTYPE']][1]] = f['pixels']

    for f in aca_packets:
        res.update(f)
    del res['pixels']
    res['IMG'] = pixels
    return res


def group_packets(packets, discard=True):
    res = []
    n = None
    s = None
    for packet in packets:
        if res and (packet['MJF'] * 128 + packet['MNF'] > n):
            if not discard or len(res) == s:
                yield res
            res = []
        if not res:
            # the number of minor frames expected within the same ACA packet
            s = {0: 1, 1: 2, 2: 2, 4: 4, 5: 4, 6: 4, 7: 4}[packet['IMGTYPE']]
            # the number of minor frames within the same ACA packet expected after this minor frame
            remaining = {0: 0, 1: 1, 2: 0, 4: 3, 5: 2, 6: 1, 7: 0}[packet['IMGTYPE']]
            n = packet['MJF'] * 128 + packet['MNF'] + 4 * remaining
        res.append(packet)
    if res and (not discard or len(res) == s):
        yield res


def get_raw_aca_packets(start, stop):
    """
    Fetch 1025-byte VCDU frames using maude and extract a list of 225-byte ACA packets.

    If the first minor frame in a group of four ACA packets is within (start, stop),
    the three following minor frames are included if present.

    returns a dictionary with keys ['TIME', 'MNF', 'MJF', 'packets', 'flags'].
    These correspond to the minor frame time, minor frame count, major frame count,
    the list of packets, and flags returned by Maude respectively.

    :param start: timestamp interpreted as a Chandra.Time.DateTime
    :param stop: timestamp interpreted as a Chandra.Time.DateTime
    :return:
    """
    date_start, date_stop = DateTime(start), DateTime(stop)  # ensure input is proper date
    stop_pad = 1.5 / 86400  # padding at the end in case of trailing partial ACA packets

    # also getting major and minor frames to figure out which is the first ACA packet in a group
    vcdu_counter = maude.get_msids(['CVCMNCTR', 'CVCMJCTR'],
                                   start=date_start,
                                   stop=date_stop + stop_pad)

    sub = vcdu_counter['data'][0]['values'] % 4  # the minor frame index within each ACA update
    vcdu_times = vcdu_counter['data'][0]['times']

    major_counter = np.cumsum(sub == 0)  # this number is monotonically increasing starting at 0

    # only unpack complete ACA frames in the original range:
    aca_frame_entries = np.ma.masked_all((major_counter.max() + 1, 4), dtype=int)
    aca_frame_entries[major_counter, sub] = np.arange(vcdu_times.shape[0])
    aca_frame_times = np.ma.masked_all((major_counter.max() + 1, 4))
    aca_frame_times[major_counter, sub] = vcdu_times

    # this will remove ACA records with at least one missing minor frame
    select = ((~np.any(aca_frame_times.mask, axis=1)) *
              (aca_frame_times[:, 0] >= date_start.secs) *
              (aca_frame_times[:, 0] <= date_stop.secs))

    # get the frames and unpack front matter
    frames = maude.get_frames(start=date_start, stop=date_stop + stop_pad)
    rf, flags, nblobs = unpack('<bHI', frames[:7])
    assert nblobs == len(sub)  # this should never fail.

    # assemble the 56 bit ACA minor records and times (time is not unpacked)
    aca = []
    aca_times = []
    for i in range(7, len(frames), 1033):
        aca_times.append(frames[i:i + 8])
        aca.append(b''.join([frames[i + j + 8:i + j + 8 + 14] for j in [18, 274, 530, 780]]))
    assert len(aca) == nblobs  # this should never fail.

    # combine them into 224 byte frames (it currently ensures all 224 bytes are there)
    aca_packets = [b''.join([aca[i] for i in entry]) for entry in aca_frame_entries[select]]
    for a in aca_packets:
        assert (len(a) == 224)

    minor_counter = vcdu_counter['data'][0]['values'][aca_frame_entries[select, 0]]
    major_counter = vcdu_counter['data'][1]['values'][aca_frame_entries[select, 0]]
    times = vcdu_counter['data'][0]['times'][aca_frame_entries[select, 0]]

    return {'flags': flags, 'packets': aca_packets,
            'TIME': times, 'MNF': minor_counter, 'MJF': major_counter}


def aca_packets_to_table(aca_packets):
    """
    Store ACA packets in a table.

    :param aca_packets: list of dict
    :return: astropy.table.Table
    """
    import copy
    dtype = np.dtype(
        [('TIME', np.float64), ('MJF', np.uint32), ('MNF', np.uint32), ('IMGNUM', np.uint32),
         ('COMMCNT', np.uint8), ('COMMPROG', np.uint8), ('GLBSTAT', np.uint8),
         ('IMGFUNC', np.uint32), ('IMGTYPE', np.uint8), ('IMGSCALE', np.uint16),
         ('IMGROW0', np.int16), ('IMGCOL0', np.int16), ('INTEG', np.uint16),
         ('BGDAVG', np.uint16), ('BGDRMS', np.uint16), ('TEMPCCD', np.int16),
         ('TEMPHOUS', np.int16), ('TEMPPRIM', np.int16), ('TEMPSEC', np.int16),
         ('BGDSTAT', np.uint8)
         ])

    array = np.ma.masked_all(len(aca_packets), dtype=dtype)
    names = copy.deepcopy(dtype.names)
    pixels = []
    img = []
    for i, aca_packet in enumerate(aca_packets):
        if 'pixels' in aca_packet:
            pixels.append(aca_packet['pixels'])
        if 'IMG' in aca_packet:
            img.append(aca_packet['IMG'])
        for k in names:
            if k in aca_packet:
                array[i][k] = aca_packet[k]

    table = Table(array)
    if pixels:
        table['PIXELS'] = pixels
    if img:
        table['IMG'] = img
    return table


def get_aca_packets(start, stop, level0=False,
                    combine=False, adjust_time=False, calibrate_pixels=False,
                    adjust_corner=False, calibrate_temperatures=False):
    """
    Fetch VCDU 1025-byte frames, extract ACA packets, unpack them and store them in a table.

    Incomplete ACA packets (if there is a minor frame missing) are discarded.

    :param start: timestamp interpreted as a Chandra.Time.DateTime
    :param stop: timestamp interpreted as a Chandra.Time.DateTime
    :param level0: bool
    :param combine: bool
    :param adjust_time: bool
    :param calibrate_pixels: bool
    :param: adjust_corner: bool
    :return: astropy.table.Table
    """
    if level0:
        adjust_time = True
        combine = True
        calibrate_pixels = True
        adjust_corner = True
        calibrate_temperatures = True

    start, stop = DateTime(start), DateTime(stop)  # ensure input is proper date
    start_pad = 0
    stop_pad = 0
    if adjust_time:
        stop_pad += 2. / 86400  # time will get shifted...
    if combine:
        stop_pad += 3.08 / 86400  # there can be trailing frames

    aca_packets = get_raw_aca_packets(start - start_pad, stop + stop_pad)
    aca_packets['packets'] = [unpack_aca_telemetry(a) for a in aca_packets['packets']]
    for i in range(len(aca_packets['packets'])):
        for j in range(8):
            aca_packets['packets'][i][j]['TIME'] = aca_packets['TIME'][i]
            aca_packets['packets'][i][j]['MJF'] = aca_packets['MJF'][i]
            aca_packets['packets'][i][j]['MNF'] = aca_packets['MNF'][i]
            aca_packets['packets'][i][j]['IMGNUM'] = j

    aca_packets = [[f[i] for f in aca_packets['packets']] for i in range(8)]
    if combine:
        aca_packets = sum([[combine_packets(g) for g in group_packets(p)] for p in aca_packets], [])
    else:
        aca_packets = [row for slot in aca_packets for row in slot]
    table = aca_packets_to_table(aca_packets)

    if calibrate_temperatures:
        for k in ['TEMPCCD', 'TEMPSEC', 'TEMPHOUS', 'TEMPPRIM']:
            table[k] = 0.4 * table[k].astype(np.float32) + 273.15

    if adjust_corner:
        table['IMGROW0'][table['IMGTYPE'] == 1] -= 1
        table['IMGCOL0'][table['IMGTYPE'] == 1] -= 1
        table['IMGROW0'][table['IMGTYPE'] == 2] -= 1
        table['IMGCOL0'][table['IMGTYPE'] == 2] -= 1

    if adjust_time:
        table['INTEG'] = table['INTEG'] * 0.016
        table['TIME'] -= table['INTEG'] / 2 + 1.025
        table['END_INTEG_TIME'] = table['TIME'] + table['INTEG'] / 2
        table = table[(table['TIME'] >= start.secs) * (table['TIME'] <= stop.secs)]

    if calibrate_pixels:
        if 'PIXELS' in table.colnames:
            table['PIXELS'] = table['PIXELS'] * table['IMGSCALE'][:, np.newaxis] / 32 - 50
        if 'IMG' in table.colnames:
            table['IMG'] = table['IMG'] * table['IMGSCALE'][:, np.newaxis, np.newaxis] / 32 - 50

    return table
