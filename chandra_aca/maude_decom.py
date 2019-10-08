"""
Classes and functions to help fetching ACA telemetry data using Maude.
"""

import numpy as np

from astropy.table import Table, vstack
import maude

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


class AcaTelemetryMsidList(list):
    """
    List of MSIDs required to assemble ACA telemetry data.
    """

    def __init__(self, pea_choice=1):
        super(AcaTelemetryMsidList, self).__init__()
        if pea_choice == 1:
            msid_prefix = 'A'
        elif pea_choice == 2:
            msid_prefix = 'R'
        else:
            raise Exception(f'Invalid PEA choice {pea_choice}')

        # This msid is not stored, it is just used for retrieving data at consistent times (?)
        primary_msid = f'{msid_prefix}CCMDS'

        integration_time = f'{msid_prefix}ACAINT0'

        px_msid_prefix = f'{msid_prefix}CIMG'
        px_ids = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P'}
        px_nums = [str(n) for n in range(1, 5)]
        px_img_nums = [str(n) for n in range(8)]

        # title = 'ACA Image Layout'
        sizes = [f'{msid_prefix}CA00040',   # Size of image 0
                 f'{msid_prefix}CA00043',   # Size of image 1
                 f'{msid_prefix}CA00046',   # Size of image 2
                 f'{msid_prefix}CA00049',   # Size of image 3
                 f'{msid_prefix}CA00052',   # Size of image 4
                 f'{msid_prefix}CA00055',   # Size of image 5
                 f'{msid_prefix}CA00058',   # Size of image 6
                 f'{msid_prefix}CA00061']   # Size of image 7

        rows = [f'{msid_prefix}CA00076',   # Row of pixel A1 of image 0
                f'{msid_prefix}CA00292',   # Row of pixel A1 of image 1
                f'{msid_prefix}CA00508',   # Row of pixel A1 of image 2
                f'{msid_prefix}CA00724',   # Row of pixel A1 of image 3
                f'{msid_prefix}CA00940',   # Row of pixel A1 of image 4
                f'{msid_prefix}CA01156',   # Row of pixel A1 of image 5
                f'{msid_prefix}CA01372',   # Row of pixel A1 of image 6
                f'{msid_prefix}CA01588']   # Row of pixel A1 of image 7

        cols = [f'{msid_prefix}CA00086',   # Column of pixel A1 of image 0
                f'{msid_prefix}CA00302',   # Column of pixel A1 of image 1
                f'{msid_prefix}CA00518',   # Column of pixel A1 of image 2
                f'{msid_prefix}CA00734',   # Column of pixel A1 of image 3
                f'{msid_prefix}CA00950',   # Column of pixel A1 of image 4
                f'{msid_prefix}CA01166',   # Column of pixel A1 of image 5
                f'{msid_prefix}CA01382',   # Column of pixel A1 of image 6
                f'{msid_prefix}CA01598']  # Column of pixel A1 of image 7

        scale_factor = [f'{msid_prefix}CA00096',   # Scale factor of image 0
                        f'{msid_prefix}CA00312',   # Scale factor of image 1
                        f'{msid_prefix}CA00528',   # Scale factor of image 2
                        f'{msid_prefix}CA00744',   # Scale factor of image 3
                        f'{msid_prefix}CA00960',   # Scale factor of image 4
                        f'{msid_prefix}CA01176',   # Scale factor of image 5
                        f'{msid_prefix}CA01392',   # Scale factor of image 6
                        f'{msid_prefix}CA01608']  # Scale factor of image 7

        pixels = []
        for px_img_num in px_img_nums:
            # title = f'{px_msid_prefix}{px_img_nums[i]}x{px_nums[k]}'
            pixels.append([f'{px_msid_prefix}{px_img_num}{px_id}{px_num}'
                           for px_num in px_nums for px_id in px_ids])

        self.append(primary_msid)
        self.extend(sizes)
        self.extend(rows)
        self.extend(cols)
        self.extend(scale_factor)
        self.extend(sum(pixels, []))
        self.append(integration_time)

        self.sizes = sizes
        self.rows = rows
        self.cols = cols
        self.scale_factor = scale_factor
        self.pixels = pixels
        self.ref = primary_msid
        self.integration_time = integration_time

    def slot(self, i):
        return [self.sizes[i], self.rows[i], self.cols[i], self.scale_factor[i]] + self.pixels[i]


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
    return table


def assemble(msids, data, full=False, calibrate=False, adjust_time=False, adjust_corner=False):
    """
    This is an example of fetching and assembling data using maude.

    Example usage::

      >>> from chandra_aca import maude_decom
      >>> data = maude_decom.fetch(start, stop, 1)

    It will be changed once we know::

      - what other telemetry to include
      - what structure should the data be in the viewer

    """

    # store it as a dictionary for convenience
    res = {e['msid']: e for e in data}

    # and reshape all values using the times from an MSID we know will be there at all sample times:
    tref = res[msids.sizes[0]]['times']
    data = {k: _reshape_values(res[k], tref) for k in msids}

    images = []
    subimage = np.zeros((8, len(tref)), dtype=int)
    for slot in range(8):
        if len(tref) == 0:
            continue
        pixel_data = {k: data[k] for k in msids.pixels[slot]}
        img_size = data[msids.sizes[slot]]['values']
        images.append(assemble_image(pixel_data, img_size))
        # there must be an MSID to fetch this, but this works
        subimage[slot] = np.char.replace(np.char.replace(np.char.replace(
            img_size, '8X8', ''), '6X6', ''), '4X4', '').astype(int) - 1

    result = []
    for slot in range(8):
        if len(tref) == 0:
            names = ['time', 'imgnum', 'size', 'row0', 'col0', 'scale_factor', 'integ', 'img']
            dtype = ['<f8', '<i8', '<U4', '<f8', '<f8', '<f8', '<f8', ('<f8', (8, 8))]
            table = {n: np.array([], dtype=t) for n, t in zip(names, dtype)}
        else:
            table = {
                'time': data[msids.sizes[slot]]['times'],
                'imgnum': np.ones(len(tref), dtype='<i8') * slot,
                'subimage': subimage[slot],
                'size': data[msids.sizes[slot]]['values'],
                'row0': data[msids.rows[slot]]['values'],
                'col0': data[msids.cols[slot]]['values'],
                'scale_factor': data[msids.scale_factor[slot]]['values'],
                'integ': 0.016 * data[msids.integration_time]['values'],
                'img': images[slot]
            }
            if full:
                table = combine_sub_images(table)
                del table['subimage']

        result.append(table)

    result = vstack([Table(r) for r in result])

    if calibrate:
        # as specified in ACA L0 ICD, section D.2.2 (scale_factor is already divided by 32)
        result['img'] *= result['scale_factor'][:, np.newaxis, np.newaxis]
        result['img'] -= 50

    if adjust_time:
        result['time'] -= (result['integ'] / 2 + 1.025)

    if adjust_corner:
        result['row0'][result['size'] == '6X61'] -= 1
        result['col0'][result['size'] == '6X61'] -= 1

    if full:
        result['size'][result['size'] == '4X41'] = '4X4'
        result['size'][result['size'] == '6X61'] = '6X6'
        result['size'][result['size'] == '8X81'] = '8X8'

    return result


def fetch(start, stop, pea_choice=1, full=False,
          calibrate=False, adjust_time=False, adjust_corner=False):
    """
    This is an example of fetching and assembling data using maude.

    Example usage::

      >>> from chandra_aca import maude_decom
      >>> data = maude_decom.fetch(start, stop, 1)

    It will be changed once we know::

      - what other telemetry to include
      - what structure should the data be in the viewer

    """
    msids = AcaTelemetryMsidList(pea_choice)

    # get maude data in batches of at most 100 (it fails otherwise)
    tmp = sum([maude.get_msids(s, start=start, stop=stop)['data'] for s in _subsets(msids, 100)],
              [])
    return assemble(msids, tmp, full=full,
                    calibrate=calibrate, adjust_time=adjust_time, adjust_corner=adjust_corner)
