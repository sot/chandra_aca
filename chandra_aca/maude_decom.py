"""
Classes and functions to help fetching ACA telemetry data using Maude.
"""

import numpy as np


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

        self.sizes = sizes
        self.rows = rows
        self.cols = cols
        self.scale_factor = scale_factor
        self.pixels = pixels
        self.ref = primary_msid


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

    size_4x4 = (img_size == '4X41')
    size_6x6 = (img_size == '6X61') + (img_size == '6X62')
    size_8x8 = ((img_size == '8X81') + (img_size == '8X82') +
                (img_size == '8X83') + (img_size == '8X84'))

    # -----------------------------------------
    # | -- | -- | -- | -- | -- | -- | -- | -- |
    # -----------------------------------------
    # | -- | -- | -- | -- | -- | -- | -- | -- |
    # -----------------------------------------
    # | -- | -- | D1 | H1 | L1 | P1 | -- | -- |
    # -----------------------------------------
    # | -- | -- | C1 | G1 | K1 | O1 | -- | -- |
    # -----------------------------------------
    # | -- | -- | B1 | F1 | J1 | N1 | -- | -- |
    # -----------------------------------------
    # | -- | -- | A1 | E1 | I1 | M1 | -- | -- |
    # -----------------------------------------
    # | -- | -- | -- | -- | -- | -- | -- | -- |
    # -----------------------------------------
    # | -- | -- | -- | -- | -- | -- | -- | -- |
    # -----------------------------------------
    img[size_4x4, 2, 2] = pixel_data[f'{msid_img}A1']['values'][size_4x4]
    img[size_4x4, 3, 2] = pixel_data[f'{msid_img}E1']['values'][size_4x4]
    img[size_4x4, 4, 2] = pixel_data[f'{msid_img}I1']['values'][size_4x4]
    img[size_4x4, 5, 2] = pixel_data[f'{msid_img}M1']['values'][size_4x4]
    img[size_4x4, 2, 3] = pixel_data[f'{msid_img}B1']['values'][size_4x4]
    img[size_4x4, 3, 3] = pixel_data[f'{msid_img}F1']['values'][size_4x4]
    img[size_4x4, 4, 3] = pixel_data[f'{msid_img}J1']['values'][size_4x4]
    img[size_4x4, 5, 3] = pixel_data[f'{msid_img}N1']['values'][size_4x4]
    img[size_4x4, 2, 4] = pixel_data[f'{msid_img}C1']['values'][size_4x4]
    img[size_4x4, 3, 4] = pixel_data[f'{msid_img}G1']['values'][size_4x4]
    img[size_4x4, 4, 4] = pixel_data[f'{msid_img}K1']['values'][size_4x4]
    img[size_4x4, 5, 4] = pixel_data[f'{msid_img}O1']['values'][size_4x4]
    img[size_4x4, 2, 5] = pixel_data[f'{msid_img}D1']['values'][size_4x4]
    img[size_4x4, 3, 5] = pixel_data[f'{msid_img}H1']['values'][size_4x4]
    img[size_4x4, 4, 5] = pixel_data[f'{msid_img}L1']['values'][size_4x4]
    img[size_4x4, 5, 5] = pixel_data[f'{msid_img}P1']['values'][size_4x4]

    # -----------------------------------------
    # | -- | -- | -- | -- | -- | -- | -- | -- |
    # -----------------------------------------
    # | -- | -- | E2 | F2 | G2 | H2 | -- | -- |
    # -----------------------------------------
    # | -- | D2 | D1 | H1 | L1 | P1 | I2 | -- |
    # -----------------------------------------
    # | -- | C2 | C1 | G1 | K1 | O1 | J2 | -- |
    # -----------------------------------------
    # | -- | B2 | B1 | F1 | J1 | N1 | K2 | -- |
    # -----------------------------------------
    # | -- | A2 | A1 | E1 | I1 | M1 | L2 | -- |
    # -----------------------------------------
    # | -- | -- | P2 | O2 | N2 | M2 | -- | -- |
    # -----------------------------------------
    # | -- | -- | -- | -- | -- | -- | -- | -- |
    # -----------------------------------------
    img[size_6x6, 2, 1] = pixel_data[f'{msid_img}P2']['values'][size_6x6]
    img[size_6x6, 3, 1] = pixel_data[f'{msid_img}O2']['values'][size_6x6]
    img[size_6x6, 4, 1] = pixel_data[f'{msid_img}N2']['values'][size_6x6]
    img[size_6x6, 5, 1] = pixel_data[f'{msid_img}M2']['values'][size_6x6]
    img[size_6x6, 1, 2] = pixel_data[f'{msid_img}A2']['values'][size_6x6]
    img[size_6x6, 2, 2] = pixel_data[f'{msid_img}A1']['values'][size_6x6]
    img[size_6x6, 3, 2] = pixel_data[f'{msid_img}E1']['values'][size_6x6]
    img[size_6x6, 4, 2] = pixel_data[f'{msid_img}I1']['values'][size_6x6]
    img[size_6x6, 5, 2] = pixel_data[f'{msid_img}M1']['values'][size_6x6]
    img[size_6x6, 6, 2] = pixel_data[f'{msid_img}L2']['values'][size_6x6]
    img[size_6x6, 1, 3] = pixel_data[f'{msid_img}B2']['values'][size_6x6]
    img[size_6x6, 2, 3] = pixel_data[f'{msid_img}B1']['values'][size_6x6]
    img[size_6x6, 3, 3] = pixel_data[f'{msid_img}F1']['values'][size_6x6]
    img[size_6x6, 4, 3] = pixel_data[f'{msid_img}J1']['values'][size_6x6]
    img[size_6x6, 5, 3] = pixel_data[f'{msid_img}N1']['values'][size_6x6]
    img[size_6x6, 6, 3] = pixel_data[f'{msid_img}K2']['values'][size_6x6]
    img[size_6x6, 1, 4] = pixel_data[f'{msid_img}C2']['values'][size_6x6]
    img[size_6x6, 2, 4] = pixel_data[f'{msid_img}C1']['values'][size_6x6]
    img[size_6x6, 3, 4] = pixel_data[f'{msid_img}G1']['values'][size_6x6]
    img[size_6x6, 4, 4] = pixel_data[f'{msid_img}K1']['values'][size_6x6]
    img[size_6x6, 5, 4] = pixel_data[f'{msid_img}O1']['values'][size_6x6]
    img[size_6x6, 6, 4] = pixel_data[f'{msid_img}J2']['values'][size_6x6]
    img[size_6x6, 1, 5] = pixel_data[f'{msid_img}D2']['values'][size_6x6]
    img[size_6x6, 2, 5] = pixel_data[f'{msid_img}D1']['values'][size_6x6]
    img[size_6x6, 3, 5] = pixel_data[f'{msid_img}H1']['values'][size_6x6]
    img[size_6x6, 4, 5] = pixel_data[f'{msid_img}L1']['values'][size_6x6]
    img[size_6x6, 5, 5] = pixel_data[f'{msid_img}P1']['values'][size_6x6]
    img[size_6x6, 6, 5] = pixel_data[f'{msid_img}I2']['values'][size_6x6]
    img[size_6x6, 2, 6] = pixel_data[f'{msid_img}E2']['values'][size_6x6]
    img[size_6x6, 3, 6] = pixel_data[f'{msid_img}F2']['values'][size_6x6]
    img[size_6x6, 4, 6] = pixel_data[f'{msid_img}G2']['values'][size_6x6]
    img[size_6x6, 5, 6] = pixel_data[f'{msid_img}H2']['values'][size_6x6]

    # -----------------------------------------
    # | H1 | P1 | H2 | P2 | H3 | P3 | H4 | P4 |
    # -----------------------------------------
    # | G1 | O1 | G2 | O2 | G3 | O3 | G4 | O4 |
    # -----------------------------------------
    # | F1 | N1 | F2 | N2 | F3 | N3 | F4 | N4 |
    # -----------------------------------------
    # | E1 | M1 | E2 | M2 | E3 | M3 | E4 | M4 |
    # -----------------------------------------
    # | D1 | L1 | D2 | L2 | D3 | L3 | D4 | L4 |
    # -----------------------------------------
    # | C1 | K1 | C2 | K2 | C3 | K3 | C4 | K4 |
    # -----------------------------------------
    # | B1 | J1 | B2 | J2 | B3 | J3 | B4 | J4 |
    # -----------------------------------------
    # | A1 | I1 | A2 | I2 | A3 | I3 | A4 | I4 |
    # -----------------------------------------
    img[size_8x8, 0, 0] = pixel_data[f'{msid_img}A1']['values'][size_8x8]
    img[size_8x8, 1, 0] = pixel_data[f'{msid_img}I1']['values'][size_8x8]
    img[size_8x8, 2, 0] = pixel_data[f'{msid_img}A2']['values'][size_8x8]
    img[size_8x8, 3, 0] = pixel_data[f'{msid_img}I2']['values'][size_8x8]
    img[size_8x8, 4, 0] = pixel_data[f'{msid_img}A3']['values'][size_8x8]
    img[size_8x8, 5, 0] = pixel_data[f'{msid_img}I3']['values'][size_8x8]
    img[size_8x8, 6, 0] = pixel_data[f'{msid_img}A4']['values'][size_8x8]
    img[size_8x8, 7, 0] = pixel_data[f'{msid_img}I4']['values'][size_8x8]
    img[size_8x8, 0, 1] = pixel_data[f'{msid_img}B1']['values'][size_8x8]
    img[size_8x8, 1, 1] = pixel_data[f'{msid_img}J1']['values'][size_8x8]
    img[size_8x8, 2, 1] = pixel_data[f'{msid_img}B2']['values'][size_8x8]
    img[size_8x8, 3, 1] = pixel_data[f'{msid_img}J2']['values'][size_8x8]
    img[size_8x8, 4, 1] = pixel_data[f'{msid_img}B3']['values'][size_8x8]
    img[size_8x8, 5, 1] = pixel_data[f'{msid_img}J3']['values'][size_8x8]
    img[size_8x8, 6, 1] = pixel_data[f'{msid_img}B4']['values'][size_8x8]
    img[size_8x8, 7, 1] = pixel_data[f'{msid_img}J4']['values'][size_8x8]
    img[size_8x8, 0, 2] = pixel_data[f'{msid_img}C1']['values'][size_8x8]
    img[size_8x8, 1, 2] = pixel_data[f'{msid_img}K1']['values'][size_8x8]
    img[size_8x8, 2, 2] = pixel_data[f'{msid_img}C2']['values'][size_8x8]
    img[size_8x8, 3, 2] = pixel_data[f'{msid_img}K2']['values'][size_8x8]
    img[size_8x8, 4, 2] = pixel_data[f'{msid_img}C3']['values'][size_8x8]
    img[size_8x8, 5, 2] = pixel_data[f'{msid_img}K3']['values'][size_8x8]
    img[size_8x8, 6, 2] = pixel_data[f'{msid_img}C4']['values'][size_8x8]
    img[size_8x8, 7, 2] = pixel_data[f'{msid_img}K4']['values'][size_8x8]
    img[size_8x8, 0, 3] = pixel_data[f'{msid_img}D1']['values'][size_8x8]
    img[size_8x8, 1, 3] = pixel_data[f'{msid_img}L1']['values'][size_8x8]
    img[size_8x8, 2, 3] = pixel_data[f'{msid_img}D2']['values'][size_8x8]
    img[size_8x8, 3, 3] = pixel_data[f'{msid_img}L2']['values'][size_8x8]
    img[size_8x8, 4, 3] = pixel_data[f'{msid_img}D3']['values'][size_8x8]
    img[size_8x8, 5, 3] = pixel_data[f'{msid_img}L3']['values'][size_8x8]
    img[size_8x8, 6, 3] = pixel_data[f'{msid_img}D4']['values'][size_8x8]
    img[size_8x8, 7, 3] = pixel_data[f'{msid_img}L4']['values'][size_8x8]
    img[size_8x8, 0, 4] = pixel_data[f'{msid_img}E1']['values'][size_8x8]
    img[size_8x8, 1, 4] = pixel_data[f'{msid_img}M1']['values'][size_8x8]
    img[size_8x8, 2, 4] = pixel_data[f'{msid_img}E2']['values'][size_8x8]
    img[size_8x8, 3, 4] = pixel_data[f'{msid_img}M2']['values'][size_8x8]
    img[size_8x8, 4, 4] = pixel_data[f'{msid_img}E3']['values'][size_8x8]
    img[size_8x8, 5, 4] = pixel_data[f'{msid_img}M3']['values'][size_8x8]
    img[size_8x8, 6, 4] = pixel_data[f'{msid_img}E4']['values'][size_8x8]
    img[size_8x8, 7, 4] = pixel_data[f'{msid_img}M4']['values'][size_8x8]
    img[size_8x8, 0, 5] = pixel_data[f'{msid_img}F1']['values'][size_8x8]
    img[size_8x8, 1, 5] = pixel_data[f'{msid_img}N1']['values'][size_8x8]
    img[size_8x8, 2, 5] = pixel_data[f'{msid_img}F2']['values'][size_8x8]
    img[size_8x8, 3, 5] = pixel_data[f'{msid_img}N2']['values'][size_8x8]
    img[size_8x8, 4, 5] = pixel_data[f'{msid_img}F3']['values'][size_8x8]
    img[size_8x8, 5, 5] = pixel_data[f'{msid_img}N3']['values'][size_8x8]
    img[size_8x8, 6, 5] = pixel_data[f'{msid_img}F4']['values'][size_8x8]
    img[size_8x8, 7, 5] = pixel_data[f'{msid_img}N4']['values'][size_8x8]
    img[size_8x8, 0, 6] = pixel_data[f'{msid_img}G1']['values'][size_8x8]
    img[size_8x8, 1, 6] = pixel_data[f'{msid_img}O1']['values'][size_8x8]
    img[size_8x8, 2, 6] = pixel_data[f'{msid_img}G2']['values'][size_8x8]
    img[size_8x8, 3, 6] = pixel_data[f'{msid_img}O2']['values'][size_8x8]
    img[size_8x8, 4, 6] = pixel_data[f'{msid_img}G3']['values'][size_8x8]
    img[size_8x8, 5, 6] = pixel_data[f'{msid_img}O3']['values'][size_8x8]
    img[size_8x8, 6, 6] = pixel_data[f'{msid_img}G4']['values'][size_8x8]
    img[size_8x8, 7, 6] = pixel_data[f'{msid_img}O4']['values'][size_8x8]
    img[size_8x8, 0, 7] = pixel_data[f'{msid_img}H1']['values'][size_8x8]
    img[size_8x8, 1, 7] = pixel_data[f'{msid_img}P1']['values'][size_8x8]
    img[size_8x8, 2, 7] = pixel_data[f'{msid_img}H2']['values'][size_8x8]
    img[size_8x8, 3, 7] = pixel_data[f'{msid_img}P2']['values'][size_8x8]
    img[size_8x8, 4, 7] = pixel_data[f'{msid_img}H3']['values'][size_8x8]
    img[size_8x8, 5, 7] = pixel_data[f'{msid_img}P3']['values'][size_8x8]
    img[size_8x8, 6, 7] = pixel_data[f'{msid_img}H4']['values'][size_8x8]
    img[size_8x8, 7, 7] = pixel_data[f'{msid_img}P4']['values'][size_8x8]

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

    # the following line would fail because times are not exactly the same
    # ok = tref[np.newaxis, :] == t[:, np.newaxis]
    # instead, I use np.isclose with a tolerance that has to be checked:
    ok = np.isclose(tref[np.newaxis, :], t[:, np.newaxis], atol=np.min(np.diff(tref)) / 2, rtol=0)
    i, j = np.broadcast_arrays(
        np.arange(ok.shape[1])[np.newaxis, :], np.arange(ok.shape[0])[:, np.newaxis])
    v = np.ones(tref.shape) * np.nan
    v[i[ok]] = data['values'][j[ok]]
    return {'times': tref, 'values': v}


def fetch(start, stop, pea_choice=1, fmt='astropy'):
    """
    This is an example of fetching and assembling data using maude.

    Example usage::

      >>> from chandra_aca import maude_decom
      >>> data = maude_decom.fetch(start, stop, 1)

    It will be changed once we know::

      - what other telemetry to include
      - what structure should the data be in the viewer

    """
    import maude
    from astropy.table import Table, vstack

    msids = AcaTelemetryMsidList(pea_choice)

    # get maude data in batches of at most 100 (it fails otherwise)
    tmp = sum([maude.get_msids(s, start=start, stop=stop)['data'] for s in _subsets(msids, 100)],
              [])

    # store it as a dictionary for convenience
    res = {e['msid']: e for e in tmp}

    # and reshape all values using the times from an MSID we know will be there at all sample times:
    tref = res[msids.sizes[0]]['times']
    data = {k: _reshape_values(res[k], tref) for k in msids}

    images = []
    for slot in range(8):
        pixel_data = {k: data[k] for k in msids.pixels[slot]}
        img_size = data[msids.sizes[slot]]['values']
        images.append(assemble_image(pixel_data, img_size))

    result = []
    for slot in range(8):
        if fmt == 'astropy':
            table = Table()
            table['time'] = data[msids.sizes[slot]]['times']
            table['imgnum'] = slot
            table['size'] = data[msids.sizes[slot]]['values']
            table['row0'] = data[msids.rows[slot]]['values']
            table['col0'] = data[msids.cols[slot]]
            table['scale_factor'] = data[msids.scale_factor[slot]]['values']
            table['img'] = images[slot]
            result.append(table)
        elif fmt == 'dict':
            table = {}
            table['size'] = data[msids.sizes[slot]]
            table['row0'] = data[msids.rows[slot]]
            table['col0'] = data[msids.cols[slot]]
            table['scale_factor'] = data[msids.scale_factor[slot]]
            table['img'] = images[slot]
            result.append(table)
        else:
            raise Exception(f'Unknown output format {fmt}')

    if fmt == 'astropy':
        result = vstack(result)
    return result
