"""
Classes and functions to help fetching ACA telemetry data using MAUDE.

See :ref:`maude_decom` for more details and examples.
"""

import copy
import itertools
from struct import Struct
from struct import unpack as _unpack

import astropy.units as u
import maude
import numpy as np
from astropy.table import Table, vstack
from cxotime import CxoTime, CxoTimeLike

# Maude fetch limits
MAUDE_SINGLE_FETCH_LIMIT = 3.0 * u.hour
MAUDE_FETCH_LIMIT = 5 * u.day

# maximum values for frame counters (for convenience)
MAX_MJF = (2 << 16) - 1
MAX_MNF = (2 << 6) - 1
MAX_VCDU = (2 << 23) - 1

# The following are the tables in the docstring above. They appear to be transposed,
# but the resultt agrees with level0.
PIXEL_MAP = {
    "4x4": np.array(
        [
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "A1", "B1", "C1", "D1", "  ", "  "],
            ["  ", "  ", "E1", "F1", "G1", "H1", "  ", "  "],
            ["  ", "  ", "I1", "J1", "K1", "L1", "  ", "  "],
            ["  ", "  ", "M1", "N1", "O1", "P1", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
        ]
    ),
    "6x6": np.array(
        [
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "A2", "B2", "C2", "D2", "  ", "  "],
            ["  ", "P2", "A1", "B1", "C1", "D1", "E2", "  "],
            ["  ", "O2", "E1", "F1", "G1", "H1", "F2", "  "],
            ["  ", "N2", "I1", "J1", "K1", "L1", "G2", "  "],
            ["  ", "M2", "M1", "N1", "O1", "P1", "H2", "  "],
            ["  ", "  ", "L2", "K2", "J2", "I2", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
        ]
    ),
    "8x8": np.array(
        [
            ["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"],
            ["I1", "J1", "K1", "L1", "M1", "N1", "O1", "P1"],
            ["A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2"],
            ["I2", "J2", "K2", "L2", "M2", "N2", "O2", "P2"],
            ["A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3"],
            ["I3", "J3", "K3", "L3", "M3", "N3", "O3", "P3"],
            ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4"],
            ["I4", "J4", "K4", "L4", "M4", "N4", "O4", "P4"],
        ]
    ),
    "DNLD": np.array(
        [
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ["  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "],
        ]
    ),
}

PIXEL_MASK = {k: PIXEL_MAP[k] == "  " for k in PIXEL_MAP}
_ROWS, _COLS = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")

PIXEL_MAP_INV = {
    k: {
        p: (i, j)
        for i, j, p in zip(
            _ROWS[PIXEL_MAP[k] != "  "],
            _COLS[PIXEL_MAP[k] != "  "],
            PIXEL_MAP[k][PIXEL_MAP[k] != "  "],
            strict=False,
        )
    }
    for k in ["6x6", "4x4", "8x8"]
}


_msid_prefix = {1: "A", 2: "R"}


def _aca_msid_list(pea):
    # helper method to make a dictionary with all global (non-slot) MSIDs used here
    return {
        "status": "AOACSTAT",  # ASPECT CAMERA DATA PROCESSING OVERALL STATUS FLAG
        "integration_time": f"{_msid_prefix[pea]}ACAINT0",
        "major_frame": "CVCMJCTR",
        "minor_frame": "CVCMNCTR",
        "cmd_count": f"{_msid_prefix[pea]}CCMDS",
        "cmd_progress_to_go": f"{_msid_prefix[pea]}AROW2GO",  # No. of ROWS TO GO COMMAND PROGRESS
        "cmd_progress": "AOCMDPG1",  # COMMAND PROGRESS COUNT
        "pixel_telemetry_type": f"{_msid_prefix[pea]}APIXTLM",
        "dynamic_background_type": f"{_msid_prefix[pea]}ABGDTYP",
    }


def _aca_image_msid_list(pea):
    # helper method to make a list of dictionaries with MSIDs for each slot in a given PEA.
    msid_prefix = _msid_prefix[pea]

    px_ids = {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
    }
    px_nums = [str(n) for n in range(1, 5)]
    px_img_nums = [str(n) for n in range(8)]

    pixels = [
        sorted(
            [
                f"{msid_prefix}CIMG{px_img_num}{px_id}{px_num}"
                for px_num in px_nums
                for px_id in px_ids
            ]
        )
        for px_img_num in px_img_nums
    ]

    res = {
        "pixels": pixels,
        "sizes": [
            f"{msid_prefix}CA00040",  # Size of image 0
            f"{msid_prefix}CA00043",  # Size of image 1
            f"{msid_prefix}CA00046",  # Size of image 2
            f"{msid_prefix}CA00049",  # Size of image 3
            f"{msid_prefix}CA00052",  # Size of image 4
            f"{msid_prefix}CA00055",  # Size of image 5
            f"{msid_prefix}CA00058",  # Size of image 6
            f"{msid_prefix}CA00061",
        ],  # Size of image 7
        "rows": [
            f"{msid_prefix}CA00076",  # Row of pixel A1 of image 0
            f"{msid_prefix}CA00292",  # Row of pixel A1 of image 1
            f"{msid_prefix}CA00508",  # Row of pixel A1 of image 2
            f"{msid_prefix}CA00724",  # Row of pixel A1 of image 3
            f"{msid_prefix}CA00940",  # Row of pixel A1 of image 4
            f"{msid_prefix}CA01156",  # Row of pixel A1 of image 5
            f"{msid_prefix}CA01372",  # Row of pixel A1 of image 6
            f"{msid_prefix}CA01588",
        ],  # Row of pixel A1 of image 7
        "cols": [
            f"{msid_prefix}CA00086",  # Column of pixel A1 of image 0
            f"{msid_prefix}CA00302",  # Column of pixel A1 of image 1
            f"{msid_prefix}CA00518",  # Column of pixel A1 of image 2
            f"{msid_prefix}CA00734",  # Column of pixel A1 of image 3
            f"{msid_prefix}CA00950",  # Column of pixel A1 of image 4
            f"{msid_prefix}CA01166",  # Column of pixel A1 of image 5
            f"{msid_prefix}CA01382",  # Column of pixel A1 of image 6
            f"{msid_prefix}CA01598",
        ],  # Column of pixel A1 of image 7
        "scale_factor": [
            f"{msid_prefix}CA00096",  # Scale factor of image 0
            f"{msid_prefix}CA00312",  # Scale factor of image 1
            f"{msid_prefix}CA00528",  # Scale factor of image 2
            f"{msid_prefix}CA00744",  # Scale factor of image 3
            f"{msid_prefix}CA00960",  # Scale factor of image 4
            f"{msid_prefix}CA01176",  # Scale factor of image 5
            f"{msid_prefix}CA01392",  # Scale factor of image 6
            f"{msid_prefix}CA01608",
        ],  # Scale factor of image 7
        "image_status": [f"AOIMAGE{i}" for i in range(8)],  # IMAGE STATUS FLAG
        "fiducial_flag": [f"AOACFID{i}" for i in range(8)],  # FIDUCIAL LIGHT FLAG (OBC)
        "image_function_obc": [f"AOACFCT{i}" for i in range(8)],  # IMAGE FUNCTION (OBC)
        "image_function": [
            f"{msid_prefix}AIMGF{i}1" for i in range(8)
        ],  # IMAGE FUNCTION (PEA)
        # this one exists also as FUNCTION2/3/4
        # 'image_function':
        #     [f'{msid_prefix}AIMGF{i}1' for i in range(8)],  # IMAGE FUNCTION1 (PEA)
        "saturated_pixel": [
            f"{msid_prefix}ASPXF{i}" for i in range(8)
        ],  # DEFECTIVE PIXEL FLAG
        "defective_pixel": [
            f"{msid_prefix}ADPXF{i}" for i in range(8)
        ],  # SATURATED PIXEL FLAG
        "quad_bound": [
            f"{msid_prefix}QBNDF{i}" for i in range(8)
        ],  # QUADRANT BOUNDRY FLAG
        "common_col": [
            f"{msid_prefix}ACOLF{i}" for i in range(8)
        ],  # COMMON COLUMN FLAG
        "multi_star": [
            f"{msid_prefix}AMSTF{i}" for i in range(8)
        ],  # MULTIPLE STAR FLAG
        "ion_rad": [
            f"{msid_prefix}AIRDF{i}" for i in range(8)
        ],  # IONIZING RADIATION FLAG
        "background_rms": [f"{msid_prefix}CRMSBG{i}" for i in range(8)],
        "background_avg": [
            f"{msid_prefix}CA00110",
            f"{msid_prefix}CA00326",
            f"{msid_prefix}CA00542",
            f"{msid_prefix}CA00758",
            f"{msid_prefix}CA00974",
            f"{msid_prefix}CA01190",
            f"{msid_prefix}CA01406",
            f"{msid_prefix}CA01622",
        ],
        "housing_temperature": [
            f"{msid_prefix}ACH1T{i}2" for i in range(8)
        ],  # AC HOUSING TEMPERATURE
        "ccd_temperature": [
            f"{msid_prefix}CCDPT{i}2" for i in range(8)
        ],  # CCD TEMPERATURE
        "primary_temperature": [
            f"{msid_prefix}QTAPMT{i}" for i in range(8)
        ],  # PRIMARY MIRROR/LENS CELL TEMP
        "secondary_temperature": [
            f"{msid_prefix}QTH2MT{i}" for i in range(8)
        ],  # AC SECONDARY MIRROR TEMPERATURE
        "magnitude": [
            f"AOACMAG{i}" for i in range(8)
        ],  # STAR OR FIDUCIAL MAGNITUDE (OBC)
        "centroid_ang_y": [
            f"AOACYAN{i}" for i in range(8)
        ],  # YAG CENTROID Y ANGLE (OBC)
        "centroid_ang_z": [
            f"AOACZAN{i}" for i in range(8)
        ],  # ZAG CENTROID Z ANGLE (OBC)
        "bgd_stat_pixels": [
            [f"ACBPX{j}1{i}" for j in "ABGH"] + [f"ACBPX{j}4{i}" for j in "IJOP"]
            for i in range(8)
        ],
    }
    return [{k: res[k][i] for k in res} for i in range(8)]


ACA_MSID_LIST = {i + 1: _aca_msid_list(i + 1) for i in range(2)}
ACA_SLOT_MSID_LIST = {i + 1: _aca_image_msid_list(i + 1) for i in range(2)}


_a2p = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
# This maps image type to a list of pixel indices over the 8x8 array.
# - Type 0 is 4x4,
# - Types 1 (first batch of 6x6) and 4 (first batch of 8x8) use the same pixel IDs as 4x4.
# - Type 3 is not a real image type. It occurs when the image telemetry is used to download
#   memory dump data. In this case, the image is treated as 4x4, but the values might be giberish.
# - Types 2 (second batch of 6x6) and 5 (second batch of 8x8) use the same pixel IDs.
_IMG_INDICES = [
    np.array([PIXEL_MAP_INV["4x4"][f"{k}1"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["6x6"][f"{k}1"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["6x6"][f"{k}2"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["4x4"][f"{k}1"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["8x8"][f"{k}1"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["8x8"][f"{k}2"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["8x8"][f"{k}3"] for k in _a2p]).T,
    np.array([PIXEL_MAP_INV["8x8"][f"{k}4"] for k in _a2p]).T,
]

######################
# VCDU-based functions
######################

# these are used multiple times
_aca_front_fmt = Struct(">HBBBBBB")
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


class _AcaImageHeaderDecom:
    """
    Class to decommute ACA image telemtry headers.

    These methods are grouped into a class because header 3 packet is split into up to 8 parts.
    The __call__ method in this class accumulates the partial packets. Once all images are of known
    types, and the packets are known, it will return the header 3 data.
    """

    def __init__(self):
        self._header = [
            self._aca_header_1,
            self._aca_header_1,
            self._aca_header_2,
            lambda b: {},
            self._aca_header_1,
            self._aca_header_2,
            self._aca_header_3,
            self._aca_header_3,
        ]

    def __call__(self, imgnum, imgtype, byte_array):
        return self._header[imgtype](byte_array)

    @staticmethod
    def _aca_header_1(bits):
        """
        Unpack ACA header 1 (ACA User Manual 5.3.2.2.1).

        Parameters
        ----------
        bits
            bytes-like object of length 7

        Returns
        -------
        dict
        """
        bits = np.unpackbits(np.array(_unpack("BBBBBBB", bits), dtype=np.uint8))
        return {
            "IMGFID": bool(bits[0]),
            "IMGNUM": _packbits(bits[1:4]),
            "IMGFUNC": _packbits(bits[4:6]),
            "IMGSTAT": _packbits(bits[6:12]),
            "SAT_PIXEL": bool(bits[6]),
            "DEF_PIXEL": bool(bits[7]),
            "QUAD_BOUND": bool(bits[8]),
            "COMMON_COL": bool(bits[9]),
            "MULTI_STAR": bool(bits[10]),
            "ION_RAD": bool(bits[11]),
            "IMGROW0": _packbits(bits[12:22], unsigned=False),
            "IMGCOL0": _packbits(bits[22:32], unsigned=False),
            "IMGSCALE": _packbits(bits[32:46]),
            "BGDAVG": _packbits(bits[46:56]),
        }

    @staticmethod
    def _aca_header_2(bits):
        """
        Unpack ACA header 2 (ACA User Manual 5.3.2.2.2).

        Parameters
        ----------
        bits
            bytes-like object of length 7

        Returns
        -------
        dict
        """
        values = _unpack("BBbbbbB", bits)
        c = np.unpackbits(np.array(values[:2], dtype=np.uint8))
        return {
            # do we want these?
            # 'FID2': bool(values[0]),
            # 'IMGNUM2': _packbits(values[1:4]),
            # 'IMGFUNC2': _packbits(values[4:6]),
            "BGDRMS": _packbits(c[6:16]),
            "TEMPCCD": values[2],
            "TEMPHOUS": values[3],
            "TEMPPRIM": values[4],
            "TEMPSEC": values[5],
            "BGDSTAT": values[6],
            "BGDSTAT_PIXELS": np.unpackbits(np.array(values[-1:], dtype=np.uint8)[-1:]),
        }

    @staticmethod
    def _aca_header_3(bits):
        """
        Unpack ACA header 3 (ACA User Manual 5.3.2.2.3).

        Parameters
        ----------
        bits
            bytes-like object of length 7

        Returns
        -------
        dict
        """
        return {"DIAGNOSTIC": _unpack("BBBBBB", bits[1:])}


def unpack_aca_telemetry(packet):
    """
    Unpack ACA telemetry encoded in 225-byte packets.

    Parameters
    ----------
    packet
        bytes

    Returns
    -------
    list of dict
    A list of length 8, one entry per slot, where each entry is a dictionary.
    """
    s1, s2, s3 = _unpack("BBB", packet[5:8])
    _size_bits[:, -3:] = np.unpackbits(
        np.array([[s1, s2, s3]], dtype=np.uint8).T, axis=1
    ).reshape((8, -1))
    img_types = np.packbits(_size_bits, axis=1).T[0]
    slots = []
    header_decom = _AcaImageHeaderDecom()
    for img_num, i in enumerate(range(8, len(packet), 27)):
        img_header = {"IMGTYPE": img_types[img_num]}
        img_header.update(header_decom(img_num, img_types[img_num], packet[i : i + 7]))
        img_pixels = _unpack("B" * 20, packet[i + 7 : i + 27])
        _pixel_bits[:, -10:] = np.unpackbits(
            np.array([img_pixels], dtype=np.uint8).T, axis=1
        ).reshape((-1, 10))
        img_pixels = np.sum(np.packbits(_pixel_bits, axis=1) * [[2**8, 1]], axis=1)
        img_header["pixels"] = img_pixels
        slots.append(img_header)

    bgd_types = {0: "FLAT", 1: "DYNB"}
    pix_tlm_types = {0: "ORIG", 1: "DYNB", 2: "DIFF", 3: "ERR3"}
    # Before the dynamic background patch, the first two bytes contained INTEG in those
    # 16 bits (named integbits).  After the dynamic background patch, the first 6 bits of
    # integbits will be repurposed: two bits for PIXTLM, next bit for BGDTYP, 3 spares,
    # and 10 bits for INTEG.
    integbits = np.unpackbits(np.array(_unpack("BB", packet[0:2]), dtype=np.uint8))
    pixtlm = _packbits(integbits[0:2])
    bgdtyp = integbits[2]
    integ = _packbits(integbits[6:])
    glbstat = _unpack("B", packet[2:3])[0]
    bits = np.unpackbits(np.array(_unpack("BBB", packet[2:5]), dtype=np.uint8))
    res = {
        "AAPIXTLM": pix_tlm_types[pixtlm],
        "AABGDTYP": bgd_types[bgdtyp],
        "INTEG": integ,
        "GLBSTAT": glbstat,
        "HIGH_BGD": bool(bits[0]),
        "RAM_FAIL": bool(bits[1]),
        "ROM_FAIL": bool(bits[2]),
        "POWER_FAIL": bool(bits[3]),
        "CAL_FAIL": bool(bits[4]),
        "COMM_CHECKSUM_FAIL": bool(bits[5]),
        "RESET": bool(bits[6]),
        "SYNTAX_ERROR": bool(bits[7]),
        "COMMCNT": _packbits(bits[8:14], unsigned=False),
        "COMMCNT_SYNTAX_ERROR": bool(bits[14]),
        "COMMCNT_CHECKSUM_FAIL": bool(bits[15]),
        "COMMPROG": _packbits(bits[16:22], unsigned=False),
        "COMMPROG_REPEAT": _packbits(bits[22:24], unsigned=False),
    }
    for s in slots:
        s.update(res)
    return slots


def _combine_aca_packets(aca_packets):
    """
    Combine a list of ACA packets into a single record.

    This is intended to combine the two 6X6 packets or the four 8X8 packets.

    Parameters
    ----------
    aca_packets
        list of dict

    Returns
    -------
    dict
    """
    # note that they are reverse-sorted so the first frame overwrites the others if they collide
    aca_packets = sorted(aca_packets, key=lambda p: p["TIME"], reverse=True)
    res = {}
    pixels = np.ma.masked_all((8, 8))
    pixels.data[:] = np.nan
    for f in aca_packets:
        # IMGTYPE 3 is not a real image. It means the pixels are used to download memory dump data
        # if IMGTYPE == 3, do nothing. All pixels will be masked
        if f["IMGTYPE"] != 3:
            i0, i1 = _IMG_INDICES[f["IMGTYPE"]]
            pixels[i0, i1] = f["pixels"]

    for f in aca_packets:
        res.update(f)
    del res["pixels"]
    res["IMG"] = pixels
    return res


def _group_packets(packets, discard=True):
    """
    Group ACA image telemetry packets to form full images.

    ACA telemetry is packed in packets of 225 bytes, and each of these is split in four VCDU frames.
    Each ACA telemetry packet includes pixel telemetry for 16 pixels in each of the 8 slots.
    A 4x4 image is complete in one packet, whereas 6x6 and 8x8 images are split into two and four
    packets respectively.

    This function takes a list of dictionaries, with each dictionary containing the pixel
    telemetry for one slot in a VCDU frame. Each dictionary must have the keys 'MJF', 'MNF'
    and 'IMGTYPE'. It returns a list of lists of dicts, where each list of dicts contains the pixel
    telemetry for a full image.

    The function assumes:
    - all entries correspond to the same slot
    - the entries are ordered by increasing VCDU counter
    - missing/repeated VCDU counters have been removed

    If this is not the case, the results will be incorrect.

    Parameters
    ----------
    packets
        list of ACA sub-packets
    discard
        bool to discard incomplete ACA packets

    Returns
    -------
    list of ACA packets
    """
    res = []
    n = -1
    s = -1
    rollover = 0
    for i, packet in enumerate(packets):
        vcdu = packet["MJF"] * 128 + packet["MNF"]
        if i > 0 and packets[i - 1]["MJF"] * 128 + packets[i - 1]["MNF"] > vcdu:
            rollover += 1
        vcdu += rollover * 1 << 24
        if vcdu > n:
            if res and not discard:
                yield res
            res = []
        if not res:
            # the number of minor frames expected within the same ACA packet
            s = {0: 1, 1: 2, 2: 2, 3: 1, 4: 4, 5: 4, 6: 4, 7: 4}[int(packet["IMGTYPE"])]
            # the number of minor frames within the same ACA packet expected after this minor frame
            remaining = {0: 0, 1: 1, 2: 0, 3: 0, 4: 3, 5: 2, 6: 1, 7: 0}[
                int(packet["IMGTYPE"])
            ]
            n = vcdu + 4 * remaining
        res.append(packet)
        if vcdu == n and len(res) == s:
            yield res
            res = []
    if res and not discard:
        yield res


def filter_vcdu_jumps(vcdu_counters):
    """
    Return a boolean mask to filter VCDU counters that are not continuous.

    The returned mask ensures that::

        - VCDU counters are a strictly monotonic sequence.
        - VCDU counters come in packets of four, with each group starting with a multiple of 4.
    """
    current_packet = []  # to group in fours
    last_frame = -1  # for monotonicity check
    selected = np.zeros(len(vcdu_counters), dtype=bool)
    for i, vcdu_counter in enumerate(vcdu_counters):
        # major frame counter rolls over at 2**17, and minor counter at 2**7
        # The VCDU counter (which includes both) rolls over at 2**24 (16777216).
        # The VCDU counter must increase or roll over (in which case it jumps by a large negative
        # number) In the worse case, it rolls over _and_ 3600*4 frames are missing
        if not (
            last_frame < vcdu_counter
            or (vcdu_counter - last_frame) <= -(16777215 - 3600 * 4)
        ):
            last_frame = vcdu_counter
            continue
        last_frame = vcdu_counter
        if vcdu_counter % 4 == 0:
            current_packet = []
        current_packet.append(i)
        if vcdu_counter % 4 == 3 and len(current_packet) == 4:
            selected[current_packet] = True
    return selected


def get_raw_aca_packets(start, stop, maude_result=None, **maude_kwargs):
    """
    Fetch 1025-byte VCDU frames using MAUDE and extract a list of 225-byte ACA packets.

    If the first minor frame in a group of four ACA packets is within (start, stop),
    the three following minor frames are included if present.

    returns a dictionary with keys ['TIME', 'MNF', 'MJF', 'packets', 'flags'].
    These correspond to the minor frame time, minor frame count, major frame count,
    the list of packets, and flags returned by MAUDE respectively.

    This function raises an exception if the VCDU frames in maude_result are not contiguous, which
    can also happen if the frames are corrupted in some way.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for packets
    stop : CxoTimeLike
        Stop time for packets
    maude_result
        the result of calling maude.get_frames. Optional.
    **maude_kwargs
        keyword args passed to maude.get_frames()

    Returns
    -------
    dict
    {'flags': int, 'packets': [],
    'TIME': np.array([]), 'MNF': np.array([]), 'MJF': np.array([])}
    """
    date_start, date_stop = CxoTime(start), CxoTime(stop)
    stop_pad = 1.5 * u.s  # padding at the end in case of trailing partial ACA packets

    # get the frames and unpack front matter
    if maude_result is None:
        frames = maude.get_frames(
            start=date_start, stop=date_stop + stop_pad, **maude_kwargs
        )["data"]
    else:
        frames = maude_result["data"]
    flags = frames["f"]
    frames = frames["frames"]

    # ACA telemetry data is split into four frames. Each frame has a counter, and the first frame
    # in each group of four has a counter that is a multiple of 4. The following unpacks
    # the VCDU counter from the front matter of each frame.
    # The VCDU counter is stored in 24 bits, and struct.unpack does not have a 24-bit format code.
    # We therefore decom the bytes separately and combine them into a single integer using the
    # following weights:
    weights = np.array([1 << 16, 1 << 8, 1 << 0])
    vcdu = np.array(
        [np.sum(weights * _unpack("BBB", frame["bytes"][2:5])) for frame in frames]
    )

    # the following ensures that:
    #  - VCDU numbers are a strictly monotonic sequence
    #  - Frames come in packets of four. A missing frame would cause the packet to be dropped.
    selected = filter_vcdu_jumps(vcdu)
    vcdu = vcdu[selected]
    frames = [frame for frame, sel in zip(frames, selected, strict=False) if sel]
    vcdu_times = np.array([frame["t"] for frame in frames])

    sub = vcdu % 4  # the minor frame index within each ACA update

    major_counter = np.cumsum(
        sub == 0
    )  # this number is monotonically increasing starting at 0

    n = major_counter.max() + 1 if len(major_counter) > 0 else 1
    # only unpack complete ACA frames in the original range:
    aca_frame_entries = np.ma.masked_all((n, 4), dtype=int)
    if len(frames) == 0:
        return {
            "flags": 0,
            "packets": [],
            "TIME": [],
            "MNF": [],
            "MJF": [],
            "VCDUCTR": [],
        }
    aca_frame_entries[major_counter, sub] = np.arange(vcdu_times.shape[0])
    aca_frame_times = np.ma.masked_all((n, 4))
    aca_frame_times[major_counter, sub] = vcdu_times

    # this will remove ACA records with at least one missing minor frame
    select = (
        (~np.any(aca_frame_times.mask, axis=1))
        * (aca_frame_times[:, 0] >= date_start.secs)
        * (aca_frame_times[:, 0] <= date_stop.secs)
    )

    # assemble the 56 bit ACA minor records and times (time is not unpacked)
    aca = []
    aca_times = []
    for frame in frames:
        aca_times.append(frame["t"])
        aca.append(b"".join([frame["bytes"][j : j + 14] for j in [18, 274, 530, 780]]))

    # combine them into 224 byte frames (it currently ensures all 224 bytes are there)
    aca_packets = [
        b"".join([aca[i] for i in entry]) for entry in aca_frame_entries[select]
    ]
    for a in aca_packets:
        if len(a) != 224:
            raise ValueError("ACA packet is not 224 bytes long")

    times = vcdu_times[aca_frame_entries[select, 0]]
    vcdu_counter = vcdu[aca_frame_entries[select, 0]]

    return {
        "flags": flags,
        "packets": aca_packets,
        "TIME": times,
        "MNF": vcdu_counter % (1 << 7),
        "MJF": vcdu_counter // (1 << 7),
        "VCDUCTR": vcdu_counter,
    }


ACA_PACKETS_DTYPE = np.dtype(
    [
        ("TIME", np.float64),
        ("VCDUCTR", np.uint32),
        ("MJF", np.uint32),
        ("MNF", np.uint32),
        ("IMGNUM", np.uint32),
        ("COMMCNT", np.uint8),
        ("COMMPROG", np.uint8),
        ("GLBSTAT", np.uint8),
        ("IMGFUNC", np.uint32),
        ("IMGTYPE", np.uint8),
        ("IMGSCALE", np.uint16),
        ("IMGROW0", np.int16),
        ("IMGCOL0", np.int16),
        ("INTEG", np.uint16),
        ("BGDAVG", np.uint16),
        ("BGDRMS", np.uint16),
        ("TEMPCCD", np.int16),
        ("TEMPHOUS", np.int16),
        ("TEMPPRIM", np.int16),
        ("TEMPSEC", np.int16),
        ("BGDSTAT", np.uint8),
        ("HIGH_BGD", bool),
        ("RAM_FAIL", bool),
        ("ROM_FAIL", bool),
        ("POWER_FAIL", bool),
        ("CAL_FAIL", bool),
        ("COMM_CHECKSUM_FAIL", bool),
        ("RESET", bool),
        ("SYNTAX_ERROR", bool),
        ("COMMCNT_SYNTAX_ERROR", bool),
        ("COMMCNT_CHECKSUM_FAIL", bool),
        ("COMMPROG_REPEAT", np.uint8),
        ("IMGFID", bool),
        ("IMGSTAT", np.uint8),
        ("SAT_PIXEL", bool),
        ("DEF_PIXEL", bool),
        ("QUAD_BOUND", bool),
        ("COMMON_COL", bool),
        ("MULTI_STAR", bool),
        ("ION_RAD", bool),
        ("IMGROW_A1", np.int16),
        ("IMGCOL_A1", np.int16),
        ("IMGROW0_8X8", np.int16),
        ("IMGCOL0_8X8", np.int16),
        ("END_INTEG_TIME", np.float64),
        ("AAPIXTLM", "<U4"),
        ("AABGDTYP", "<U4"),
        ("IMG", "<f8", (8, 8)),
    ]
)


def _aca_packets_to_table(aca_packets, dtype=None):
    """
    Store ACA packets in a table.

    Parameters
    ----------
    aca_packets
        list of dict
    dtype
        dtype to use in the resulting table. Optional.

    Returns
    -------
    astropy.table.Table
    """
    import copy

    if dtype is None:
        dtype = ACA_PACKETS_DTYPE

    array = np.ma.masked_all([len(aca_packets)], dtype=dtype)
    names = copy.deepcopy(dtype.names)
    img = []
    for i, aca_packet in enumerate(aca_packets):
        if "IMG" in aca_packet:
            img.append(aca_packet["IMG"])
        for name in names:
            if name in aca_packet:
                array[name][i] = aca_packet[name]

    table = Table(array)
    if img:
        table["IMG"] = img
        for i, aca_packet in enumerate(aca_packets):
            table["IMG"].mask[i] = aca_packet["IMG"].mask
    return table


def get_aca_packets(
    start,
    stop,
    level0=False,
    combine=False,
    adjust_time=False,
    calibrate=False,
    blobs=None,
    frames=None,
    dtype=None,
    **maude_kwargs,
):
    """
    Fetch VCDU 1025-byte frames, extract ACA packets, unpack them and store them in a table.

    Incomplete ACA packets (if there is a minor frame missing) can be combined or not into records
    with complete ACA telemetry. Compare these to calls to the function:

            >>> from chandra_aca import maude_decom
            >>> img = maude_decom.get_aca_packets(684089000, 684089016, combine=True)
            >>> img = img[img['IMGNUM'] == 0]
            >>> img['TIME', 'MJF', 'MNF', 'COMMCNT', 'GLBSTAT', 'IMGTYPE', 'IMGROW0', 'IMGCOL0',
            >>>     'TEMPCCD', 'TEMPHOUS']
            <Table masked=True length=4>
                 TIME      MJF    MNF   COMMCNT GLBSTAT IMGTYPE IMGROW0 IMGCOL0 TEMPCCD TEMPHOUS
               float64    uint32 uint32  uint8   uint8   uint8   int16   int16   int16   int16
            ------------- ------ ------ ------- ------- ------- ------- ------- ------- --------
            684089001.869  78006     32       0       0       4     469    -332     -20       83
            684089005.969  78006     48       0       0       4     469    -332     -20       83
            684089010.069  78006     64       0       0       4     469    -332     -20       83
            684089014.169  78006     80       0       0       4     469    -332     -20       83

    Using combined=False, results in records with incomplete images. In this case, data can be
    missing from some records. For example, with 8X8 images, IMGROW0 and IMGCOL0 are present in the
    first ACA packet (image type 4) while the temperature is present in the second (image type 5):

        >>> from chandra_aca import maude_decom
        >>> img = maude_decom.get_aca_packets(684089000, 684089016, combine=False)
        >>> img = img[img['IMGNUM'] == 0]
        >>> img['TIME', 'MJF', 'MNF', 'COMMCNT', 'GLBSTAT', 'IMGTYPE', 'IMGROW0', 'IMGCOL0',
        >>>     'TEMPCCD', 'TEMPHOUS']
            <Table masked=True length=15>
                 TIME      MJF    MNF   COMMCNT GLBSTAT IMGTYPE IMGROW0 IMGCOL0 TEMPCCD TEMPHOUS
               float64    uint32 uint32  uint8   uint8   uint8   int16   int16   int16   int16
            ------------- ------ ------ ------- ------- ------- ------- ------- ------- --------
            684089000.844  78006     28       0       0       7      --      --      --       --
            684089001.869  78006     32       0       0       4     469    -332      --       --
            684089002.894  78006     36       0       0       5      --      --     -20       83
            684089003.919  78006     40       0       0       6      --      --      --       --
            684089004.944  78006     44       0       0       7      --      --      --       --
            684089005.969  78006     48       0       0       4     469    -332      --       --
            684089006.994  78006     52       0       0       5      --      --     -20       83
            684089008.019  78006     56       0       0       6      --      --      --       --
            684089009.044  78006     60       0       0       7      --      --      --       --
            684089010.069  78006     64       0       0       4     469    -332      --       --
            684089011.094  78006     68       0       0       5      --      --     -20       83
            684089012.119  78006     72       0       0       6      --      --      --       --
            684089013.144  78006     76       0       0       7      --      --      --       --
            684089014.169  78006     80       0       0       4     469    -332      --       --
            684089015.194  78006     84       0       0       5      --      --     -20       83

        >>> img['IMG'].data[1]
        masked_BaseColumn(data =
         [[60.0 97.0 70.0 120.0 74.0 111.0 103.0 108.0]
         [67.0 90.0 144.0 96.0 88.0 306.0 82.0 67.0]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]],
                          mask =
         [[False False False False False False False False]
         [False False False False False False False False]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]],
                    fill_value = 1e+20)

        >>> img['IMG'].data[2]
        masked_BaseColumn(data =
         [[-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [76.0 81.0 160.0 486.0 449.0 215.0 88.0 156.0]
         [68.0 91.0 539.0 483.0 619.0 412.0 105.0 77.0]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]],
                          mask =
         [[ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [False False False False False False False False]
         [False False False False False False False False]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]],
                    fill_value = 1e+20)

        >>> img['IMG'].data[3]
        masked_BaseColumn(data =
         [[-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [86.0 101.0 408.0 344.0 556.0 343.0 122.0 67.0]
         [196.0 195.0 114.0 321.0 386.0 115.0 69.0 189.0]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]],
                          mask =
         [[ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [False False False False False False False False]
         [False False False False False False False False]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]],
                    fill_value = 1e+20)

        >>> img['IMG'].data[4]
        Out[10]:
        masked_BaseColumn(data =
         [[-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [-- -- -- -- -- -- -- --]
         [67.0 61.0 67.0 176.0 99.0 72.0 79.0 88.0]
         [70.0 62.0 101.0 149.0 163.0 89.0 60.0 76.0]],
                          mask =
         [[ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [ True  True  True  True  True  True  True  True]
         [False False False False False False False False]
         [False False False False False False False False]],
                    fill_value = 1e+20)

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the ACA packets
    stop : CxoTimeLike
        Stop time for the ACA packets
    level0 : bool.
        Implies combine=True, adjust_time=True, calibrate=True
    combine : bool.
        If True, ACA subimages are combined to form a full image (depending on size),
        If False, ACA subimages are not combined, resulting in multiple rows for 6x6 and 8x8 images.
    adjust_time : bool
        If True, TIME is at the middle of the integration window.
        If False, TIME is the VCDU time in telemetry of the packet frame (combine=False)
        or the VCDU time of the first sub-image of the combined image (combine=True).
    calibrate : bool
        If True, pixel values will be 'value * imgscale / 32 - 50' and temperature values will
        be: 0.4 * value + 273.15
    blobs : bool or dict
        If set, data is assembled from MAUDE blobs. If it is a dictionary, it must be the
        output of maude.get_blobs ({'blobs': ... }).
    frames : bool or dict
        If set, data is assembled from MAUDE frames. If it is a dictionary, it must be the
        output of maude.get_frames ({'data': ... }).
    dtype : np.dtype. Optional.
        the dtype to use when creating the resulting table. This is useful to add columns
        including MSIDs that are present in blobs. If used with frames, most probably you will get
        and empty column. This option is intended to augment the default dtype. If a more
        restrictive dtype is used, a KeyError can be raised.
    **maude_kwargs
        Keyword args passed to maude

    Returns
    -------
    astropy.table.Table
    """
    if not blobs and not frames:
        frames = True

    if (not blobs and not frames) or (frames and blobs):
        raise ValueError("Specify one and only one of 'blobs' or 'frames'")

    if level0:
        adjust_time = True
        combine = True
        calibrate = True

    date_start, date_stop = CxoTime(start), CxoTime(stop)
    if date_stop - date_start > MAUDE_SINGLE_FETCH_LIMIT:
        raise ValueError(
            f"Requested {(date_stop - date_start).to_value('hr')} hr of telemetry. "
            f"Maximum allowed is {MAUDE_SINGLE_FETCH_LIMIT} at a time "
            "(see MAUDE_SINGLE_FETCH_LIMIT)."
        )

    stop_pad = 0 * u.s
    if adjust_time:
        stop_pad += 2.0 * u.s  # time will get shifted...
    if combine:
        stop_pad += 3.08 * u.s  # there can be trailing frames

    if frames:
        n = int(np.ceil((date_stop - date_start).sec / 300))
        dt = (date_stop - date_start) / n
        batches = [(date_start + i * dt, date_start + (i + 1) * dt) for i in range(n)]
        aca_packets = []
        for t1, t2 in batches:
            maude_result = (
                frames if (type(frames) is dict and "data" in frames) else None
            )
            raw_aca_packets = get_raw_aca_packets(
                t1, t2 + stop_pad, maude_result=maude_result, **maude_kwargs
            )
            packets = _get_aca_packets(
                raw_aca_packets,
                t1,
                t2,
                combine=combine,
                adjust_time=adjust_time,
                calibrate=calibrate,
                dtype=dtype,
            )
            aca_packets.append(packets)
        aca_data = vstack(aca_packets)
    else:
        maude_result = blobs if (type(blobs) is dict and "blobs" in blobs) else None
        merged_blobs = get_raw_aca_blobs(
            date_start, date_stop + stop_pad, maude_result=maude_result, **maude_kwargs
        )["blobs"]
        aca_packets = [
            [blob_to_aca_image_dict(b, i) for b in merged_blobs] for i in range(8)
        ]
        aca_data = _get_aca_packets(
            aca_packets,
            date_start,
            date_stop + stop_pad,
            combine=combine,
            adjust_time=adjust_time,
            calibrate=calibrate,
            blobs=True,
            dtype=dtype,
        )
        aca_data = aca_data[(aca_data["TIME"] >= start) & (aca_data["TIME"] < stop)]
    return aca_data


def _get_aca_packets(
    aca_packets,
    start,
    stop,
    combine=False,
    adjust_time=False,
    calibrate=False,
    blobs=False,
    dtype=None,
):
    """
    This is a convenience function that splits get_aca_packets for testing without MAUDE.

    Same arguments as get_aca_packets plus aca_packets, the raw ACA 225-byte packets.

    NOTE: This function has a side effect. It adds decom_packets to the input aca_packets.
    """
    start, stop = CxoTime(start), CxoTime(stop)  # ensure input is proper date

    if not blobs:
        # this adds TIME, MJF, MNF and VCDUCTR, which is already there in the blob dictionary
        aca_packets["decom_packets"] = [
            unpack_aca_telemetry(a) for a in aca_packets["packets"]
        ]
        for i in range(len(aca_packets["decom_packets"])):
            for j in range(8):
                aca_packets["decom_packets"][i][j]["TIME"] = aca_packets["TIME"][i]
                aca_packets["decom_packets"][i][j]["MJF"] = aca_packets["MJF"][i]
                aca_packets["decom_packets"][i][j]["MNF"] = aca_packets["MNF"][i]
                if "VCDUCTR" in aca_packets:
                    aca_packets["decom_packets"][i][j]["VCDUCTR"] = aca_packets[
                        "VCDUCTR"
                    ][i]
                aca_packets["decom_packets"][i][j]["IMGNUM"] = j

        aca_packets = [[f[i] for f in aca_packets["decom_packets"]] for i in range(8)]

    if combine:
        aca_packets = sum(
            [[_combine_aca_packets(g) for g in _group_packets(p)] for p in aca_packets],
            [],
        )
    else:
        aca_packets = [
            _combine_aca_packets([row]) for slot in aca_packets for row in slot
        ]
    table = _aca_packets_to_table(aca_packets, dtype=dtype)

    if calibrate:
        for k in ["TEMPCCD", "TEMPSEC", "TEMPHOUS", "TEMPPRIM"]:
            table[k] = 0.4 * table[k].astype(np.float32) + 273.15

    # IMGROW0/COL0 are actually the row/col of pixel A1, so we just copy them
    table["IMGROW_A1"] = table["IMGROW0"]
    table["IMGCOL_A1"] = table["IMGCOL0"]

    # if the image is embedded in an 8x8 image, row/col of the 8x8 shifts for sizes 4x4 and 6x6
    table["IMGROW0_8X8"] = table["IMGROW_A1"]
    table["IMGCOL0_8X8"] = table["IMGCOL_A1"]
    table["IMGROW0_8X8"][table["IMGTYPE"] < 4] -= 2
    table["IMGCOL0_8X8"][table["IMGTYPE"] < 4] -= 2

    # and the usual row/col needs to be adjusted only for 6x6 images
    table["IMGROW0"][table["IMGTYPE"] == 1] -= 1
    table["IMGCOL0"][table["IMGTYPE"] == 1] -= 1
    table["IMGROW0"][table["IMGTYPE"] == 2] -= 1
    table["IMGCOL0"][table["IMGTYPE"] == 2] -= 1

    table["INTEG"] = table["INTEG"] * 0.016

    # the time of an image is the time of its first subimage IMG_TIME, which is set 1.025 seconds
    # after the end of integration.
    # d_vcductr is the number of VCDU frames since the first subimage.
    d_vcductr = np.where(
        table["IMGTYPE"] > 4,
        table["IMGTYPE"] - 4,
        np.where(table["IMGTYPE"] == 2, 1, 0),
    )
    table["IMG_VCDUCTR"] = (table["VCDUCTR"] - 4 * d_vcductr) % MAX_VCDU
    img_time = table["TIME"] - 1.025 * d_vcductr
    table["END_INTEG_TIME"] = img_time - 1.025

    if adjust_time:
        # After this adjustment, TIME corresponds to the center of integration interval and
        # END_INTEG_TIME to the end:
        #     END_INTEG_TIME == TIME + INTEG / 2.0
        # See also https://cxc.harvard.edu/mta/ASPECT/Docs/aca_l0_icd.pdf section D.2.4.
        table["TIME"] = table["END_INTEG_TIME"] - table["INTEG"] / 2

    if calibrate:
        if "IMG" in table.colnames:
            table["IMG"] = (
                table["IMG"] * table["IMGSCALE"][:, np.newaxis, np.newaxis] / 32 - 50
            )

    table = table[(table["TIME"] >= start.secs) * (table["TIME"] < stop.secs)]
    if dtype:
        table = table[dtype.names]
    return table


def get_aca_images(start: CxoTimeLike, stop: CxoTimeLike, **kwargs):
    """
    Fetch ACA image telemetry

    Fetch ACA image telemetry from MAUDE and return it as an astropy Table. With the
    default settings and no additional kwargs, this calls `get_aca_packets()` in a
    configuration that uses MAUDE frames, combines image data, and sets the TIME
    associated with each image to the midpoint of the integration time during which that
    pixel data was collected (matches CXC L0 times). See `get_aca_packets()`.

    The 'IMG' column is always Nx8x8 and masked, where the mask is a per-pixel mask that
    indicates missing data for 4x4 or 6x6 images. The units of 'IMG' are DN.

    For queries including 4x4 data, the 'BGDRMS', 'TEMPCCD', 'TEMPHOUS', 'TEMPPRIM',
    'TEMPSEC', and 'BGDSTAT' columns will be masked since they are not present in the
    4x4 image data.

    There are three different specifiers of the image row/col location:

    - IMGROW0_8x8/IMGCOL0_8x8: the row/col of the lower-left pixel of the 8x8 masked
      image. This is generally the most useful.
    - IMGROW0/IMGCOL0: the row/col of the lower-left pixel of the actual 4x4, 6x6, or
      8x8 image data. For 6x6 this corresponds to the mouse-bitten corner pixel.
    - IMGROW_A1/IMGCOL_A1: the row/col of the A1 pixel in telemetry (see ACA EQ-spec).

    The full list of columns is::

               name          dtype  unit
      --------------------- ------- -----------
                       TIME float64 CXC seconds
                    VCDUCTR  uint32
                        MJF  uint32
                        MNF  uint32
                     IMGNUM  uint32
                    COMMCNT   uint8
                   COMMPROG   uint8
                    GLBSTAT   uint8
                    IMGFUNC  uint32
                    IMGTYPE   uint8
                   IMGSCALE  uint16
                    IMGROW0   int16
                    IMGCOL0   int16
                      INTEG float64 s
                     BGDAVG  uint16 DN
                     BGDRMS  uint16 DN
                    TEMPCCD float32 degC
                   TEMPHOUS float32 degC
                   TEMPPRIM float32 degC
                    TEMPSEC float32 degC
                    BGDSTAT   uint8
                   HIGH_BGD    bool
                   RAM_FAIL    bool
                   ROM_FAIL    bool
                 POWER_FAIL    bool
                   CAL_FAIL    bool
         COMM_CHECKSUM_FAIL    bool
                      RESET    bool
               SYNTAX_ERROR    bool
       COMMCNT_SYNTAX_ERROR    bool
      COMMCNT_CHECKSUM_FAIL    bool
            COMMPROG_REPEAT   uint8
                     IMGFID    bool
                    IMGSTAT   uint8
                  SAT_PIXEL    bool
                  DEF_PIXEL    bool
                 QUAD_BOUND    bool
                 COMMON_COL    bool
                 MULTI_STAR    bool
                    ION_RAD    bool
                  IMGROW_A1   int16
                  IMGCOL_A1   int16
                IMGROW0_8X8   int16
                IMGCOL0_8X8   int16
             END_INTEG_TIME float64
                   AAPIXTLM    str4
                   AABGDTYP    str4
                        IMG float64 DN
                IMG_VCDUCTR   int64

    This function can be used to fetch up to 5 days of ACA image telemetry at a time.
    If more is needed, you can set the module variable ``MAUDE_FETCH_LIMIT`` to a larger
    value. Internally, this function will fetch the data in intervals of
    ``MAUDE_SINGLE_FETCH_LIMIT``.

    Parameters
    ----------
    start
        timestamp, CxoTimeLike
    stop
        timestamp, CxoTimeLike.  stop - start cannot be greater than MAUDE_FETCH_LIMIT
    kwargs
        keyword args passed to get_aca_packets

    Returns
    -------
    astropy.table.Table
    """
    # This is strictly for testing, hence undocumented.
    set_times_metadata = kwargs.pop("set_times_metadata", False)

    start = CxoTime(start)
    stop = CxoTime(stop)
    if (stop - start) > MAUDE_FETCH_LIMIT:
        raise ValueError(
            f"stop - start cannot be greater than {MAUDE_FETCH_LIMIT}. "
            "Set module variable MAUDE_FETCH_LIMIT if needed."
        )
    maude_fetch_times = CxoTime.linspace(
        start, stop, step_max=MAUDE_SINGLE_FETCH_LIMIT - 1 * u.s
    )
    packet_stack = [
        get_aca_packets(
            start=istart,
            stop=istop,
            level0=True,
            **kwargs,
        )
        for istart, istop in itertools.pairwise(maude_fetch_times)
    ]
    out = vstack(packet_stack)
    if set_times_metadata:
        out.meta["times"] = maude_fetch_times

    # Remove mask from columns where no values are masked. IMG is excepted because
    # the mask reflects the presence of 4x4 or 6x6 images, not entirely missing data
    # per row.
    for col in out.itercols():
        if not np.any(col.mask) and col.name != "IMG":
            out[col.name] = col.data.data

    return out


######################
# blob-based functions
######################


def get_raw_aca_blobs(start, stop, maude_result=None, **maude_kwargs):
    """
    Fetch MAUDE blobs and group them according to the underlying 225-byte ACA packets.

    If the first minor frame in a group of four ACA packets is within (start, stop),
    the three following minor frames are included if present.

    Returns a dictionary with keys ['TIME', 'MNF', 'MJF', 'packets', 'flags'].
    These correspond to the minor frame time, minor frame count, major frame count,
    the list of packets, and flags returned by MAUDE respectively.

    This is to blobs what `get_raw_aca_packets` is to frames.

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the ACA blobs
    stop : CxoTimeLike
        Stop time for the ACA blobs
    maude_result
        the result of calling maude.get_blobs. Optional.
    **maude_kwargs
        keyword args passed to maude.get_frames()

    Returns
    -------
    dict
    {'blobs': [], 'names': np.array([]), 'types': np.array([])}
    """
    date_start, date_stop = CxoTime(start), CxoTime(stop)

    stop_pad = 1.5 * u.s  # padding at the end in case of trailing partial ACA packets

    if maude_result is None:
        maude_blobs = maude.get_blobs(
            start=date_start, stop=date_stop + stop_pad, **maude_kwargs
        )
    else:
        maude_blobs = maude_result

    aca_block_start = [
        ACA_SLOT_MSID_LIST[1][3]["sizes"] in [c["n"] for c in b["values"]]
        for b in maude_blobs["blobs"]
    ]

    names = maude_blobs["names"].copy()
    types = maude_blobs["types"].copy()
    if not np.any(aca_block_start):
        return {
            "blobs": [],
            "names": names,
            "types": types,
        }

    blobs = copy.deepcopy(maude_blobs["blobs"][np.argwhere(aca_block_start).min() :])
    if len(blobs) % 4:
        blobs = blobs[: -(len(blobs) % 4)]
    for b in blobs:
        b["values"] = {v["n"]: v["v"] for v in b["values"]}

    merged_blobs = []
    for i in range(0, len(blobs), 4):
        b = {"time": blobs[i]["time"]}
        # blobs are merged in reverse order so the frame counters are the one in the first blob.
        for j in range(4)[::-1]:
            b.update(blobs[i + j]["values"])
        merged_blobs.append(b)

    result = {
        "blobs": [b for b in merged_blobs if b["time"] < date_stop.secs],
        "names": names,
        "types": types,
    }
    return result


def blob_to_aca_image_dict(blob, imgnum, pea=1):
    """
    Assemble ACA image MSIDs from a blob into a dictionary.

    This does to blobs what unpack_aca_telemetry does to frames, but for a single image.

    Parameters
    ----------
    blob

    imgnum

    pea


    Returns
    -------

    """
    global_msids = ACA_MSID_LIST[pea]
    slot_msids = ACA_SLOT_MSID_LIST[pea][imgnum]
    glbstat_bits = np.unpackbits(np.array(blob[global_msids["status"]], dtype=np.uint8))
    comm_count_bits = np.unpackbits(
        np.array(blob[global_msids["cmd_count"]], dtype=np.uint8)
    )
    comm_prog_bits = np.unpackbits(
        np.array(blob[global_msids["cmd_progress"]], dtype=np.uint8)
    )
    result = {
        "TIME": float(blob["time"]),
        "MJF": int(blob["CVCMJCTR"]),
        "MNF": int(blob["CVCMNCTR"]),
        "VCDUCTR": int(blob["CVCDUCTR"]),
        "IMGNUM": imgnum,
        "IMGTYPE": int(blob[slot_msids["sizes"]]),
        # 'IMGFID': '1' == blob[slot_msids['fiducial_flag']],
        # 'IMGNUM': imgnum,
        # 'IMGFUNC': int(blob[slot_msids['image_function']]),
        # 'IMGSTAT': int(blob[slot_msids['image_status']]),
        "pixels": np.array(
            [int(blob[pixel]) for pixel in slot_msids["pixels"] if pixel in blob]
        ),
        "AAPIXTLM": blob.get(global_msids["pixel_telemetry_type"], "ORIG"),
        "AABGDTYP": blob.get(global_msids["dynamic_background_type"], "FLAT"),
        "INTEG": int(blob[global_msids["integration_time"]]),
        "GLBSTAT": int(blob[global_msids["status"]]),
        "HIGH_BGD": bool(glbstat_bits[0]),
        "RAM_FAIL": bool(glbstat_bits[1]),
        "ROM_FAIL": bool(glbstat_bits[2]),
        "POWER_FAIL": bool(glbstat_bits[3]),
        "CAL_FAIL": bool(glbstat_bits[4]),
        "COMM_CHECKSUM_FAIL": bool(glbstat_bits[5]),
        "RESET": bool(glbstat_bits[6]),
        "SYNTAX_ERROR": bool(glbstat_bits[7]),
        "COMMCNT": _packbits(comm_count_bits[1:6]),
        "COMMCNT_SYNTAX_ERROR": comm_count_bits[6],
        "COMMCNT_CHECKSUM_FAIL": comm_count_bits[7],
        "COMMPROG": int(blob[global_msids["cmd_progress_to_go"]]),
        "COMMPROG_REPEAT": _packbits(comm_prog_bits[6:8]),
    }

    if result["IMGTYPE"] in [0, 1, 4]:
        imgstat_bits = np.array(
            [
                blob[slot_msids["saturated_pixel"]],
                blob[slot_msids["defective_pixel"]],
                blob[slot_msids["quad_bound"]],
                blob[slot_msids["common_col"]],
                blob[slot_msids["multi_star"]],
                blob[slot_msids["ion_rad"]],
            ]
        ).astype(int)
        imgstat = int(_packbits(imgstat_bits))

        result.update(
            {
                "IMGFID": "1" == blob[slot_msids["fiducial_flag"]],
                "IMGNUM": imgnum,
                "IMGFUNC": int(blob[slot_msids["image_function"]]),
                "IMGSTAT": imgstat,
                "SAT_PIXEL": bool(int(blob[slot_msids["saturated_pixel"]])),
                "DEF_PIXEL": bool(int(blob[slot_msids["defective_pixel"]])),
                "QUAD_BOUND": bool(int(blob[slot_msids["quad_bound"]])),
                "COMMON_COL": bool(int(blob[slot_msids["common_col"]])),
                "MULTI_STAR": bool(int(blob[slot_msids["multi_star"]])),
                "ION_RAD": bool(int(blob[slot_msids["ion_rad"]])),
                "IMGROW0": int(blob[slot_msids["rows"]]),
                "IMGCOL0": int(blob[slot_msids["cols"]]),
                # 'IMGSCALE': float(blob[slot_msids['scale_factor']]),  # this is scale / 32
                "IMGSCALE": np.round(
                    float(blob[slot_msids["scale_factor"]]) * 32
                ).astype(int),
                "BGDAVG": int(blob[slot_msids["background_avg"]]),
            }
        )

    if result["IMGTYPE"] in [2, 5]:
        bgd_stat_pixels = [int(blob[pixel]) for pixel in slot_msids["bgd_stat_pixels"]]
        result.update(
            {
                "BGDRMS": int(blob[slot_msids["background_rms"]]),
                # 'TEMPCCD': float(blob[slot_msids['ccd_temperature']]),  # this is 0.4 * T
                # 'TEMPHOUS': float(blob[slot_msids['housing_temperature']]),  # this is 0.4 * T
                # 'TEMPPRIM': float(blob[slot_msids['primary_temperature']]),  # this is 0.4 * T
                # 'TEMPSEC': float(blob[slot_msids['secondary_temperature']]),  # this is 0.4 * T
                "TEMPCCD": np.round(
                    float(blob[slot_msids["ccd_temperature"]]) / 0.4
                ).astype(int),
                "TEMPHOUS": np.round(
                    float(blob[slot_msids["housing_temperature"]]) / 0.4
                ).astype(int),
                "TEMPPRIM": np.round(
                    float(blob[slot_msids["primary_temperature"]]) / 0.4
                ).astype(int),
                "TEMPSEC": np.round(
                    float(blob[slot_msids["secondary_temperature"]]) / 0.4
                ).astype(int),
                "BGDSTAT": int(_packbits(bgd_stat_pixels)),
                "BGDSTAT_PIXELS": bgd_stat_pixels,
            }
        )
    if result["IMGTYPE"] in [6, 7]:
        result.update(
            {
                "DIAGNOSTIC": (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )  # not set, and is not used downstream yet.
            }
        )

    if slot_msids["centroid_ang_y"] in blob:
        result.update(
            {
                "YAGS": float(blob[slot_msids["centroid_ang_y"]]),
                "ZAGS": float(blob[slot_msids["centroid_ang_z"]]),
            }
        )

    if "AOATTQT1" in blob:
        result["AOATTQT1"] = float(blob["AOATTQT1"])
    if "AOATTQT2" in blob:
        result["AOATTQT2"] = float(blob["AOATTQT2"])
    if "AOATTQT3" in blob:
        result["AOATTQT3"] = float(blob["AOATTQT3"])
    if "AOATTQT4" in blob:
        result["AOATTQT4"] = float(blob["AOATTQT4"])

    if "COBSRQID" in blob:
        result["COBSRQID"] = int(blob["COBSRQID"])

    return result
