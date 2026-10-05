"""Read raw ACA image and OBC telemetry packets from a PEA dump

Read raw 224-byte ACA image packets from a PEA ASP_TLM.DAT dump and build the
``raw_aca_packets`` dict accepted by ``chandra_aca.maude_decom.get_aca_images``.
Read 60-byte OBC telemetry packets from the companion OBC_TLM.DAT dump. A complete
SWATS run (ASP_TLM, OBC_TLM and ACA_CMDS files bundled in a tar archive) is read with
``read_swats_tar``.

The dumps are bench IO-RAM dumps with no real timing, so TIME and the VCDU frame
counters are synthesized starting at zero and incrementing at the ACA readout cadence:
one 224-byte packet per 1.025 s update period, with the VCDU counter stepping by 4.
"""

import re
import tarfile
from pathlib import PurePath

import numpy as np
from astropy.table import MaskedColumn, Table

from chandra_aca import maude_decom

# Matches the 224-byte ACA image blocks (address 0200/0600), skipping the 512-byte RAM
# dumps and the 120-byte tails (address 0270/0670). A 224-byte data line is
# 112 words x "XXXX " = 112*4 + 111 spaces = 559 chars, inside the {558,562} window.
_ACA_PACKET_RE = re.compile(
    r"[\n\r]+# IO RAM Address set to: 0[26]00[\n\r]+([0-9A-F ]{558,562})[\n\r]"
)

# Matches the 60-byte OBC telemetry blocks (address 0100/0500) in an OBC_TLM.DAT dump,
# skipping the 448-byte RAM dumps at the same addresses. A 60-byte data line is
# 30 words x "XXXX " = 30*4 + 29 spaces = 149 chars, inside the {148,152} window.
_OBC_PACKET_RE = re.compile(
    r"[\n\r]+# IO RAM Address set to: 0[15]00[\n\r]+([0-9A-F ]{148,152})[\n\r]"
)

_DT_ACA = 1.025  # ACA readout period [s] -> one 224-byte packet per period
_VCDU_PER_PACKET = 4  # VCDU minor frames per ACA packet (counter step)

# Integration-cycle markers interleaved with the IO-RAM dumps. The cycle numbers are
# the "step" numbers used in the companion ACA_CMDS file, so they tie each packet to
# the test case being run when it was produced.
_CYCLE_RE = re.compile(r"# Integration Cycle # (\d+)")


def _read_text(source):
    """
    Return the text content of ``source``, a path or a (possibly binary) file object.
    """
    if hasattr(source, "read"):
        text = source.read()
        return text.decode() if isinstance(text, bytes) else text
    return open(source).read()


def _extract_packets(text, packet_re, size):
    """
    Return ``(packets, cycles)`` for the packets matching ``packet_re`` in ``text``.

    Each packet is tagged with the number of the last ``# Integration Cycle # N``
    marker preceding it in the dump (0 if there is none).
    """
    marker_pos = np.array([m.start() for m in _CYCLE_RE.finditer(text)])
    marker_num = np.array([int(m.group(1)) for m in _CYCLE_RE.finditer(text)])
    packets = []
    cycles = []
    for m in packet_re.finditer(text):
        packets.append(bytes.fromhex(m.group(1).replace(" ", "")))
        idx = np.searchsorted(marker_pos, m.start()) - 1
        cycles.append(int(marker_num[idx]) if idx >= 0 else 0)
    if not all(len(p) == size for p in packets):
        raise ValueError(f"expected all packets to be {size} bytes")
    return packets, cycles


def _read_aca_packets_(source):
    """
    Return ``(packets, cycles)`` for the 224-byte ACA packets in an ASP_TLM.DAT file.
    """
    return _extract_packets(_read_text(source), _ACA_PACKET_RE, 224)


def _read_obc_packets_(source):
    """
    Return ``(packets, cycles)`` for the 60-byte OBC packets in an OBC_TLM.DAT file.
    """
    return _extract_packets(_read_text(source), _OBC_PACKET_RE, 60)


def read_aca_packets(source):
    """
    Return the 224-byte ACA packets found in an ASP_TLM.DAT file as a dict for get_aca_images.

    ``source`` is a path or a file object. The dict includes a CYCLE entry with the
    integration-cycle number of each packet.
    """
    packets, cycles = _read_aca_packets_(source)
    return build_raw_aca_packets(packets, cycles=cycles)


def read_obc_packets(source):
    """
    Return the decommuted OBC telemetry packets found in an OBC_TLM.DAT file.

    Each packet is decoded with ``chandra_aca.maude_decom.unpack_obc_telemetry``,
    so this returns a list of dicts, each with a CYCLE key giving the packet's
    integration-cycle number. ``source`` is a path or a file object.
    """
    packets, cycles = _read_obc_packets_(source)
    return [
        maude_decom.unpack_obc_telemetry(packet) | {"CYCLE": cycle}
        for packet, cycle in zip(packets, cycles, strict=True)
    ]


def build_raw_aca_packets(packets, t0=0.0, vcdu0=0, cycles=None):
    """Build the raw_aca_packets dict for maude_decom.get_aca_images.

    TIME starts at ``t0`` (default 0) and steps by 1.025 s; VCDUCTR starts at ``vcdu0``
    (default 0) and steps by 4. MNF/MJF are derived from VCDUCTR. The VCDU counter rolls back to 0
    at 2**24. If ``cycles`` (integration-cycle number per packet) is given, it is
    included as CYCLE.
    """
    n = len(packets)
    vcductr = (
        (vcdu0 + _VCDU_PER_PACKET * np.arange(n)) % (maude_decom.MAX_VCDU + 1)
    ).astype(np.uint32)
    result = {
        "flags": 0,
        "packets": packets,
        "TIME": t0 + _DT_ACA * np.arange(n),
        "VCDUCTR": vcductr,
        "MNF": vcductr % (1 << 7),  # 128
        "MJF": vcductr // (1 << 7),
    }
    if cycles is not None:
        result["CYCLE"] = np.asarray(cycles, dtype=np.int32)
    return result


# A SWATS tar archive must contain exactly one member matching each of these patterns
# (matched case-insensitively against the member basename). This covers the bench's
# canonical *.DAT names as well as e.g. "swats_asp_tlm.txt"-style names.
_TAR_MEMBERS = {
    "aca_packets": re.compile("ASP_TLM", re.IGNORECASE),
    "obc_packets": re.compile("OBC_TLM", re.IGNORECASE),
    "cmds": re.compile("ACA_CMDS", re.IGNORECASE),
}


def read_swats_tar(path):
    """
    Read a SWATS run from a tar archive with the ASP_TLM, OBC_TLM and ACA_CMDS files.

    The archive must contain exactly one member whose basename matches each of
    "ASP_TLM", "OBC_TLM" and "ACA_CMDS" (case-insensitive substring).

    :param path: str or pathlib.Path
    :return: dict with keys

        - ``aca_packets``: raw_aca_packets dict (see :func:`read_aca_packets`)
        - ``obc_packets``: list of decommuted OBC telemetry dicts
          (see :func:`read_obc_packets`)
        - ``cmds``: list of test-case dicts (see :func:`parse_aca_cmds`)
    :raises ValueError: if ``path`` is not a tar archive or a member is
        missing/ambiguous
    """
    if not tarfile.is_tarfile(path):
        raise ValueError(
            f"{path} is not a tar archive. SWATS input must be a tar containing the "
            "ASP_TLM, OBC_TLM and ACA_CMDS files (bare ASP_TLM files are not accepted)."
        )

    with tarfile.open(path) as tar:
        members = [
            m
            for m in tar.getmembers()
            # skip the AppleDouble ("._*") members that macOS tar adds
            if m.isfile() and not PurePath(m.name).name.startswith("._")
        ]
        selected = {}
        for key, pattern in _TAR_MEMBERS.items():
            matches = [m for m in members if pattern.search(PurePath(m.name).name)]
            if not matches:
                names = sorted(m.name for m in members)
                raise ValueError(
                    f"SWATS tar {path} has no member matching {pattern.pattern!r}; "
                    f"members found: {names}"
                )
            if len(matches) > 1:
                names = sorted(m.name for m in matches)
                raise ValueError(
                    f"SWATS tar {path} has multiple members matching "
                    f"{pattern.pattern!r}: {names}"
                )
            selected[key] = matches[0]

        return {
            "aca_packets": read_aca_packets(tar.extractfile(selected["aca_packets"])),
            "obc_packets": read_obc_packets(tar.extractfile(selected["obc_packets"])),
            "cmds": parse_aca_cmds(_read_text(tar.extractfile(selected["cmds"]))),
        }


# Test-case block separator in an ACA_CMDS file, e.g.
# "########################################### Test case #5735"
_CMDS_CASE_RE = re.compile(r"^#{2,} *Test case *#?(\d+)?.*$", re.MULTILINE)

# (key, label, converter) for the "# <label> = <value>" comment lines of a test-case
# block. Labels are matched whitespace-tolerantly; missing lines give None.
_CMDS_FIELDS = [
    ("aca_temperature", "ACA Temperature", float),
    ("pred_aca_temp", r"Pred\. ACA Temp", float),
    ("int_time", "Int Time", float),
    ("obsid", "ObsID", int),
    ("target_quat", "Target Quat", None),
    ("est_flight_quat", "Est Flight Quat", None),
    ("search_box_hw", "Search Box HW", None),
    ("yang", "Yang", None),
    ("zang", "Zang", None),
    ("cat_mags", "Cat Mags", None),
    ("star_mags", "Star Mags", None),
    ("mag_limits", "Mag Limits", None),
    ("flight_time", "Flight Time", str),
]

_CMDS_STAR_LOCATION_RE = re.compile(
    r"^# *Star #(\d+) Location *= *(\S+) +(\S+)", re.MULTILINE
)


def _parse_cmds_block(block, test_case=None):
    """
    Parse one test-case block of an ACA_CMDS file into a dict.
    """
    result = {"test_case": test_case}
    for key, label, converter in _CMDS_FIELDS:
        match = re.search(rf"^# *{label} *=([^\n]*)", block, re.MULTILINE)
        if match is None:
            result[key] = None
            continue
        value = match.group(1).strip()
        result[key] = (
            converter(value) if converter else [float(v) for v in value.split()]
        )
    result["star_locations"] = {
        int(m.group(1)): (float(m.group(2)), float(m.group(3)))
        for m in _CMDS_STAR_LOCATION_RE.finditer(block)
    }
    # the leading integer of the command lines is the integration cycle at which the
    # block's commands execute
    match = re.search(r"^ *(\d+) +\S", block, re.MULTILINE)
    result["step"] = int(match.group(1)) if match else None
    return result


def parse_aca_cmds(text):
    """
    Parse the text of an ACA_CMDS file into a list of test-case dicts.

    Each ``#### Test case #NNNN`` block gives one dict with keys ``test_case``,
    ``step`` (the integration cycle at which the block's commands execute, matching
    the CYCLE numbers of :func:`read_aca_packets`/:func:`read_obc_packets`),
    ``aca_temperature``, ``pred_aca_temp``, ``int_time``, ``obsid``, ``target_quat``,
    ``est_flight_quat`` (lists of 4), ``search_box_hw``, ``yang``, ``zang``,
    ``cat_mags``, ``star_mags``, ``mag_limits`` (lists of 8), ``flight_time`` and
    ``star_locations`` ({slot: (row, col)}). Missing values are None. Text without
    block separators is parsed as a single block if it has recognizable keys.

    :param text: str, content of an ACA_CMDS file
    :return: list of dicts, in file (= execution) order
    """
    separators = list(_CMDS_CASE_RE.finditer(text))
    if not separators:
        if re.search(r"^# *(Target Quat|Yang) *=", text, re.MULTILINE):
            return [_parse_cmds_block(text)]
        return []
    blocks = []
    for match, next_match in zip(separators, separators[1:] + [None], strict=True):
        end = next_match.start() if next_match else len(text)
        test_case = int(match.group(1)) if match.group(1) else None
        blocks.append(_parse_cmds_block(text[match.end() : end], test_case))
    return blocks


# OBC packet j reports on the same 1.025 s cycle as ACA image packet index
# j + OBC_ACA_INDEX_OFFSET. Determined empirically from the swats_asp_tlm/swats_obc_tlm
# test dumps: all 78 per-slot track-onset transitions (IMGFUNC -> 1) occur at identical
# packet indices in the two streams (78/78 matched at offset 0, 0/78 at every other
# offset in [-4, 4]).
OBC_ACA_INDEX_OFFSET = 0

# packet-level and per-slot keys of the unpack_obc_telemetry dict (column order)
_OBC_GLOBAL_KEYS = [
    "INTEG",
    "GLBSTAT",
    "HIGH_BGD",
    "RAM_FAIL",
    "ROM_FAIL",
    "POWER_FAIL",
    "CAL_FAIL",
    "COMM_CHECKSUM_FAIL",
    "RESET",
    "SYNTAX_ERROR",
    "COMMPROG",
    "COMMPROG_REPEAT",
]
_OBC_SLOT_KEYS = [
    "IMGNUM",
    "IMGFID",
    "IMGFUNC",
    "SAT_PIXEL",
    "DEF_PIXEL",
    "QUAD_BOUND",
    "COMMON_COL",
    "MULTI_STAR",
    "ION_RAD",
    "YAG",
    "ZAG",
    "MAG",
]
_TIMING_KEYS = ["TIME", "VCDUCTR", "MJF", "MNF"]


def obc_telemetry_tables(obc_packets, aca_packets):
    """
    Build tables of OBC telemetry aligned to the ACA packet timeline.

    OBC packet ``j`` is assigned the TIME/VCDUCTR/MJF/MNF of ACA packet index
    ``j + OBC_ACA_INDEX_OFFSET``; packets that map outside the ACA packet range are
    dropped. In the ``slot`` table, YAG/ZAG/MAG are masked wherever the slot is not
    tracking (``IMGFUNC != 1``), where the OBC writes sentinel values instead of data.

    :param obc_packets: list of dicts from :func:`read_obc_packets`
    :param aca_packets: raw_aca_packets dict from :func:`read_aca_packets`
    :return: dict with keys

        - ``global``: Table with one row per aligned OBC packet (timing columns plus
          INTEG, GLBSTAT, the status flags, COMMPROG, COMMPROG_REPEAT)
        - ``slot``: Table with eight rows per aligned OBC packet (timing columns plus
          the per-slot image flags and YAG, ZAG, MAG)

    Both tables include a CYCLE column (the packets' integration-cycle numbers) when
    the OBC packets carry one.
    """
    n_aca = len(aca_packets["TIME"])
    aca_index = np.arange(len(obc_packets)) + OBC_ACA_INDEX_OFFSET
    ok = (aca_index >= 0) & (aca_index < n_aca)
    kept = [(j, i) for j, i in enumerate(aca_index) if ok[j]]
    idx = np.array([i for _, i in kept], dtype=int)
    timing = {key: np.asarray(aca_packets[key])[idx] for key in _TIMING_KEYS}

    have_cycles = bool(obc_packets) and "CYCLE" in obc_packets[0]
    if have_cycles:
        timing["CYCLE"] = np.array(
            [obc_packets[j]["CYCLE"] for j, _ in kept], dtype=np.int32
        )

    global_data = Table(timing)
    for key in _OBC_GLOBAL_KEYS:
        global_data[key] = [obc_packets[j][key] for j, _ in kept]

    slot_data = Table({key: np.repeat(timing[key], 8) for key in timing})
    for key in _OBC_SLOT_KEYS:
        slot_data[key] = [
            obc_packets[j]["image_data"][num][key] for j, _ in kept for num in range(8)
        ]
    tracking = np.asarray(slot_data["IMGFUNC"]) == 1
    for key in ["YAG", "ZAG", "MAG"]:
        slot_data[key] = MaskedColumn(slot_data[key], mask=~tracking)

    return {"global": global_data, "slot": slot_data}


def get_aca_images(path, **kwargs):
    raw = read_aca_packets(path)
    start, stop = raw["TIME"][0], raw["TIME"][-1] + 0.1
    images = maude_decom.get_aca_images(start, stop, raw_aca_packets=raw, **kwargs)
    images["TIME"].format = "%.3f"

    return images
