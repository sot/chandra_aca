"""Read raw 224-byte ACA image packets from a PEA dump

Read raw 224-byte ACA image packets from a PEA ASP_TLM.DAT dump and build the
``raw_aca_packets`` dict accepted by ``chandra_aca.maude_decom.get_aca_images``.

ASP_TLM.DAT is a bench IO-RAM dump with no real timing, so TIME and the VCDU frame
counters are synthesized starting at zero and incrementing at the ACA readout cadence:
one 224-byte packet per 1.025 s update period, with the VCDU counter stepping by 4.
"""

import re

import numpy as np

from chandra_aca import maude_decom

# Matches the 224-byte ACA image blocks (address 0200/0600), skipping the 512-byte RAM
# dumps and the 120-byte tails (address 0270/0670). A 224-byte data line is
# 112 words x "XXXX " = 112*4 + 111 spaces = 559 chars, inside the {558,562} window.
_ACA_PACKET_RE = re.compile(
    r"[\n\r]+# IO RAM Address set to: 0[26]00[\n\r]+([0-9A-F ]{558,562})[\n\r]"
)

_DT_ACA = 1.025  # ACA readout period [s] -> one 224-byte packet per period
_VCDU_PER_PACKET = 4  # VCDU minor frames per ACA packet (counter step)


def _read_aca_packets_(path):
    """
    Return the list of 224-byte ACA packets (bytes) found in an ASP_TLM.DAT file.
    """
    text = open(path).read()
    packets = [bytes.fromhex(m.replace(" ", "")) for m in _ACA_PACKET_RE.findall(text)]
    if not all(len(p) == 224 for p in packets):
        raise ValueError("expected all packets to be 224 bytes")
    return packets


def read_aca_packets(path):
    """
    Return the 224-byte ACA packets found in an ASP_TLM.DAT file as a dict for get_aca_images.
    """
    return build_raw_aca_packets(_read_aca_packets_(path))


def build_raw_aca_packets(packets, t0=0.0, vcdu0=0):
    """Build the raw_aca_packets dict for maude_decom.get_aca_images.

    TIME starts at ``t0`` (default 0) and steps by 1.025 s; VCDUCTR starts at ``vcdu0``
    (default 0) and steps by 4. MNF/MJF are derived from VCDUCTR. The VCDU counter rolls back to 0
    at 2**24.
    """
    n = len(packets)
    vcductr = (
        (vcdu0 + _VCDU_PER_PACKET * np.arange(n)) % (maude_decom.MAX_VCDU + 1)
    ).astype(np.uint32)
    return {
        "flags": 0,
        "packets": packets,
        "TIME": t0 + _DT_ACA * np.arange(n),
        "VCDUCTR": vcductr,
        "MNF": vcductr % (1 << 7),  # 128
        "MJF": vcductr // (1 << 7),
    }


def get_aca_images(path, **kwargs):
    raw = read_aca_packets(path)
    start, stop = raw["TIME"][0], raw["TIME"][-1] + 0.1
    images = maude_decom.get_aca_images(start, stop, raw_aca_packets=raw, **kwargs)
    images["TIME"].format = "%.3f"

    return images
