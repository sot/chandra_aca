.. _maude_decom:

===================
MAUDE ACA telemetry
===================

See :mod:`chandra_aca.maude_decom` for API details.

Telemetry Specification
=======================

Aspect telemetry is described in chapter 5 of the User's Manual. This module deals mostly with
Aspect telemetry, specified in section 5.3 of the user's manual and section 3.2.1.15.12 of the ACA
specification document EQ7-278 F.

In general, the telemetry data is specified in the following documents (available on the Aspect
Twiki page):

* MSFC-STD-1274B. MSFC HOSC Telemetry Format Standard
* MSFC-DOC-1949. MSFC HOSC Database Definitions

Timing
======

Timing in MAUDE Telemetry
-------------------------

What follows is a summary from the user's guide section 6, and EQ7-278 F section 3.2.1.7. Check
there for more details, especially Figures 6-1 to 6-4 in the user's manual and Figure 7 in
EQ7-278 F.

The ACA updates its output in regular 1.025 second periods that either begin or end at the time of
an RCTU science header pulse (which occur every 2.05 seconds). These are called update periods,
following the convention in the user manual section 6.1.

The ACA CCD operating cycle starts with a flush of charge from the CCD, followed by CCD integration,
CCD readout, and ends with an idle period. The start/end of the update period does not coincide with
the start/end of the CCD cycle. Instead, the end of the integration coincides with the start/end of
the update period. This is accomplished by adjusting the idle period.

The data from and integration period is available to the OBC at the end of the following update
period (EQ7-278 F Figure 7). That is 1.025 sec after the end of integration. This is the VCDU time
seen in MAUDE telemetry::

    TIME = END_INTEG_TIME + 1.025

.. image:: images/aca_timing_manual.png
  :width: 600

In the case of 6x6 and 8x8 images, the entire image cannot be updated in a single update period,
because Aspect pixel telemetry contains only eight pixels per update. 6x6 images take two update
periods, and 8x8 images take four. The end of the integration interval is the same for all the
sub-images, and corresponds to::

    END_INTEG_TIME = TIME - 1.025

When the choice of integration time causes the CCD cycle to last longer than the time it takes to
update a full image (1.025 seconds for a 4x4 image, 2.05 seconds for a 6x6 image or 4.1 seconds for
an 8x8 image) the most recent image is repeated until new pixel data is available.

Timing in level0 Data Products
------------------------------

The times for pixel telemetry in level0 data products is adjusted to coincide with the middle of the
integration interval::

    TIME = END_INTEG_TIME - INTEG / 2

This means that the difference between the time in telemetry and the time in the level0 data
products is::

    TIME<telem> - TIME<l0> = 1.025 + INTEG / 2

Global variables in this module
===============================

These include the following global variables

    * MAX_VCDU: the maximum possible VCDU frame counter value
    * MAX_MJF: the maximum possible major frame counter value
    * MAX_MNF: the maximum possible minor frame counter value
    * PIXEL_MAP: dict of np.array, with values mapping integer pixel indices to pixel string ID
    * PIXEL_MAP_INV: dict of dict, with values mapping pixel string ID to integer pixel indices.
    * PIXEL_MASK: dict of np.array. Values are boolean masks that apply to images of different sizes
    * ACA_MSID_LIST: dictionary of commonly-used ACA telemetry MSIDs.
    * ACA_SLOT_MSID_LIST: dictionary of ACA image telemetry MSIDs.

PIXEL_MAP contains maps between pixel indices and pixel IDm depending on the image size.
In the following tables, column index increases to the right and row index increases to the top
(c.f. ACA User Manual Figs 1.8 and 1.9 )::

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
