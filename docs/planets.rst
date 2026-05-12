Planet Utilities
================

The planet utilities in ``chandra_aca.planets`` support three common tasks:

- Getting planet positions for Earth or Chandra observers.
- Estimating angular separation from a target pointing.
- Working with bright-planet mitigation states for operations workflows.

Ephemeris Sources
-----------------

Different helpers use different ephemeris paths depending on speed and fidelity needs:

- ``earth``: fast approximate observer geometry using built-in kernels.
- ``chandra``: Chandra observer geometry from mission ephemeris.
- ``chandra-horizons``: highest-fidelity check against JPL Horizons (network-based).

For Chandra observer geometry, ``get_planet_chandra()`` supports ``ephem_source='cxc'``
or ``ephem_source='stk'``.

Common Workflows
----------------

Planet position from Chandra (ECI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  from chandra_aca.planets import get_planet_chandra

  eci = get_planet_chandra("jupiter", "2024:120:12:00:00", ephem_source="stk")

Planet position on the ACA CCD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``get_planet_chandra_ccd_position()`` to get row/col samples across an observation
duration, optionally with a padded CCD boundary.

.. code-block:: python

  from chandra_aca.planets import get_planet_chandra_ccd_position

  positions = get_planet_chandra_ccd_position(
     "jupiter",
     date="2024:120:12:00:00",
     duration=4000,
     att=[0, 0, 0, 1],
     ccd_pad=100,
     ephem_source="stk",
  )

  # columns: time, row, col
  print(len(positions), positions.colnames)

Angular separation check
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  from chandra_aca.planets import get_planet_angular_sep

  sep_deg = get_planet_angular_sep(
     "jupiter",
     ra=120.0,
     dec=30.0,
     time="2024:120:12:00:00",
     observer_position="earth",
  )
  print(sep_deg)

Bright Planet States
--------------------

get_planet_mag_states
---------------------

Use ``get_planet_mag_states(planet, start, stop)`` to get intervals where a bright
planet is in a specific mitigation/action band. The function reads precomputed state
files in ``chandra_aca/data`` and returns only rows that overlap the requested time
interval.

Supported planets:

- ``jupiter``
- ``saturn``
- ``mars``
- ``venus``

Returned columns include:

- ``datestart`` and ``datestop``
- ``duration``
- ``tstart`` and ``tstop`` (seconds)
- ``label``
- ``mag_start`` and ``mag_stop``

Magnitude Action Bands
----------------------

The action bands are defined by the ``MAG_ACTION_BINS`` table in
``chandra_aca.planets``. The bins are half-open intervals
``[mag_start, mag_stop)`` and are ordered from most severe through least severe.

This table is used to compress ephemeris-derived magnitude time series into state
intervals so the packaged data stays small while preserving useful timing fidelity
at about 1-hour resolution.

.. list-table:: Bright-planet magnitude action bands
   :header-rows: 1

   * - Magnitude range
     - Label
   * - -30.0 to -5.0
     - obo too bright
   * - -5.0 to -2.9
     - full mitigation
   * - -2.9 to -2.0
     - partial mitigation
   * - -2.0 to 0.0
     - instrument notify
   * - 0.0 to 40.0
     - no action

Example
-------

.. code-block:: python

   from chandra_aca.planets import get_planet_mag_states

   states = get_planet_mag_states(
       "jupiter",
       start="2024:001",
       stop="2024:100",
   )

   print(states["datestart", "datestop", "label", "mag_start", "mag_stop"])

API Reference
-------------

Detailed API reference pages are available in the generated AutoAPI documentation
for ``chandra_aca.planets``.
