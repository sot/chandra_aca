# Licensed under a 3-clause BSD style license - see LICENSE.rst

import matplotlib
import numpy as np
import pytest
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chandra_aca.plot import _plot_planets


def test_plot_planets_uses_stk_and_plots(monkeypatch):
    """Verify planets inside limits are plotted and STK helper arguments are used.

    This test confirms three behaviors:
    - planets farther than 2 degrees are skipped before helper calls,
    - helper calls use ephem_source='stk' with ccd_pad=100,
    - visible planet points trigger plot calls and a legend.
    """

    fig, ax = plt.subplots()
    helper_calls = []

    def fake_get_planet_angular_sep_with_venus_far(
        planet, ra, dec, time, observer_position
    ):
        if planet == "venus":
            return np.array([3.0, 3.0, 3.0])
        return np.array([1.0, 1.0, 1.0])

    def fake_get_planet_chandra_ccd_position_visible_points(
        planet, date, duration, att, ccd_pad, ephem_source
    ):
        helper_calls.append((planet, ephem_source, ccd_pad))
        return Table({"row": [0.0, 20.0], "col": [0.0, 20.0]})

    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_angular_sep",
        fake_get_planet_angular_sep_with_venus_far,
    )
    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_chandra_ccd_position",
        fake_get_planet_chandra_ccd_position_visible_points,
    )

    try:
        _plot_planets(
            ax,
            att=[0, 0, 0, 1],
            date0="2024:001:00:00:00",
            duration=1000,
            lim0=-580,
            lim1=590,
        )
    finally:
        plt.close(fig)

    assert helper_calls
    assert all(call[1] == "stk" for call in helper_calls)
    assert all(call[2] == 100 for call in helper_calls)
    assert all(call[0] != "venus" for call in helper_calls)

    assert len(ax.lines) == 9
    assert ax.get_legend() is not None


def test_plot_planets_no_points_no_legend(monkeypatch):
    """Verify no legend is drawn when all computed positions are outside limits."""

    fig, ax = plt.subplots()

    def fake_get_planet_angular_sep_small_separation(
        planet, ra, dec, time, observer_position
    ):
        return np.array([1.0, 1.0, 1.0])

    def fake_get_planet_chandra_ccd_position_outside_limits(
        planet, date, duration, att, ccd_pad, ephem_source
    ):
        return Table({"row": [1000.0, 1200.0], "col": [1000.0, 1200.0]})

    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_angular_sep",
        fake_get_planet_angular_sep_small_separation,
    )
    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_chandra_ccd_position",
        fake_get_planet_chandra_ccd_position_outside_limits,
    )

    try:
        _plot_planets(
            ax,
            att=[0, 0, 0, 1],
            date0="2024:001:00:00:00",
            duration=1000,
            lim0=-580,
            lim1=590,
        )
    finally:
        plt.close(fig)

    assert len(ax.lines) == 0
    assert ax.get_legend() is None


def test_plot_planets_stk_failure_propagates(monkeypatch):
    """Verify helper failures are not swallowed and propagate to the caller."""

    fig, ax = plt.subplots()

    def fake_get_planet_angular_sep_small_separation(
        planet, ra, dec, time, observer_position
    ):
        return np.array([1.0, 1.0, 1.0])

    def raise_stk_ephemeris_unavailable(*args, **kwargs):
        raise RuntimeError("STK ephemeris unavailable")

    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_angular_sep",
        fake_get_planet_angular_sep_small_separation,
    )
    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_chandra_ccd_position",
        raise_stk_ephemeris_unavailable,
    )

    try:
        with pytest.raises(RuntimeError, match="STK ephemeris unavailable"):
            _plot_planets(
                ax,
                att=[0, 0, 0, 1],
                date0="2024:001:00:00:00",
                duration=1000,
                lim0=-580,
                lim1=590,
            )
    finally:
        plt.close(fig)
