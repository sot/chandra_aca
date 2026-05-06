# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from astropy.table import Table

from chandra_aca.plot import _plot_planets


class FakeAxes:
    def __init__(self):
        self.plot_calls = []
        self.legend_calls = []

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))

    def legend(self, *args, **kwargs):
        self.legend_calls.append((args, kwargs))


def test_plot_planets_uses_stk_and_plots(monkeypatch):
    ax = FakeAxes()
    helper_calls = []

    def fake_get_planet_angular_sep(planet, ra, dec, time, observer_position):
        if planet == "venus":
            return np.array([3.0, 3.0, 3.0])
        return np.array([1.0, 1.0, 1.0])

    def fake_get_planet_chandra_ccd_position(
        planet, date, duration, att, ccd_pad, ephem_source
    ):
        helper_calls.append((planet, ephem_source, ccd_pad))
        return Table({"row": [0.0, 20.0], "col": [0.0, 20.0]})

    monkeypatch.setattr("chandra_aca.plot.get_planet_angular_sep", fake_get_planet_angular_sep)
    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_chandra_ccd_position",
        fake_get_planet_chandra_ccd_position,
    )

    _plot_planets(
        ax,
        att=[0, 0, 0, 1],
        date0="2024:001:00:00:00",
        duration=1000,
        lim0=-580,
        lim1=590,
    )

    assert helper_calls
    assert all(call[1] == "stk" for call in helper_calls)
    assert all(call[2] == 100 for call in helper_calls)
    assert all(call[0] != "venus" for call in helper_calls)

    assert len(ax.plot_calls) == 9
    assert len(ax.legend_calls) == 1


def test_plot_planets_no_points_no_legend(monkeypatch):
    ax = FakeAxes()

    def fake_get_planet_angular_sep(planet, ra, dec, time, observer_position):
        return np.array([1.0, 1.0, 1.0])

    def fake_get_planet_chandra_ccd_position(
        planet, date, duration, att, ccd_pad, ephem_source
    ):
        return Table({"row": [1000.0, 1200.0], "col": [1000.0, 1200.0]})

    monkeypatch.setattr("chandra_aca.plot.get_planet_angular_sep", fake_get_planet_angular_sep)
    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_chandra_ccd_position",
        fake_get_planet_chandra_ccd_position,
    )

    _plot_planets(
        ax,
        att=[0, 0, 0, 1],
        date0="2024:001:00:00:00",
        duration=1000,
        lim0=-580,
        lim1=590,
    )

    assert len(ax.plot_calls) == 0
    assert len(ax.legend_calls) == 0


def test_plot_planets_stk_failure_propagates(monkeypatch):
    ax = FakeAxes()

    def fake_get_planet_angular_sep(planet, ra, dec, time, observer_position):
        return np.array([1.0, 1.0, 1.0])

    def fake_get_planet_chandra_ccd_position(
        planet, date, duration, att, ccd_pad, ephem_source
    ):
        raise RuntimeError("STK ephemeris unavailable")

    monkeypatch.setattr("chandra_aca.plot.get_planet_angular_sep", fake_get_planet_angular_sep)
    monkeypatch.setattr(
        "chandra_aca.plot.get_planet_chandra_ccd_position",
        fake_get_planet_chandra_ccd_position,
    )

    with pytest.raises(RuntimeError, match="STK ephemeris unavailable"):
        _plot_planets(
            ax,
            att=[0, 0, 0, 1],
            date0="2024:001:00:00:00",
            duration=1000,
            lim0=-580,
            lim1=590,
        )
