"""Generate bright-planet magnitude state data files for ``chandra_aca/data``.

This script queries or reuses cached ephemeris-derived magnitude samples, maps those
samples into ``MAG_ACTION_BINS``, and writes compact state intervals for each planet.
"""

from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from cheta.utils import state_intervals
from cxotime import CxoTime

from chandra_aca.planets import BRIGHT_PLANETS, MAG_ACTION_BINS, get_planet_horizons

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Year range for sampling (inclusive).
START_YEAR = 1999
STOP_YEAR = 2040

# Horizons sampling cadence used to generate yearly magnitude samples.
HORIZONS_STEP_SIZE = "1h"

# Horizons center code. '@399' is Earth geocenter.
HORIZONS_CENTER = "@399"

# Optional cache for yearly magnitude samples written/read by this script.
CACHE_DIR = Path(".")

# Output directory for packaged planet-state files.
OUTPUT_DIR = Path("chandra_aca/data")


def _get_yearly_mag_samples(planet: str, year: int) -> Table:
    """Get cached yearly magnitude samples for ``planet``, generating if needed."""
    sample_file = CACHE_DIR / f"planet_mag_{planet}_{year}_{HORIZONS_STEP_SIZE}.dat"

    if sample_file.exists():
        return Table.read(sample_file, format="ascii")

    samples = get_planet_horizons(
        planet,
        f"{year}:001",
        f"{year + 1}:001",
        step_size=HORIZONS_STEP_SIZE,
        center=HORIZONS_CENTER,
    )
    samples.meta.clear()
    samples.write(sample_file, format="ascii", overwrite=True)
    return samples


def make_planet_mag_states(planet: str) -> Table:
    """Make magnitude-action state intervals for one planet."""
    yearly_samples = [
        _get_yearly_mag_samples(planet, year)
        for year in range(START_YEAR, STOP_YEAR + 1)
    ]
    sample_table = vstack(yearly_samples)

    # Keep rows where magnitude is valid (not masked / finite).
    if hasattr(sample_table["mag"], "mask"):
        sample_table = sample_table[~sample_table["mag"].mask]
    else:
        sample_table = sample_table[np.isfinite(sample_table["mag"])]

    sample_table["magid"] = 0
    for mag_bin_id, mag_action_bin in enumerate(MAG_ACTION_BINS):
        in_bin = (sample_table["mag"] >= mag_action_bin["mag_start"]) & (
            sample_table["mag"] < mag_action_bin["mag_stop"]
        )
        sample_table["magid"][in_bin] = mag_bin_id

    states = state_intervals(CxoTime(sample_table["time"]).secs, sample_table["magid"])
    states["label"] = [MAG_ACTION_BINS[state["val"]]["label"] for state in states]
    states["mag_start"] = [
        MAG_ACTION_BINS[state["val"]]["mag_start"] for state in states
    ]
    states["mag_stop"] = [MAG_ACTION_BINS[state["val"]]["mag_stop"] for state in states]
    states.remove_column("val")
    return states


def main() -> None:
    """Generate and write state files for all configured bright planets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for planet in BRIGHT_PLANETS:
        states = make_planet_mag_states(planet)
        output_file = OUTPUT_DIR / f"planet_mag_states_{planet}.dat"
        states.write(output_file, format="ascii", overwrite=True)


if __name__ == "__main__":
    main()
