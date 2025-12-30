"""
Make planet magnitude state data files for inclusion in chandra_aca.data directory.
"""

# %%
import numpy as np
from astropy.table import Table, vstack
from cheta.utils import state_intervals
from cxotime import CxoTime

from chandra_aca.planets import MAG_ACTION, get_planet_horizons


# %%
def make_planet_mag_states(planet):
    dat_stack = []
    for year in range(1999, 2041, 1):
        file = f"planet_mag_{planet}_{year}_1h.dat"
        try:
            dat = Table.read(file, format="ascii")
            # print(f"Read existing data file: {file}")
        except FileNotFoundError:
            # print(f"Generating data for year {year}")
            dat = get_planet_horizons(
                planet, f"{year}:001", f"{year + 1}:001", step_size="1h", center="@399"
            )
            # We don't need the metadata so let's just clear it.
            dat.meta.clear()
            # Save the data for future use
            dat.write(file, format="ascii", overwrite=True)
            # And append to the stack
        dat_stack.append(dat)
    dat = vstack(dat_stack)
    # filter to just the ones that have mag data, not masked
    # keep rows where mag is not masked/NaN/inf
    if hasattr(dat["mag"], "mask"):
        dat = dat[~dat["mag"].mask]
    else:
        dat = dat[np.isfinite(dat["mag"])]
    dat["magid"] = 0
    for bin_id, bin in MAG_ACTION.items():
        sel = (dat["mag"] >= bin["mag_start"]) & (dat["mag"] < bin["mag_stop"])
        dat["magid"][sel] = bin_id
    states = state_intervals(CxoTime(dat["time"]).secs, dat["magid"])
    # relabel the states with the information about the bins
    state_labels = [MAG_ACTION[s["val"]]["label"] for s in states]
    mag_start = [MAG_ACTION[s["val"]]["mag_start"] for s in states]
    mag_stop = [MAG_ACTION[s["val"]]["mag_stop"] for s in states]
    states["label"] = state_labels
    states["mag_start"] = mag_start
    states["mag_stop"] = mag_stop
    states.remove_column("val")
    return states


# %%
for planet in ["mars", "jupiter", "saturn", "venus"]:
    states = make_planet_mag_states(planet)
    states.write(
        f"chandra_aca/data/planet_mag_states_{planet}.dat",
        format="ascii",
        overwrite=True,
    )
