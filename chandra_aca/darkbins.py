"""Define bins for dark current distribution calculations"""
import numpy as np

# Even log spacing from exactly 2 to 2000, then additional bins up to 11000
x0 = 2
x1 = 2000
x2 = 11000
dx = 0.05

lx0 = np.log10(x0)
lx1 = np.log10(x1)
lx2 = np.log10(x2)

n_bins = np.round((lx1 - lx0) / dx)
dx = (lx1 - lx0) / n_bins

log_bins = np.arange(lx0, lx2, dx)
bins = 10 ** log_bins
bin_centers = 10 ** ((log_bins[1:] + log_bins[:-1]) / 2.0)
