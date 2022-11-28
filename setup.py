# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
from pathlib import Path

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}


# Check for de432s.bsp JPL ephemeris file. This is not kept in the git repo
# but included in the distribution at chandra_aca/data/de432s.bsp (11 Mb).
ephem_file = Path("chandra_aca", "data", "de432s.bsp")
if "--version" not in sys.argv and not ephem_file.exists():
    import shutil
    import urllib.request

    url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp"
    with urllib.request.urlopen(url, timeout=60) as f_in, open(
        ephem_file, "wb"
    ) as f_out:
        shutil.copyfileobj(f_in, f_out)

setup(
    name="chandra_aca",
    author="Jean Connelly, Tom Aldcroft",
    description="Chandra Aspect Camera Tools",
    author_email="jconnelly@cfa.harvard.edu",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    zip_safe=False,
    packages=["chandra_aca", "chandra_aca.tests"],
    package_data={
        "chandra_aca.tests": [
            "data/*.txt",
            "data/*.dat",
            "data/*.fits.gz",
            "data/*.pkl",
            "data/*.ecsv",
        ],
        "chandra_aca": [
            "data/*.dat",
            "data/*.fits.gz",
            "data/star_probs/*.fits.gz",
            "data/de432s.bsp",
        ],
    },
    tests_require=["pytest"],
    cmdclass=cmdclass,
)
