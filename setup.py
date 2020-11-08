# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup
from pathlib import Path

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}


# Check for de432s.bsp JPL ephemeris file. This is not kept in the git repo
# but included in the distribution at chandra_aca/data/de432s.bsp (11 Mb).
ephem_file = Path('chandra_aca', 'data', 'de432s.bsp')
if not ephem_file.exists():
    from astropy.utils.data import download_file
    import shutil
    url = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp'
    file = download_file(url, cache=True, show_progress=False)
    shutil.copy(file, ephem_file)


setup(name='chandra_aca',
      author='Jean Connelly, Tom Aldcroft',
      description='Chandra Aspect Camera Tools',
      author_email='jconnelly@cfa.harvard.edu',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      zip_safe=False,
      packages=['chandra_aca', 'chandra_aca.tests'],
      package_data={'chandra_aca.tests': ['data/*.txt', 'data/*.dat',
                                          'data/*.fits.gz', 'data/*.pkl'],
                    'chandra_aca': ['data/*.dat', 'data/*.fits.gz',
                                    'data/star_probs/*.fits.gz',
                                    'data/de432s.bsp']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
