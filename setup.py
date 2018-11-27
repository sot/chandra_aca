# Licensed under a 3-clause BSD style license - see LICENSE.rst
from chandra_aca import __version__

from setuptools import setup

try:
    from testr.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='chandra_aca',
      author='Jean Connelly, Tom Aldcroft',
      description='Chandra Aspect Camera Tools',
      author_email='jconnelly@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['chandra_aca', 'chandra_aca.tests'],
      package_data={'chandra_aca.tests': ['data/*.txt', 'data/*.dat'],
                    'chandra_aca': ['data/*.dat', 'data/star_probs/*.fits.gz']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
