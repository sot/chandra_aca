from chandra_aca import __version__

from setuptools import setup

try:
    from ska_test.setup_helper import cmdclass
except ImportError:
    cmdclass = {}

setup(name='chandra_aca',
      author='Jean Connelly, Tom Aldcroft',
      description='Chandra Aspect Camera Tools',
      author_email='jconnelly@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['chandra_aca', 'chandra_aca.tests'],
      package_data={'chandra_aca.tests': ['data/*.txt']},
      tests_require=['pytest'],
      cmdclass=cmdclass,
      )
