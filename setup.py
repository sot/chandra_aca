import sys

from chandra_aca import __version__

from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.args = []

    def run_tests(self):
        # Import here because outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.args)
        sys.exit(errno)


setup(name='chandra_aca',
      author='Jean Connelly, Tom Aldcroft',
      description='Chandra Aspect Camera Tools',
      author_email='jconnelly@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['chandra_aca'],
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      )
