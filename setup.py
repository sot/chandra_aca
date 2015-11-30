from chandra_aca import __version__

from setuptools import setup

setup(name='chandra_aca',
      author='Jean Connelly, Tom Aldcroft',
      description='Chandra Aspect Camera Tools',
      author_email='jconnelly@cfa.harvard.edu',
      version=__version__,
      zip_safe=False,
      packages=['chandra_aca'],
      )
