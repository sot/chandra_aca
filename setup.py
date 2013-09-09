import os
import nose
from setuptools import setup
setup(name='Chandra.ACA',
      author = 'Jean Connelly',
      description='Chandra Aspect Camera Tools',
      author_email = 'jconnelly@cfa.harvard.edu',
      py_modules = ['Chandra.ACA'],
      version='0.01',
      zip_safe=False,
      namespace_packages=['Chandra'],
      packages=['Chandra'],
      package_dir={'Chandra' : 'Chandra'},
      package_data={}
      )
