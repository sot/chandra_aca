# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .transform import *

__version__ = '4.23.1'


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
