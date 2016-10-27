__version__ = '0.8'

from .transform import *


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import ska_test
    return ska_test.test(*args, **kwargs)
