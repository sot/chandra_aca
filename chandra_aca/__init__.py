from .transform import *

__version__ = '3.10'


def test(*args, **kwargs):
    """
    Run py.test unit tests.
    """
    import testr
    return testr.test(*args, **kwargs)
