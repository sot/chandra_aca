"""Configuration for chandra_aca"""

from astropy import config
from astropy.config import ConfigNamespace


class ConfigItem(config.ConfigItem):
    rootname = "chandra_aca"


class Conf(ConfigNamespace):
    """
    Configuration parameters for chandra_aca.transform.
    """

    transform_use_legacy_coeffs = ConfigItem(
        False,
        "Use legacy coefficients for ACA coordinate transforms instead of "
        "2020 ground/flight values",
    )


conf = Conf()
