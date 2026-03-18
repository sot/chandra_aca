"""Configuration for chandra_aca"""

from astropy import config
from astropy.config import ConfigNamespace


class ConfigItem(config.ConfigItem):
    rootname = "chandra_aca"


class Conf(ConfigNamespace):
    """
    Configuration parameters for chandra_aca.transform.
    """

    t_aca_default = ConfigItem(
        35.0,
        "Default ACA temperature (degC) used by transform coordinate conversions "
        "when t_aca is not supplied.",
    )


conf = Conf()
