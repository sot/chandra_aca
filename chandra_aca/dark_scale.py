

DARK_SCALE_4C = 1.0 / 0.70  # Increase in dark current per 4 degC increase in T_ccd

def dark_temp_scale(t_ccd, t_ccd_ref=-19.0, scale_4c=None):
    """Return the multiplicative scale factor to convert a CCD dark map
    or dark current value from temperature ``t_ccd`` to temperature
    ``t_ccd_ref``::

      scale = scale_4c ** ((t_ccd_ref - t_ccd) / 4.0)

    In other words, if you have a dark current value that corresponds to ``t_ccd``
    and need the value at a different temperature ``t_ccd_ref`` then use the
    the following.  Do not be misled by the misleading parameter names.

      >>> from chandra_aca.dark_scale import dark_temp_scale
      >>> scale = dark_temp_scale(t_ccd, t_ccd_ref, scale_4c)
      >>> dark_curr_at_t_ccd_ref = scale * dark_curr_at_t_ccd

    The default value for ``scale_4c`` is 1.0 / 0.7.  It is written this way
    because the equation was previously expressed using 1 / scale_4c with a
    value of 0.7. This value is based on best global fit for dark current model
    in `plot_predicted_warmpix.py`.  This represents the multiplicative change
    in dark current for each 4 degC increase::

      >>> dark_temp_scale(t_ccd=-18, t_ccd_ref=-10, scale_4c=2.0)
      4.0

    :param t_ccd: actual temperature (degC)
    :param t_ccd_ref: reference temperature (degC, default=-19.0)
    :param scale_4c: increase in dark current per 4 degC increase (default=1.0 / 0.7)

    :returns: scale factor
    """
    if scale_4c is None:
        scale_4c = DARK_SCALE_4C

    return scale_4c ** ((t_ccd_ref - t_ccd) / 4.0)
