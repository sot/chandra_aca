import ska_helpers.docs

# Get variables from the ska_helpers.docs.conf module that are not private, modules, or
# classes and inject into this namespace. This provides the base configuration for the
# Sphinx documentation.
globals().update(
    ska_helpers.docs.get_conf_module_vars(
        project="chandra_aca",
        author="Tom Aldcroft",
    )
)

# Add any custom configuration here
