"""
desisim.cosmology
=================

All cosmology related routines of desisim should be put here for consistency.
"""


import astropy.cosmology

# Fiducial cosmology is defined here
# It is LCDM , without neutrinos
try:
    fiducial_cosmology=astropy.cosmology.FlatLambdaCDM(H0=100,Om0=0.3)
except TypeError:
    #
    # TypeError will be thrown during Sphinx documentation builds
    # because the astropy.cosmology module is mocked up.
    #
    pass
