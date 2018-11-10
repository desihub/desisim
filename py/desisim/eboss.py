"""
desisim.eboss
===========

Functions and methods for mimicking eBOSS survey.

"""
from __future__ import division, print_function
import numpy as np

def sdss_subsample(ra,dec,input_highz_density):
    """ Downsample input list of angular positions based on SDSS footprint
        and input density of high-z quasars (z>2.15).
    Args:
        ra (ndarray): Right ascension (degrees)
        dec (ndarray): Declination (degrees)
        input_highz_density (float): Input density of high-z quasars per sq.deg.
    Returns:
        selection (ndarray): mask to apply to downsample input list
    """
    debug=True
    # figure out expected SDSS density, in quasars / sq.deg., above z=2.15
    N=len(ra)
    density = sdss_highz_density(ra,dec)
    if debug: print(np.min(density),'<density<',np.max(density))
    fraction = density/input_highz_density
    if debug: print(np.min(fraction),'<fraction<',np.max(fraction))
    selection = np.where(np.random.uniform(size=N)<fraction)[0]
    if debug: print(len(selection),'selected out of',N)

    return selection

def sdss_highz_density(ra,dec):
    """ Density of quasars in BOSS+eBOSS with z>2.15 in particular area.
    Args:
        ra (ndarray): Right ascension (degrees)
        dec (ndarray): Declination (degrees)
    Returns:
        density (ndarray): density of quasars per sq.deg. with z > 2.15
    """
    # here we should actually read the HEALPix mask from DRQ
    # for now hack something to test densities for pixel 333
    density=np.ones_like(ra)*16.0
    density[ra<160]=24.0
    density[ra<159]=0.0
    density[ra>161]=0.0
    density[dec<35]=0.0
    density[dec>36]=0.0

    return density

