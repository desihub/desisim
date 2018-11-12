"""
desisim.eboss
===========

Functions and methods for mimicking eBOSS survey.

"""
from __future__ import division, print_function
import numpy as np
import os
import healpy
from pkg_resources import resource_filename

def read_sdss_footprint(nside):
    if not nside==16:
        raise ValueError('add eBOSS footprint for nside='+str(nside))
    fname=resource_filename('desisim','data/eboss_footprint_nside_16.txt')
    print('in read_sdss_footprint from file',fname)
    if not os.path.isfile(fname):
        print('eBOSS footprint file',fname)
        raise ValueError('file with eBOSS footprint does not exist')
    data=np.loadtxt(fname)
    pix=np.asarray(data[0], dtype=int)
    dens=data[1]
    return pix,dens

class FootprintEBOSS(object):
    """ Class to store eBOSS footprint and provide useful functions. """

    def __init__(self,nside=16):
        self.nside=nside
        print('in eboss_footprint constructor, nside',nside)
        data=read_sdss_footprint(self.nside)
        print('got data from file')
        self.eboss_pix=data[0]
        self.eboss_dens=data[1]

    def highz_density(self,ra,dec):
        pixs = healpy.ang2pix(self.nside, np.pi/2.-dec*np.pi/180., 
                    ra*np.pi/180.,nest=True)
        dens=np.zeros_like(ra)
        for p,d in zip(self.eboss_pix,self.eboss_dens):
            dens[pixs==p]=d
        return dens

def sdss_subsample(ra,dec,input_highz_density,eboss_footprint):
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
    density = eboss_footprint.highz_density(ra,dec)
    #density = sdss_highz_density(ra,dec)
    if debug: print(np.min(density),'<density<',np.max(density))
    fraction = density/input_highz_density
    if debug: print(np.min(fraction),'<fraction<',np.max(fraction))
    selection = np.where(np.random.uniform(size=N)<fraction)[0]
    if debug: print(len(selection),'selected out of',N)

    return selection

