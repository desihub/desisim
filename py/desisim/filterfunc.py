"""
Class for dealing with the filter curves in $DESISIM/data.  Includes
code for convolving an input SED with a specified filter.

Needs error checking.

J. Moustakas
2014 Dec
"""

import os
import numpy as np

class filterfunc():
    """
    Define the filterfunc class.  Could add additional functions
    to compute the filter effective wavelength, the Vega 
    zeropoints, etc.  Pieces of the code below are based on the 
    astLib.astSED package.
    """

    def __init__(self, filtername=None):
        """
        Initialize the filter class.  Reads and stores a single
        filter curve.  Also constructs an interpolation object 
        for ease-of-interpolation later.
        """

        from astropy.io import ascii
        from scipy.interpolate import interp1d

        if filtername is None:
            print('Need to specify FILTERNAME!')

        # raise an exception if FILTERNAME does not exist; could list
        # the available filters
        
        filterpath = os.path.join(os.getenv('DESISIM'),'data')
        filt = ascii.read(os.path.join(filterpath,filtername),
                          names=['lambda','pass'],format='basic')

        self.name = filtername
        self.wave = np.array(filt['lambda'])
        self.transmission = np.array(filt['pass'])
        self.interp = interp1d(self.wave,self.transmission,kind='linear')

        # calculate the AB zeropoint flux for this filter
        nwave = 100000
        ABwave = np.logspace(1.0,8.0,nwave) # Angstroms
        ABflux = 2.99792E18*3.631E-20*np.ones(nwave)/(ABwave**2)
        self.ABzpt = self.get_flux(ABwave,ABflux)

    def get_flux(self, wave, flux):
        """
        Convolve an input SED (wave,flux) with the filter transmission 
        curve.  The output is the *unnormalized* flux.  In general this 
        function will not be called on its own; it is used just to get
        the AB magnitude zeropoint for this filter.
        """
        lo = np.greater(wave,self.wave.min())
        hi = np.less(wave,self.wave.max())
        indx = np.logical_and(lo,hi)
        flux1 = flux[indx]
        wave1 = wave[indx]
        
        fluxinband = self.interp(wave1)*flux1
        flux = np.trapz(fluxinband*wave1,wave1)
        flux /= np.trapz(self.interp(wave1)*wave1,wave1)
        return flux
    
    def get_maggies(self, wave, flux):
        """
        Convolve an input SED (wave,flux) with the filter transmission 
        curve and return maggies.
        """
        maggies = self.get_flux(wave,flux)/self.ABzpt
        return maggies

