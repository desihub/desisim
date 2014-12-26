"""
Functions to generate simulated galaxy templates.

J. Moustakas
2014 Dec
"""

from os import getenv
from os.path import join
import specter.io as io
import numpy as np

class filter:
    """
    Define the filter class.  Could add additional functions
    to compute the filter effective wavelength, the Vega 
    zeropoints, etc.  Much of the code below is based on the 
    astLib.astSED package.
    """

    def __init__(self,filtername):
        """
        Initialize the filter class.  Reads and stores a single
        filter curve.  Also constructs an interpolation object 
        for easy-of-interpolation later.
        """

        from astropy.io import ascii
        from scipy import interpolate

        filterpath = join(getenv('DESISIM'),'data')
        filt = ascii.read(join(filterpath,filtername),
                          names=['lambda','pass'],format='basic')

        self.name = filtername
        self.wave = np.array(filt['lambda'])
        self.transmission = np.array(filt['pass'])
        self.interp = interpolate.interp1d(self.wave,self.transmission,kind='linear')

#       Calculate the AB zeropoint flux for this filter.
        ABwave = np.logspace(1.0,8.0,1E5) # Angstroms
        ABflux = 2.99792E18*3.631E-20*np.ones(1E5)/(ABwave**2)
        self.ABzpt = filter.get_flux(self,ABwave,ABflux)

    def get_flux(self,wave,flux):
        """
        Convolve an input SED (wave,flux) with the filter transmission 
        curve.  The output is the *unnormalized* flux.
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
    
def get_maggies(wave,flux,filter):
    """
    Convolve an input SED (wave,flux) with the filter transmission 
    curve and return maggies.
    """
    maggies = filter.get_flux(wave,flux)/filter.ABzpt
    return maggies

def read_templates(objtype,version):
    """
    Read the parent templates.
    """

    # Need to add error checking.
    templatepath = join(getenv('DESI_ROOT'),'spectro',
                        'templates',objtype+'_templates',version)
    templatefile = templatepath+'/'+objtype+'_templates_'+version+'.fits.gz'
    #print('Reading '+templatefile+')'
    models = io.read_simspec(templatefile)
    return models

def elgs():
    """
    Generate and write out a set of simulated ELG spectra.
    
    Inputs:
    - ...

    Returns ...
    """
    
    objtype = 'elg'
    version = 'v1.2'

    # Read the parent templates.
    models = read_templates(objtype,version)
    
    return models
