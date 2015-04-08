"""
Functions to generate simulated galaxy templates.

J. Moustakas
2014 Dec
"""

from os import getenv
from os.path import join
#import specter.io as io
import numpy as np
import matplotlib.pyplot as plt

light = 2.99792458E5

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
    
class emspectrum:
    """
    Class for building a complete nebular emission-line spectrum. 
    """

    def __init__(self, minwave=3650.0, maxwave=6700.0, linesigma=200.0, zshift=0.0):
        """
        Initialize the emission-line spectrum class with default values.
        """
        self.pixsize_kms = 20.0 # pixel size [km/s]
        self.pixsize = self.pixsize_kms/light/np.log(10) # pixel size [log-10 A]
        self.minwave = np.log10(minwave)
        self.maxwave = np.log10(maxwave)
        self.linesigma = linesigma
        self.zshift = zshift
        self.npix = (self.maxwave-self.minwave)/self.pixsize+1

    def wavelength(self):
        """
        Generate the default wavelength array.
        """
        wave = np.linspace(self.minwave,self.maxwave,self.npix) # log10 spacing
        return wave

    def linedata(self):
        from astropy.io import ascii
        from astropy.table import Table, Column

        nHeI = 0.0897 # adopted n(HeI)/n(HI) abundance ratio (see Kotulla+09)
        nlyc = 1.0 # fiducial number of Lyman-continuum photons (sec^-1)

        # read the file which contains the hydrogen and helium emissivities
        emfile = join(getenv('DESISIM'),'data','hydrogen_helium_emissivities.dat')
        emdata = ascii.read(emfile,comment='#',names=['name','wave','emissivity','transition'])
        nline = len(emdata)

        isha = np.where(emdata['name']=='Halpha')
        ishb = np.where(emdata['name']=='Hbeta')
            
        # initialize and then fill the line-information table
        line = Table(emdata.columns[0:2])
        line['flux'] = Column(dtype='f8',length=nline)  # integrated line-flux [erg/s]
        line['amp'] = Column(dtype='f8',length=nline)   # amplitude
        line['ratio'] = Column(emdata['emissivity']/emdata['emissivity'][ishb],
                               dtype='f8',length=nline) # ratio with respect to H-beta

        # calculate the luminosity of each line (in erg/s/A) given
        # N(Lyc); adopted conversions are from Kennicutt 1998 (but see
        # also Leitherer & Heckman 1995 and Hao+2011)
        for ii in range(nline):
            if 'HeI_' in line['name'][ii]:
                abundfactor = nHeI
            else:
                abundfactor = 1.0
            line['flux'][ii] = abundfactor*1.367E-12*nlyc* \
            emdata['emissivity'][ii]/emdata['emissivity'][isha]

        return line

    def emlines(self):
        """
        Build an emission-line spectrum
        """
        sigma = self.linesigma/light/np.log(10) # line-width [log-10 Angstrom]

        log10wave = em.wavelength()
        line = em.linedata()

        emspectrum = np.zeros(self.npix)
        for ii in range(len(line)):
            amp = line['flux'][ii]/line['wave'][ii]/np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii]*(1.0+self.zshift))
            print thislinewave, amp

            # [erg/s/cm2/A, rest]
            emspectrum += amp*np.exp(-0.5*(log10wave-thislinewave)**2/sigma**2)\
                          /(np.sqrt(2.0*np.pi)*sigma)

        return emspectrum

#def get_maggies(wave,flux,filter):
#    """
#    Convolve an input SED (wave,flux) with the filter transmission 
#    curve and return maggies.
#    """
#    maggies = filter.get_flux(wave,flux)/filter.ABzpt
#    return maggies

#def read_templates(objtype,version):
#    """
#    Read the parent templates.
#    """
#
#    # Need to add error checking.
#    templatepath = join(getenv('DESI_ROOT'),'spectro',
#                        'templates',objtype+'_templates',version)
#    templatefile = templatepath+'/'+objtype+'_templates_'+version+'.fits'
#    templatefile = templatepath+'/test.fits'
#    print('Reading '+templatefile+')'
#    #models = io.read_simspec(templatefile)
#    #return models
#    bob = 0
#    return bob
#
#def build_elgs():
#    """
#    Generate and write out a set of simulated ELG spectra.
#    Inputs:
#    - ...
#    Returns ...
#    """
#    
#    objtype = 'elg'
#    version = 'v1.2'
#
#    # Read the parent templates
#    #models = read_templates(objtype,version)
#    #return models

if __name__ == '__main__':

    em = emspectrum()
    log10wave = em.wavelength()
    emspectrum = em.emlines()

    plt.clf()
    plt.plot(10**log10wave,emspectrum)
    #plt.xlim([6400,6700])
    plt.show(block=False)
