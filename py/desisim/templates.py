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

class filter():
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
    
class emspectrum():
    """
    Class for building a complete nebular emission-line spectrum. 
    """

    def __init__(self, minwave=3650.0, maxwave=6700.0, linesigma=75.0,
                 zshift=0.0, oiidoubletratio=0.75, oiiihbeta=-0.4):
        """
        Initialize the emission-line spectrum class with default values.
        """
        self.pixsize_kms = 20.0 # pixel size [km/s]
        self.pixsize = self.pixsize_kms/light/np.log(10) # pixel size [log-10 A]
        self.minwave = np.log10(minwave)
        self.maxwave = np.log10(maxwave)
        self.linesigma = linesigma
        self.zshift = zshift
        self.oiidoubletratio = oiidoubletratio
        self.oiiihbeta = oiiihbeta
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

        nii_doublet = 2.93579  # [NII] 6584/6548 doublet ratio
        oiii_doublet = 2.88750 # [OIII] 5007/4959 doublet ratio
        sii_doublet = 1.3      # [SII] 6716/6731 doublet ratio (depends on density)

        # read the file which contains the hydrogen and helium emissivities
        emfile = join(getenv('DESISIM'),'data','hydrogen_helium_emissivities.dat')
        emdata = ascii.read(emfile,comment='#',names=['name',
                    'wave','emissivity','transition'])
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

        # add in the forbidden lines, starting with [OIII] 5007 
        line.add_row(['[OIII]_5007',5006.842,0.0,0.0,0.0])
        line[-1]['ratio'] = 10**self.oiiihbeta # NB: no scatter
        line[-1]['flux'] = line[-1]['ratio']*line['flux'][ishb]

        line.add_row(['[OIII]_4959',4958.911,0.0,0.0,0.0])
        line[-1]['ratio'] = line[-2]['ratio']/oiii_doublet
        line[-1]['flux'] = line[-2]['flux']/oiii_doublet

        # add the [NII] 6548,6584 doublet (with no scatter)
        coeff = np.asarray([-0.20248,-0.73766,-0.53829])
        line.add_row(['[NII]_6584',6583.458,0.0,0.0,0.0])
        line[-1]['ratio'] = 10**np.polyval(coeff,self.oiiihbeta)
        line[-1]['flux'] = line[-1]['ratio']*line['flux'][ishb]

        line.add_row(['[NII]_6548',6548.043,0.0,0.0,0.0])
        line[-1]['ratio'] = line[-2]['ratio']/nii_doublet
        line[-1]['flux'] = line[-2]['flux']/nii_doublet

        # add the [SII] 6716,6731 doublet (with no scatter)
        coeff = np.asarray([-0.23058,-0.32967,-0.64326])
        line.add_row(['[SII]_6716',6716.440,0.0,0.0,0.0])
        line[-1]['ratio'] = 10**np.polyval(coeff,self.oiiihbeta)
        line[-1]['flux'] = line[-1]['ratio']*line['flux'][ishb]

        line.add_row(['[SII]_6731',6730.815,0.0,0.0,0.0])
        line[-1]['ratio'] = line[-2]['ratio']/sii_doublet
        line[-1]['flux'] = line[-2]['flux']/sii_doublet

        # add [NeIII] 3869 (with no scatter)
        coeff = np.asarray([-1.1647,1.0876])
        line.add_row(['[NeIII]_3869',3868.752,0.0,0.0,0.0])
        line[-1]['ratio'] = 10**np.polyval(coeff,self.oiiihbeta)
        line[-1]['flux'] = line[-1]['ratio']*line['flux'][ishb]

        # add [OII] 3727, split into the individual [OII] 3726,3729
        # lines according to the desired doublet ratio
        coeff = np.asarray([0.45476,0.44351,-0.74810,-0.52131])
        oiihbeta = 10**np.polyval(coeff,self.oiiihbeta) # [OII]/Hbeta
        oiiflux = oiihbeta*line['flux'][ishb]           # [OII] flux

        factor = self.oiidoubletratio/(1.0+self.oiidoubletratio)
        line.add_row(['[OII]_3726',3726.032,0.0,0.0,0.0])
        line[-1]['ratio'] = factor*oiihbeta
        line[-1]['flux'] = factor*oiiflux
        
        factor = 1.0/(1.0+self.oiidoubletratio)
        line.add_row(['[OII]_3729',3728.814,0.0,0.0,0.0])
        line[-1]['ratio'] = factor*oiihbeta
        line[-1]['flux'] = factor*oiiflux

        return line

    def emlines(self):
        """
        Build an emission-line spectrum
        """
        log10sigma = self.linesigma/light/np.log(10) # line-width [log-10 Angstrom]

        log10wave = em.wavelength()
        line = em.linedata()

        emspectrum = np.zeros(self.npix)
        for ii in range(len(line)):
            amp = line['flux'][ii]/line['wave'][ii]/np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii]*(1.0+self.zshift))
            line['amp'] = amp/(np.sqrt(2.0*np.pi)*log10sigma)  # [erg/s/A]

            # [erg/s/cm2/A, rest]
            emspectrum += amp*np.exp(-0.5*(log10wave-thislinewave)**2/log10sigma**2)\
                          /(np.sqrt(2.0*np.pi)*log10sigma)

        return emspectrum

def get_maggies(wave,flux,filter):
    """
    Convolve an input SED (wave,flux) with the filter transmission 
    curve and return maggies.
    """
    maggies = filter.get_flux(wave,flux)/filter.ABzpt
    return maggies

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

    import sys
    from optparse import OptionParser

    light = 2.99792458E5

    # In the Monte Carlo simulation linesigma, oiidoubletratio,
    # oiiihbeta should be drawn from a uniform (or log-normal)
    # distribution

    # parse the simulation parameters from the command line or choose a
    # reasonable set of default values
    parser = OptionParser(usage='%prog',description='This is a neat piece of code.')
    parser.add_option('--nmodel', default=3, type=long, action="store",
                      help='Number of model (template) spectra to generate')
    parser.add_option('--oiiihbeta_range', type=float, nargs=2, action="store",
                      help='Minimum and maximum logarithmic [OIII]/Hbeta ratios to consider (default "-0.5 0.0")')
    opt, args = parser.parse_args()

    if opt.oiiihbeta_range is None:
        oiiihbeta_range = (-0.5,0.0)

    # draw random values
    oiiihbeta = np.random.uniform(oiiihbeta_range[0],oiiihbeta_range[1],nmodel)

    # build a default spectrum in order to initialize the
    # emission-line data and output wavelength array
    em = emspectrum()
    line = em.linedata()
    log10wave = em.wavelength()

    for ii in range(nmodel):
        em = emspectrum(linesigma=75.0, oiidoubletratio=0.75, oiiihbeta=oiiihbeta[ii])
        emspectrum = em.emlines()

        plt.clf()
        plt.plot(10**log10wave,emspectrum)
        #plt.xlim([6500,6800])
        #plt.ylim([0,1E-12])
        plt.show(block=True)
