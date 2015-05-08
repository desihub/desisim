"""
Functions to help generate spectral templates of galaxies, quasars, and stars. 
"""

from __future__ import division, print_function

import os
import numpy as np

light = 2.99792458E5

class EMSpectrum():
    """
    Class for building a complete nebular emission-line spectrum.

    ToDo: Allow for AGN-like emission-line ratios.
    """
    def __init__(self):
        """
        Initialize the emission-line spectrum class.
        """
        from astropy.io import ascii
        from astropy.table import Table, Column, vstack

        # Read the file which contains the recombination and forbidden lines. 
        # Need to throw an exception if this file is not found!
        recombfile = os.path.join(os.getenv('DESISIM'),'data','recombination_lines.dat')
        forbidfile = os.path.join(os.getenv('DESISIM'),'data','forbidden_lines.dat')

        recombdata = ascii.read(recombfile,names=['name','wave','ratio'])
        forbiddata = ascii.read(forbidfile,names=['name','wave','ratio'])
        emdata = vstack([recombdata,forbiddata],join_type='exact')
        nline = len(emdata)

        # Initialize and populate the line-information table.
        self.line = Table()
        self.line['name'] = Column(emdata['name'],dtype='a15')
        self.line['wave'] = Column(emdata['wave'])
        self.line['ratio'] = Column(emdata['ratio'])
        self.line['flux'] = Column(dtype='f8',length=nline)  # integrated line-flux
        self.line['amp'] = Column(dtype='f8',length=nline)   # amplitude

    def normalize(self, oiiihbeta=-0.4, oiidoublet=0.75, siidoublet=1.3, 
                  oiiflux=None, hbetaflux=None):
        """
        Normalize the emission-line spectrum.

        This step involves three main pieces.  First, the ratio of the
        forbidden emission line-strengths relative to H-beta are derived
        using an input [OIII] 5007/H-beta ratio and the empirical
        relations documented elsewhere.  Second, the requested [OII]
        3726,29 and [SII] 6716,31 doublet ratios are imposed.  And
        finally the full emission-line spectrum is self-consistently
        normalized to *either* an integrated [OII] 3726,29 line-flux
        *or* an integrated H-beta line-flux.  Generally an ELG and LRG
        spectrum will be normalized using [OII] while the a BGS spectrum
        will be normalized using H-beta.

        ToDo: All the random seed to be fixed.
        """
        self.oiiihbeta = oiiihbeta   # *logarithmic* [OIII] 5007/H-beta line-ratio
        self.oiidoublet = oiidoublet # [OII] 3726/3729 doublet ratio (depends on density)
        self.siidoublet = siidoublet # [SII] 6716/6731 doublet ratio (depends on density)

        self.oiiidoublet = 2.8875    # [OIII] 5007/4959 doublet ratio (set by atomic physics)
        self.niidoublet = 2.93579    # [NII] 6584/6548 doublet ratio (set by atomic physics)

        if (oiiflux is None) and (hbetaflux is None):
            oiiflux = 1E-17

        self.oiiflux = oiiflux
        self.hbetaflux = hbetaflux

        line = self.line

        ishbeta = np.where(line['name']=='Hbeta')[0]

        # normalize [OIII] 4959, 5007 
        is4959 = np.where(line['name']=='[OIII]_4959')[0]
        is5007 = np.where(line['name']=='[OIII]_5007')[0]
        line['ratio'][is5007] = 10**self.oiiihbeta # NB: no scatter
        print(line['ratio'][is5007], self.oiiihbeta)

        line['ratio'][is4959] = line['ratio'][is5007]/self.oiiidoublet

        # normalize [NII] 6548,6584
        is6548 = np.where(line['name']=='[NII]_6548')[0]
        is6584 = np.where(line['name']=='[NII]_6584')[0]
        coeff = np.asarray([-0.53829,-0.73766,-0.20248])
        disp = 0.1 # dex

        line['ratio'][is6584] = 10**(np.polyval(coeff,self.oiiihbeta)+
                                          np.random.normal(0.0,disp))
        line['ratio'][is6548] = line['ratio'][is6584]/self.niidoublet

        # normalize [SII] 6716,6731
        is6716 = np.where(line['name']=='[SII]_6716')[0]
        is6731 = np.where(line['name']=='[SII]_6731')[0]
        coeff = np.asarray([-0.64326,-0.32967,-0.23058])
        disp = 0.1 # dex

        line['ratio'][is6716] = 10**(np.polyval(coeff,self.oiiihbeta)+
                                          np.random.normal(0.0,disp))
        line['ratio'][is6731] = line['ratio'][is6716]/self.siidoublet

        # normalize [NeIII] 3869
        is3869 = np.where(line['name']=='[NeIII]_3869')[0]
        coeff = np.asarray([1.0876,-1.1647])
        disp = 0.1 # dex

        line['ratio'][is3869] = 10**(np.polyval(coeff,self.oiiihbeta)+
                                          np.random.normal(0.0,disp))

        # normalize [OII] 3727, split into [OII] 3726,3729
        is3726 = np.where(line['name']=='[OII]_3726')[0]
        is3729 = np.where(line['name']=='[OII]_3729')[0]
        coeff = np.asarray([-0.52131,-0.74810,0.44351,0.45476])
        disp = 0.1 # dex

        oiihbeta = 10**(np.polyval(coeff,self.oiiihbeta)+ # [OII] 3727/Hbeta
                        np.random.normal(0.0,disp)) 

        factor1 = self.oiidoublet/(1.0+self.oiidoublet)
        factor2 = 1.0/(1.0+self.oiidoublet)
        line['ratio'][is3726] = factor1*oiihbeta
        line['ratio'][is3729] = factor2*oiihbeta
        
        ## Finally normalize the full spectrum to the desired integrated [OII]
        ## 3727 or H-beta flux.  Note that the H-beta normalization trumps the
        ## [OII] normalization!
        #if self.oiiflux is not None:
        #    factor = self.oiiflux/(1.0+self.oiidoublet)/line[is3729]['flux']
        #    for ii in range(len(line)):
        #        line['flux'][ii] *= factor
        #        line['amp'][ii] *= factor
        #        
        #if self.hbetaflux is not None:
        #    factor = self.hbetaflux/line[ishbeta]['flux']
        #    for ii in range(len(line)):
        #        line['flux'][ii] *= factor
        #        line['amp'][ii] *= factor

        return line

    def wavelength(self, minwave=3650.0, maxwave=6700.0, pixsize_kms=20.0):
        """
        Generate the default emission-line wavelength array.
        """
        self.pixsize_kms = pixsize_kms # pixel size [km/s]
        self.pixsize = self.pixsize_kms/light/np.log(10) # pixel size [log-10 A]
        self.minwave = np.log10(minwave)
        self.maxwave = np.log10(maxwave)
        self.npix = (self.maxwave-self.minwave)/self.pixsize+1

        wave = np.linspace(self.minwave,self.maxwave,self.npix) # log10 spacing
        return wave

    def emlines(self, linesigma=75.0, zshift=0.0):
        """
        Build an emission-line spectrum.
        """
        self.linesigma = linesigma
        self.zshift = zshift
        
        line = self.linedata()
        log10wave = self.wavelength()
        log10sigma = linesigma/light/np.log(10) # line-width [log-10 Angstrom]

        emspec = np.zeros(self.npix)
        for ii in range(len(line)):
            amp = line['flux'][ii]/line['wave'][ii]/np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii]*(1.0+self.zshift))
            line['amp'] = amp/(np.sqrt(2.0*np.pi)*log10sigma)  # [erg/s/A]

            # [erg/s/cm2/A, rest]
            emspec += amp*np.exp(-0.5*(log10wave-thislinewave)**2/log10sigma**2)\
                      /(np.sqrt(2.0*np.pi)*log10sigma)

        return emspec

def read_templates(objtype='elg', observed=False, continuum=False, kcorrections=False):
    """
    Returns the base templates for each objtype
    """
    from astropy.io import fits
    from astropy.table import Table
    from desispec.io.util import header2wave

    key = 'DESI_'+objtype.upper()+'_TEMPLATES'
    if key not in os.environ:
        raise ValueError('ERROR: $%s environment variable not set', key)

    objfile = os.getenv(key)

    # Optionally read the K-corrections.
    if kcorrections is True:
        kcorrfile = objfile.replace('templates_','templates_kcorr_')

    # Handle special cases for the ELG & BGS templates.
    if objtype.upper()=='ELG' or objtype.upper()=='BGS':
        if continuum is True:
            objfile = objfile.replace('templates_','continuum_templates_')
        if observed is True:
            objfile = objfile.replace('templates_','templates_obs_')

    if os.path.isfile(objfile) is False:
        raise ValueError('ERROR: Templates file %s not found', objfile)

    flux, hdr = fits.getdata(objfile, 0, header=True)
    meta = Table(fits.getdata(objfile, 1))
    wave = header2wave(hdr)

    if kcorrections is False:
        return flux, wave, meta
    else:
        kcorr = fits.getdata(kcorrfile, 0)
        return flux, wave, meta, kcorr
