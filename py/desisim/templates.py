"""
Functions to help generate spectral templates of galaxies, quasars, and stars. 
"""

from __future__ import division, print_function

import os
import numpy as np

LIGHT = 2.99792458E5

class ELG():
    """
    Class for building a complete spectrum of an ELG.
    """
    def __init__(self, nmodel=10, minwave=3600.0, maxwave=1E4,
                 pixsize=2.0):
        """
        Initialize the ELG class.

        Read the base (continuum) templates and initialize the filter profiles.

        pixsize - Angstrom/pixel - currently only supports linear output wavelength!
        minwave, maxwave - desired output wavelength array
        """
        from desisim.filter import filter
        from desisim.templates import read_base_templates

        self.nmodel = nmodel

        # Read the rest-frame continuum basis spectra.  Need an exception if
        # this fails!
        baseflux, basewave, basemeta = read_base_templates(objtype='ELG',continuum=True)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta
        self.nbase = len(basemeta)

        # Initialize the filter profiles. Need an exception if reading these
        # fails!
        self.gfilt = filter(filtername='decam_g.txt')
        self.rfilt = filter(filtername='decam_r.txt')
        self.zfilt = filter(filtername='decam_z.txt')
        self.w1filt = filter(filtername='wise_w1.txt')

        # Initialize the output wavelength array
        self.minwave = minwave
        self.maxwave = maxwave
        self.pixsize = pixsize
        self.npix = (self.maxwave-self.minwave)/self.pixsize+1
        self.wave = np.linspace(self.minwave,self.maxwave,self.npix) # linear spacing

    def make_templates(self, redshift_minmax=(0.6,1.6), rmag_minmax=(21.0,23.5),
                       oiiihbeta_minmax=(-0.5,0.0), oiidoublet_meansig=(0.73,0.05),
                       linesigma_meansig=(80.0,20.0), minoiiflux=1e-18):
        """
        Build Monte Carlo sets of ELG templates.

        Need error checking on the input values (e.g., min is less than max, etc.).

        Need a way to alter the grz color cuts.
        """
        from astropy.table import Table, Column
        from desisim.templates import EMSpectrum
        from desispec.interpolation import resample_flux
        from imaginglss.analysis import cuts

        EM = EMSpectrum()
       
        # Initialize the output flux array and metadata Table.
        outflux = np.ndarray([self.nmodel,self.npix])
        meta = dict()
        meta['TEMPLATEID'] = np.zeros(self.nmodel,dtype='i4')
        meta['REDSHIFT'] = np.zeros(self.nmodel,dtype='f4')
        meta['GMAG'] = np.zeros(self.nmodel,dtype='f4')
        meta['RMAG'] = np.zeros(self.nmodel,dtype='f4')
        meta['ZMAG'] = np.zeros(self.nmodel,dtype='f4')
        meta['OIIFLUX'] = np.zeros(self.nmodel,dtype='f4')
        meta['EWOII'] = np.zeros(self.nmodel,dtype='f4')
        meta['OIIIHBETA'] = np.zeros(self.nmodel,dtype='f4')
        meta['OIIDOUBLET'] = np.zeros(self.nmodel,dtype='f4')
        meta['LINESIGMA_KMS'] = np.zeros(self.nmodel,dtype='f4')
        meta['D4000'] = np.zeros(self.nmodel,dtype='f4')

        nobj = 0
        nchunk = min(self.nmodel,5)

        while nobj<=self.nmodel:
            # Choose a random subset of the base templates
            chunkindx = np.random.randint(0,self.nbase-1,nchunk)

            # Assign uniform redshift and r-magnitude distributions.
            redshift = np.random.uniform(redshift_minmax[0],redshift_minmax[1],nchunk)
            rmag = np.random.uniform(rmag_minmax[0],rmag_minmax[1],nchunk)

            # Assume the emission-line priors are uncorrelated.
            oiiihbeta = np.random.uniform(oiiihbeta_minmax[0],oiiihbeta_minmax[1],nchunk)
            oiidoublet = np.random.normal(oiidoublet_meansig[0],oiidoublet_meansig[1],nchunk)
            linesigma = np.random.normal(linesigma_meansig[0],linesigma_meansig[1],nchunk)

            d4000 = self.basemeta['D4000'][chunkindx]
            ewoii = 10.0**(np.polyval([1.1074,-4.7338,5.6585],d4000)+ # rest-frame, Angstrom
                           np.random.normal(0.0,0.3)) 

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                # Add the continuum and emission-line spectra with the right [OII] flux. 
                oiiflux = self.basemeta['OII_CONTINUUM'][iobj]*ewoii[ii] # [erg/s/cm2]
                emflux1, emwave, emline = EM.spectrum(linesigma=linesigma[ii],
                                                      oiidoublet=oiidoublet[ii],
                                                      oiiihbeta=oiiihbeta[ii],
                                                      oiiflux=oiiflux)
                emflux = resample_flux(self.basewave,emwave,emflux1) # [erg/s/cm2/A, rest]

                flux1 = self.baseflux[iobj,:] + emflux # [erg/s/cm2/A @10pc]

                # Does this object passes the ELG color and [OII] flux cuts?
                zwave = self.basewave*(1.0+redshift[ii])

                rnorm = 10.0**(-0.4*rmag[ii])/self.rfilt.get_maggies(zwave,flux1)
                flux = flux1*rnorm # [erg/s/cm2/A, @redshift[ii]]

                rflux = 10.0**(-0.4*(rmag[ii]-22.5))                 # [nanomaggies]
                gflux = self.gfilt.get_maggies(zwave,flux)*10**(0.4*22.5) # [nanomaggies]
                zflux = self.zfilt.get_maggies(zwave,flux)*10**(0.4*22.5) # [nanomaggies]
                zoiiflux = oiiflux*rnorm

                oiiflux = self.basemeta['OII_CONTINUUM'][iobj]*ewoii[ii]*rnorm # [erg/s/cm2]
        
                grzmask = cuts.Fluxes.ELG(gflux=gflux,rflux=rflux,zflux=zflux)
                oiimask = [oiiflux>minoiiflux]

                print(ii, iobj, nobj)
                if all(grzmask) and all(oiimask):
                    outflux[nobj,:] = resample_flux(self.wave,self.basewave,flux) # [erg/s/cm2/A]

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(gflux)
                    meta['RMAG'][nobj] = rmag[ii]
                    meta['ZMAG'][nobj] = -2.5*np.log10(zflux)
                    meta['OIIFLUX'][nobj] = oiiflux
                    meta['EWOII'][nobj] = ewoii[ii]
                    meta['OIIIHBETA'][nobj] = oiiihbeta[ii]
                    meta['OIIDOUBLET'][nobj] = oiidoublet[ii]
                    meta['LINESIGMA_KMS'][nobj] = linesigma[ii]
                    meta['D4000'][nobj] = d4000[ii]

                    nobj = nobj+1
                    
            if nobj<=self.nmodel: break

        # Test: write out
        from astropy.io import fits
        from desispec.io import util
        from desisim.obs import _dict2ndarray
        
        outfile = 'elg-test.fits'

        header = dict(
            CRVAL1 = (self.minwave, 'Starting wavelength [Angstrom]'),
            CDELT1 = (self.pixsize, 'Wavelength step [Angstrom]'),
            AIRORVAC = ('vac', 'Vacuum wavelengths'),
            LOGLAM = (0, 'linear wavelength steps, not log10')
            )
        hdr = util.fitsheader(header)
        fits.writeto(outfile,outflux.astype(np.float32),header=hdr,clobber=True)

        comments = dict(
            TEMPLATEID  = 'template ID',
            #OBJTYPE     = 'Object type (ELG, LRG, QSO, BGS, STD, STAR)',
            REDSHIFT    = 'object redshift',
            OIIFLUX     = '[OII] flux [erg/s/cm2]',
            )

        units = dict(
            OIIFLUX       = 'erg/s/cm2',
            )

        outmeta = _dict2ndarray(meta,columns=('TEMPLATEID','REDSHIFT','OIIFLUX'))
        util.write_bintable(outfile, outmeta, header=None, extname='METADATA',
                       comments=comments, units=units)
        return outflux, meta
        
class EMSpectrum():
    """
    Class for building a nebular emission-line spectrum.

    ToDo: Allow for AGN-like emission-line ratios.
    """
    def __init__(self, minwave=3650.0, maxwave=7075.0,
                 pixsize_kms=20.0):
        """
        Initialize the emission-line spectrum class.
        """
        from astropy.io import ascii
        from astropy.table import Table, Column, vstack

        # Initialize the default wavelength array
        self.pixsize_kms = pixsize_kms # pixel size [km/s]
        self.pixsize = self.pixsize_kms/LIGHT/np.log(10) # pixel size [log-10 A]
        self.minwave = np.log10(minwave)
        self.maxwave = np.log10(maxwave)
        self.npix = (self.maxwave-self.minwave)/self.pixsize+1
        self.wave = np.linspace(self.minwave,self.maxwave,self.npix) # log10 spacing

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
        self.line['flux'] = Column(np.ones(nline),dtype='f8')  # integrated line-flux
        self.line['amp'] = Column(np.ones(nline),dtype='f8')   # amplitude

    def spectrum(self, oiiihbeta=-0.4, oiidoublet=0.75, siidoublet=1.3,
                 linesigma=75.0, zshift=0.0, oiiflux=None, hbetaflux=None):
        """
        Build an emission-line spectrum.

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

        oiiihbeta  - *logarithmic* [OIII] 5007/H-beta line-ratio
        oiidoublet - [OII] 3726/3729 doublet ratio (depends on density)
        siidoublet - [SII] 6716/6731 doublet ratio (depends on density)

        Note that the H-beta normalization trumps the [OII] normalization! 

        ToDo: All the random seed to be fixed.
        ToDo: Add the nebular continuum.
        """
        oiiidoublet = 2.8875    # [OIII] 5007/4959 doublet ratio (set by atomic physics)
        niidoublet = 2.93579    # [NII] 6584/6548 doublet ratio (set by atomic physics)

        line = self.line
        nline = len(line)

        # normalize [OIII] 4959, 5007 
        is4959 = np.where(line['name']=='[OIII]_4959')[0]
        is5007 = np.where(line['name']=='[OIII]_5007')[0]
        line['ratio'][is5007] = 10**oiiihbeta # NB: no scatter

        line['ratio'][is4959] = line['ratio'][is5007]/oiiidoublet

        # normalize [NII] 6548,6584
        is6548 = np.where(line['name']=='[NII]_6548')[0]
        is6584 = np.where(line['name']=='[NII]_6584')[0]
        coeff = np.asarray([-0.53829,-0.73766,-0.20248])
        disp = 0.1 # dex

        line['ratio'][is6584] = 10**(np.polyval(coeff,oiiihbeta)+
                                          np.random.normal(0.0,disp))
        line['ratio'][is6548] = line['ratio'][is6584]/niidoublet

        # normalize [SII] 6716,6731
        is6716 = np.where(line['name']=='[SII]_6716')[0]
        is6731 = np.where(line['name']=='[SII]_6731')[0]
        coeff = np.asarray([-0.64326,-0.32967,-0.23058])
        disp = 0.1 # dex

        line['ratio'][is6716] = 10**(np.polyval(coeff,oiiihbeta)+
                                          np.random.normal(0.0,disp))
        line['ratio'][is6731] = line['ratio'][is6716]/siidoublet

        # normalize [NeIII] 3869
        is3869 = np.where(line['name']=='[NeIII]_3869')[0]
        coeff = np.asarray([1.0876,-1.1647])
        disp = 0.1 # dex

        line['ratio'][is3869] = 10**(np.polyval(coeff,oiiihbeta)+
                                          np.random.normal(0.0,disp))

        # normalize [OII] 3727, split into [OII] 3726,3729
        is3726 = np.where(line['name']=='[OII]_3726')[0]
        is3729 = np.where(line['name']=='[OII]_3729')[0]
        coeff = np.asarray([-0.52131,-0.74810,0.44351,0.45476])
        disp = 0.1 # dex

        oiihbeta = 10**(np.polyval(coeff,oiiihbeta)+ # [OII] 3727/Hbeta
                        np.random.normal(0.0,disp)) 

        factor1 = oiidoublet/(1.0+oiidoublet) # convert 3727-->3726
        factor2 = 1.0/(1.0+oiidoublet)        # convert 3727-->3729
        line['ratio'][is3726] = factor1*oiihbeta
        line['ratio'][is3729] = factor2*oiihbeta
        
        # Normalize the full spectrum to the desired integrated [OII] 3727 or
        # H-beta flux (but not both!)
        if (oiiflux is None) and (hbetaflux is None):
            line['flux'] = line['ratio']
        
        if (hbetaflux is None) and (oiiflux is not None):
            for ii in range(nline):
                line['ratio'][ii] /= line['ratio'][is3729]
                line['flux'][ii] = oiiflux/(1.0+oiidoublet)*line['ratio'][ii]
                
        if (hbetaflux is not None) and (oiiflux is None):
            for ii in range(nline):
                line['flux'][ii] = hbetaflux*line['ratio'][ii]

        # Finally build the emission-line spectrum
        log10sigma = linesigma/LIGHT/np.log(10) # line-width [log-10 Angstrom]
        emspec = np.zeros(self.npix)
        for ii in range(len(line)):
            amp = line['flux'][ii]/line['wave'][ii]/np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii]*(1.0+zshift))
            line['amp'][ii] = amp/(np.sqrt(2.0*np.pi)*log10sigma)  # [erg/s/A]

            # [erg/s/cm2/A, rest]
            emspec += amp*np.exp(-0.5*(self.wave-thislinewave)**2/log10sigma**2)\
                      /(np.sqrt(2.0*np.pi)*log10sigma)

        return emspec, self.wave, self.line

def read_base_templates(objtype='elg', observed=False, continuum=False,
                        kcorrections=False):
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
    
