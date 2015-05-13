"""
desisim.templates
=================

Functions to simulate spectral templates for DESI.
"""

from __future__ import division, print_function

import os
import numpy as np

from desispec.log import get_logger

log = get_logger()

class ELG():
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs).

    """
    def __init__(self, nmodel=50, minwave=3600.0, maxwave=10000.0,
                 cdelt=2.0):
        """Read the ELG continuum templates, grzW1 filter profiles and
           initialize the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        Args:
          nmodel (int, optional): Number of models to generate (default 50). 
          minwave (float, optional): minimum value of the output wavelength
            array [Angstrom, default 3600].
          maxwave (float, optional): minimum value of the output wavelength
            array [Angstrom, default 10000].
          cdelt (float, optional): spacing of the output wavelength array
            [Angstrom/pixel, default 2].
    
        Attributes:
          nmodel (int): See Args.
          wave (numpy.ndarray): Output wavelength array constructed from the input
            wavelength arguments [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            ELG continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template.
          gfilt (FILTERFUNC instance): DECam g-band filter profile class.
          rfilt (FILTERFUNC instance): DECam r-band filter profile class.
          zfilt (FILTERFUNC instance): DECam z-band filter profile class.
          w1filt (FILTERFUNC instance): WISE W1-band filter profile class.
        
        """
        from desisim.filterfunc import filterfunc as filt
        from desisim.templates import read_base_templates

        self.nmodel = nmodel

        # Initialize the output wavelength array (linear spacing)
        npix = (maxwave-minwave)/cdelt+1
        self.wave = np.linspace(minwave,maxwave,npix) 

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_base_templates(objtype='ELG')
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.gfilt = filt(filtername='decam_g.txt')
        self.rfilt = filt(filtername='decam_r.txt')
        self.zfilt = filt(filtername='decam_z.txt')
        self.w1filt = filt(filtername='wise_w1.txt')

    def make_templates(self,
                       zrange=(0.6,1.6),rmagrange=(21.0,23.5),
                       oiiihbrange=(-0.5,0.0),
                       oiidoublet_meansig=(0.73,0.05),
                       linesigma_meansig=(80.0,20.0),
                       minoiiflux=1e-18,
                       outfile=None):
        """
        Build Monte Carlo sets of ELG templates.

        Need error checking on the input values (e.g., min is less than max,
        etc.).

        Need a way to alter the grz color cuts.

        Optionally write the results out to a file.
        """
        from astropy.table import Table, Column

        from desispec.interpolation import resample_flux
        from desisim.templates import EMSpectrum
        from imaginglss.analysis import cuts

        #import matplotlib.pyplot as plt

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample. 
        EM = EMSpectrum(log10wave=np.log10(self.basewave))
       
        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([self.nmodel,len(self.wave)]) # [erg/s/cm2/A]
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
        meta['LINESIGMA'] = np.zeros(self.nmodel,dtype='f4')
        meta['D4000'] = np.zeros(self.nmodel,dtype='f4')

        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(self.nmodel,500)

        while nobj<=(self.nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = np.random.randint(0,nbase-1,nchunk)

            # Assign uniform redshift and r-magnitude distributions.
            redshift = np.random.uniform(zrange[0],zrange[1],nchunk)
            rmag = np.random.uniform(rmagrange[0],rmagrange[1],nchunk)

            # Assume the emission-line priors are uncorrelated.
            oiiihbeta = np.random.uniform(oiiihbrange[0],oiiihbrange[1],nchunk)
            oiidoublet = np.random.normal(oiidoublet_meansig[0],
                                          oiidoublet_meansig[1],nchunk)
            linesigma = np.random.normal(linesigma_meansig[0],
                                         linesigma_meansig[1],nchunk)

            d4000 = self.basemeta['D4000'][chunkindx]
            ewoii = 10.0**(np.polyval([1.1074,-4.7338,5.6585],d4000)+ 
                           np.random.normal(0.0,0.3)) # rest-frame, Angstrom

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave*(1.0+redshift[ii])

                # Add the continuum and emission-line spectra with the
                # right [OII] flux [erg/s/cm2]
                oiiflux = self.basemeta['OII_CONTINUUM'][iobj]*ewoii[ii] 
                emflux, emwave, emline = EM.spectrum(linesigma=linesigma[ii],
                                                      oiidoublet=oiidoublet[ii],
                                                      oiiihbeta=oiiihbeta[ii],
                                                      oiiflux=oiiflux)
                restflux = self.baseflux[iobj,:] + emflux # [erg/s/cm2/A @10pc]
                rnorm = 10.0**(-0.4*rmag[ii])/self.rfilt.get_maggies(
                    zwave,restflux)
                flux = restflux*rnorm # [erg/s/cm2/A, @redshift[ii]]

                # [grz]flux are in nanomaggies
                rflux = 10.0**(-0.4*(rmag[ii]-22.5))                      
                gflux = self.gfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                zflux = self.zfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                zoiiflux = oiiflux*rnorm

                grzmask = cuts.Fluxes.ELG(gflux=gflux,rflux=rflux,zflux=zflux)
                oiimask = [zoiiflux>=minoiiflux]

                print(ii, iobj, nobj)
                if all(grzmask) and all(oiimask):
                    outflux[nobj,:] = resample_flux(self.wave,zwave,flux)
                    
                    #plt.plot(self.wave,outflux[nobj,:])
                    #plt.show()

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(gflux)+22.5
                    meta['RMAG'][nobj] = rmag[ii]
                    meta['ZMAG'][nobj] = -2.5*np.log10(zflux)+22.5
                    meta['OIIFLUX'][nobj] = zoiiflux
                    meta['EWOII'][nobj] = ewoii[ii]
                    meta['OIIIHBETA'][nobj] = oiiihbeta[ii]
                    meta['OIIDOUBLET'][nobj] = oiidoublet[ii]
                    meta['LINESIGMA'][nobj] = linesigma[ii]
                    meta['D4000'][nobj] = d4000[ii]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(self.nmodel-1): break

        # Optionally write out and then return.
        if outfile is not None:
            log.info('Writing {}.'.format(outfile))
            self.write_templates(outflux,self.wave,meta,'elg',outfile=outfile)

        return outflux, meta

    def write_templates(self, flux, wave, meta, objtype, outfile=None):
        """ Write out simulated galaxy templates.
        
        Should this go in the desisim.io module?
        """
        from astropy.io import fits
        from desispec.io import util
        from desisim.obs import _dict2ndarray

        if outfile is None:
            pass
            
        header = dict(
            OBJTYPE = (objtype, 'Object type (ELG, LRG, QSO, BGS, STD, STAR)'),
            CUNIT = ('Angstrom', 'units of wavelength array'),
            CRPIX1 = (1, 'reference pixel number'),
            CRVAL1 = (wave[0], 'Starting wavelength [Angstrom]'),
            CDELT1 = (wave[1]-wave[0], 'Wavelength step [Angstrom]'),
            LOGLAM = (0, 'linear wavelength steps, not log10'),
            AIRORVAC = ('vac', 'wavelengths in vacuum (vac) or air'),
            BUNIT = ('erg/s/cm2/A', 'spectrum flux units')
            )
        hdr = util.fitsheader(header)
        fits.writeto(outfile,flux.astype(np.float32),header=hdr,clobber=True)
    
        # Add the version number to the metadata header
        
        comments = dict(
            TEMPLATEID = 'template ID',
            REDSHIFT = 'object redshift',
            GMAG = 'DECam g-band AB magnitude',
            RMAG = 'DECam r-band AB magnitude',
            ZMAG = 'DECam z-band AB magnitude',
            OIIFLUX = '[OII] 3727 flux',
            EWOII = 'rest-frame equivalenth width of [OII] 3727',
            OIIIHBETA = 'logarithmic [OIII] 5007/H-beta ratio',
            OIIDOUBLET = '[OII] 3726/3729 doublet ratio',
            LINESIGMA = 'emission line velocity width',
            D4000 = '4000-Angstrom break',
            )
            
        units = dict(
            OIIFLUX = 'erg/s/cm2',
            EWOII = 'Angstrom',
            OIIIHBETA = 'dex',
            LINESIGMA = 'km/s',
            )
    
        outmeta = _dict2ndarray(meta)
        util.write_bintable(outfile, outmeta, header=None, extname='METADATA',
                            comments=comments, units=units)

class EMSpectrum():
    """Construct a complete nebular emission-line spectrum.

    """
    def __init__(self, minwave=3650.0, maxwave=7075.0, cdelt_kms=20.0, log10wave=None):
        """
        Read the requisite external data files and initialize the output wavelength array.

        The desired output wavelength array can either by passed directly using LOG10WAVE
        (note: must be a log-base10, i.e., constant-velocity pixel array!) or via the MINWAVE,
        MAXWAVE, and CDELT_KMS arguments.

        In addition, two data files are required: ${DESISIM}/data/recombination_lines.dat and
        ${DESISIM}/data/forbidden_lines.dat.

        TODO (@moustakas): Incorporate AGN-like emission-line ratios.

        Args:
          minwave (float, optional): Minimum value of the output wavelength
            array [Angstrom, default 3600].
          maxwave (float, optional): Minimum value of the output wavelength
            array [Angstrom, default 10000].
          cdelt_kms (float, optional): Spacing of the output wavelength array
            [km/s, default 20].
        
          log10wave (numpy.ndarray, optional): Input/output wavelength array
            (log10-Angstrom, default None).

        Attributes:
          log10wave (numpy.ndarray): Wavelength array constructed from the input arguments.
          line (astropy.Table): Table containing the laboratoy (vacuum) wavelengths and nominal
            line-ratios for several dozen forbidden and recombination nebular emission lines. 

        Raises:
          IOError: If the required data files are not found.

        """
        from astropy.io import ascii
        from astropy.table import Table, Column, vstack

        # Build a wavelength array if one is not given.
        if log10wave is None:
            cdelt = cdelt_kms/2.99792458E5/np.log(10) # pixel size [log-10 A]
            npix = (np.log10(maxwave)-np.log10(minwave))/cdelt+1
            self.log10wave = np.linspace(np.log10(minwave),np.log10(maxwave),cdelt)
        else:
            self.log10wave = log10wave

        # Read the files which contain the recombination and forbidden lines. 
        recombfile = os.path.join(os.getenv('DESISIM'),'data','recombination_lines.dat')
        forbidfile = os.path.join(os.getenv('DESISIM'),'data','forbidden_lines.dat')

        if not os.path.isfile(recombfile):
            log.error('Required data file {} not found!'.format(recombfile))
            raise IOError
        if not os.path.isfile(forbidfile):
            log.error('Required data file {} not found!'.format(forbidfile))
            raise IOError

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

    def spectrum(self, oiiihbeta=-0.2, oiidoublet=0.73, siidoublet=1.3,
                 linesigma=75.0, zshift=0.0, oiiflux=None, hbetaflux=None):
        """Build the actual emission-line spectrum.

        Building the emission-line spectrum involves three main steps.
        First, the ratio of the forbidden emission line-strengths
        relative to H-beta are derived using an input [OIII] 5007/H-beta
        ratio and the empirical relations documented elsewhere.

        Second, the requested [OII] 3726,29 and [SII] 6716,31 doublet
        ratios are imposed.

        And finally the full emission-line spectrum is self-consistently
        normalized to *either* an integrated [OII] 3726,29 line-flux
        *or* an integrated H-beta line-flux.  Generally an ELG and LRG
        spectrum will be normalized using [OII] while the a BGS spectrum
        will be normalized using H-beta.  Note that the H-beta normalization
        trumps the [OII] normalization (in the case that both are given).

        TODO (@moustakas): Allow the random seed to be passed so that a given
        (random) spectrum can be reproduced.
        TODO (@moustakas): Add a suitably scaled nebular continuum spectrum.
        TODO (@moustakas): Add more emission lines.

        Args:
          oiiihbeta (float, optional): Desired logarithmic [OIII] 5007/H-beta
            line-ratio (default -0.2).  A sensible range is [-0.5,0.1].
          oiidoublet (float, optional): Desired [OII] 3726/3729 doublet ratio
            (default 0.73).
          siidoublet (float, optional): Desired [SII] 6716/6731 doublet ratio
            (default 1.3).
          linesigma (float, optional): Intrinsic emission-line velocity width/sigma
            (default 75 km/s).  A sensible range is [30-150].
          zshift (float, optional): Perturb the emission lines from their laboratory
            (rest) wavelengths by a factor 1+ZSHIFT (default 0.0).  Use with caution! 
          oiiflux (float, optional): Normalize the emission-line spectrum to this
            integrated [OII] emission-line flux (default None).
          hbetaflux (float, optional): Normalize the emission-line spectrum to this
            integrated H-beta emission-line flux (default None).

        Returns:
          emspec (numpy.ndarray): Array [npix] of flux values [erg/s/cm2/A].
          wave (numpy.ndarray): Array [npix] of vacuum wavelengths corresponding to
            FLUX [Angstrom, linear spacing].
          line (astropy.Table): Table of emission-line parameters used to generate
            the emission-line spectrum.

        """
        oiiidoublet = 2.8875    # [OIII] 5007/4959 doublet ratio (set by atomic physics)
        niidoublet = 2.93579    # [NII] 6584/6548 doublet ratio (set by atomic physics)

        line = self.line
        nline = len(line)

        # Normalize [OIII] 4959, 5007 .
        is4959 = np.where(line['name']=='[OIII]_4959')[0]
        is5007 = np.where(line['name']=='[OIII]_5007')[0]
        line['ratio'][is5007] = 10**oiiihbeta # NB: no scatter

        line['ratio'][is4959] = line['ratio'][is5007]/oiiidoublet

        # Normalize [NII] 6548,6584.
        is6548 = np.where(line['name']=='[NII]_6548')[0]
        is6584 = np.where(line['name']=='[NII]_6584')[0]
        coeff = np.asarray([-0.53829,-0.73766,-0.20248])
        disp = 0.1 # dex

        line['ratio'][is6584] = 10**(np.polyval(coeff,oiiihbeta)+
                                          np.random.normal(0.0,disp))
        line['ratio'][is6548] = line['ratio'][is6584]/niidoublet

        # Normalize [SII] 6716,6731.
        is6716 = np.where(line['name']=='[SII]_6716')[0]
        is6731 = np.where(line['name']=='[SII]_6731')[0]
        coeff = np.asarray([-0.64326,-0.32967,-0.23058])
        disp = 0.1 # dex

        line['ratio'][is6716] = 10**(np.polyval(coeff,oiiihbeta)+
                                          np.random.normal(0.0,disp))
        line['ratio'][is6731] = line['ratio'][is6716]/siidoublet

        # Normalize [NeIII] 3869.
        is3869 = np.where(line['name']=='[NeIII]_3869')[0]
        coeff = np.asarray([1.0876,-1.1647])
        disp = 0.1 # dex

        line['ratio'][is3869] = 10**(np.polyval(coeff,oiiihbeta)+
                                          np.random.normal(0.0,disp))

        # Normalize [OII] 3727, split into [OII] 3726,3729.
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

        if (hbetaflux is not None) and (oiiflux is not None):
            log.warning('Both HBETAFLUX and OIIFLUX were given; using HBETAFLUX.')
            for ii in range(nline):
                line['flux'][ii] = hbetaflux*line['ratio'][ii]

        # Finally build the emission-line spectrum
        log10sigma = linesigma/2.99792458E5/np.log(10) # line-width [log-10 Angstrom]
        emspec = np.zeros(len(self.log10wave))
        for ii in range(len(line)):
            amp = line['flux'][ii]/line['wave'][ii]/np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii]*(1.0+zshift))
            line['amp'][ii] = amp/(np.sqrt(2.0*np.pi)*log10sigma)  # [erg/s/A]

            # Construct the spectrum [erg/s/cm2/A, rest]
            emspec += amp*np.exp(-0.5*(self.log10wave-thislinewave)**2/log10sigma**2)\
                      /(np.sqrt(2.0*np.pi)*log10sigma)

        return emspec, 10**self.log10wave, self.line

def read_base_templates(objtype='ELG', observed=False, emlines=False):
    """Return the base, rest-frame, spectral continuum templates for each objtype.

    The appropriate environment variable must be set depending on OBJTYPE.  For example,
    DESI_ELG_TEMPLATES, DESI_LRG_TEMPLATES, etc., otherwise an exception will be raised.

    Args:
      objtype (str, optional): object type to read (ELG, LRG, QSO, BGS, STD, or STAR;
        defaults to 'ELG').
      observed (bool): Read the observed-frame templates (defaults to False).
      emlines (bool): Read the spectral templates which include emission lines (defaults
        to False; only applies to object types ELG and BGS).

    Returns:
      flux (numpy.ndarray): Array [ntemplate,npix] of flux values [erg/s/cm2/A].
      wave (numpy.ndarray): Array [npix] of wavelengths for FLUX [Angstrom].
      meta (astropy.Table): Meta-data table for each object.  The contents of this
        table varies depending on what OBJTYPE has been read.

    Raises:
      EnvironmentError: If the appropriate environment variable is not set.
      IOError: If the base templates are not found.
    
    """
    from astropy.io import fits
    from astropy.table import Table
    from desispec.io.util import header2wave

    key = 'DESI_'+objtype.upper()+'_TEMPLATES'
    if key not in os.environ:
        log.error('Required ${} environment variable not set'.format(key))
        raise EnvironmentError

    objfile = os.getenv(key)

    # Handle special cases for the ELG & BGS templates.
    if objtype.upper()=='ELG' or objtype.upper()=='BGS':
        if emlines is not True:
            objfile = objfile.replace('templates_','continuum_templates_')
        if observed is True:
            objfile = objfile.replace('templates_','templates_obs_')

    if not os.path.isfile(objfile):
        log.error('Base templates file {} not found'.format(objfile))
        raise IOError

    flux, hdr = fits.getdata(objfile, 0, header=True)
    meta = Table(fits.getdata(objfile, 1))
    wave = header2wave(hdr)

    return flux, wave, meta
