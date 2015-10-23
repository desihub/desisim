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

class TargetCuts():
    """Select targets from flux cuts.  This is a placeholder class that will be
       refactored into desitarget.  Hence, the documentation here is
       intentionally sparse.

    """
    def __init__(self):
        pass
        
    def BGS(self,rflux=None):
        BGS = rflux > 10**((22.5-19.35)/2.5)
        return BGS

    def ELG(self,gflux=None, rflux=None, zflux=None):
        ELG  = rflux > 10**((22.5-23.4)/2.5)
        ELG &= zflux > 10**(0.3/2.5) * rflux
        ELG &= zflux < 10**(1.5/2.5) * rflux
        ELG &= rflux**2 < gflux * zflux * 10**(-0.2/2.5)
        ELG &= zflux < gflux * 10**(1.2/2.5)
        return ELG

    def FSTD(self,gflux=None, rflux=None, zflux=None):
        gr = -2.5*np.log10(gflux/rflux)-0.32
        rz = -2.5*np.log10(rflux/zflux)-0.13
        mdist = np.sqrt(gr**2 + rz**2)
        FSTD = mdist<0.06
        return FSTD

    def LRG(self,rflux=None, zflux=None, w1flux=None):
        LRG  = rflux > 10**((22.5-23.0)/2.5)
        LRG &= zflux > 10**((22.5-20.56)/2.5)
        LRG &= w1flux > 10**((22.5-19.35)/2.5)
        LRG &= zflux > rflux * 10**(1.6/2.5)
        LRG &= w1flux * rflux ** (1.33-1) > zflux**1.33 * 10**(-0.33/2.5)
        return LRG

    def QSO(self,gflux=None, rflux=None, zflux=None, w1flux=None, w2flux=None):
        wflux = 0.75 * w1flux + 0.25 * w2flux

        QSO  = rflux > 10**((22.5-23.0)/2.5)
        QSO &= rflux < 10**(1.0/2.5) * gflux
        QSO &= zflux > 10**(-0.3/2.5) * rflux
        QSO &= zflux < 10**(1.1/2.5) * rflux
        QSO &= wflux * gflux**1.2 > 10**(2/2.5) * rflux**(1+1.2)
        return QSO

    def WD(self):
        pass

class ELG():
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs).

    """
    def __init__(self, nmodel=50, minwave=3600.0, maxwave=10000.0,
                 cdelt=2.0, seed=None):
        """Read the ELG basis continuum templates, grzW1 filter profiles and initialize
           the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        TODO (@moustakas): Incorporate size and morphological priors.

        Args:
          objtype (str): object type
          nmodel (int, optional): Number of models to generate (default 50). 
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          seed (long, optional): input seed for the random numbers
    
        Attributes:
          objtype (str): See Args.
          nmodel (int): See Args.
          seed (long): See Args.
          rand (numpy.RandomState): instance of numpy.random.RandomState(seed)
          wave (numpy.ndarray): Output wavelength array constructed from the input
            wavelength arguments [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            ELG continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          gfilt (FILTERFUNC instance): DECam g-band filter profile class.
          rfilt (FILTERFUNC instance): DECam r-band filter profile class.
          zfilt (FILTERFUNC instance): DECam z-band filter profile class.
          w1filt (FILTERFUNC instance): WISE W1-band filter profile class.

        """
        from desisim.filterfunc import filterfunc as filt
        from desisim.io import read_base_templates

        self.objtype = 'ELG'
        self.nmodel = nmodel
        self.seed = seed
        self.rand = np.random.RandomState(seed=self.seed)

        # Initialize the output wavelength array (linear spacing)
        npix = (maxwave-minwave)/cdelt+1
        self.wave = np.linspace(minwave,maxwave,npix) 

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_base_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.gfilt = filt(filtername='decam_g.txt')
        self.rfilt = filt(filtername='decam_r.txt')
        self.zfilt = filt(filtername='decam_z.txt')
        self.w1filt = filt(filtername='wise_w1.txt')

    def make_templates(self, zrange=(0.6,1.6), rmagrange=(21.0,23.5),
                       oiiihbrange=(-0.5,0.1), oiidoublet_meansig=(0.73,0.05),
                       linesigma_meansig=(1.887,0.175), minoiiflux=1E-17,
                       no_colorcuts=False, header_comments=None, outfile=None):
        """Build Monte Carlo set of ELG spectra/templates.

        This function chooses random subsets of the ELG continuum spectra, constructs
        an emission-line spectrum, redshifts, and then finally normalizes the spectrum
        to a specific r-band magnitude.

        TODO (@moustakas): optionally normalized to a g-band magnitude

        Args:
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.6,1.6).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (21,23.5).
          oiiihbrange (float, optional): Minimum and maximum logarithmic
            [OIII] 5007/H-beta line-ratio.  Defaults to a uniform distribution
            between (-0.5,0.1).
        
          oiidoublet_meansig (float, optional): Mean and sigma values for the (Gaussian) 
            [OII] 3726/3729 doublet ratio distribution.  Defaults to (0.73,0.05).
          linesigma_meansig (float, optional): *Logarithmic* mean and sigma values for the
            (Gaussian) emission-line velocity width distribution.  Defaults to
            log10-sigma(=1.887+/0.175) km/s.

          minoiiflux (float, optional): Minimum [OII] 3727 flux [default 1E-17 erg/s/cm2].
            Set this parameter to zero to not have a minimum flux cut.
          no_colorcuts (bool, optional): Do not apply the fiducial grz color-cuts
            cuts (default False).
        
          outfile (str, optional): Write the template spectra (with header information) and
            the corresponding meta-data table to this file (default None).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A]. 
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table, Column

        from desisim.templates import EMSpectrum
        from desisim.io import write_templates
        from desispec.interpolation import resample_flux

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample. 
        EM = EMSpectrum(log10wave=np.log10(self.basewave),seed=self.seed)
       
        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([self.nmodel,len(self.wave)]) # [erg/s/cm2/A]

        meta = Table()
        meta['TEMPLATEID'] = Column(np.zeros(self.nmodel,dtype='i4'))
        meta['REDSHIFT'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['GMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['RMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['ZMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['W1MAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['OIIFLUX'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['EWOII'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['OIIIHBETA'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['OIIDOUBLET'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['LINESIGMA'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['D4000'] = Column(np.zeros(self.nmodel,dtype='f4'))

        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(self.nmodel,500)

        Cuts = TargetCuts()
        while nobj<=(self.nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = self.rand.randint(0,nbase-1,nchunk)

            # Assign uniform redshift and r-magnitude distributions.
            redshift = self.rand.uniform(zrange[0],zrange[1],nchunk)
            rmag = self.rand.uniform(rmagrange[0],rmagrange[1],nchunk)

            # Assume the emission-line priors are uncorrelated.
            oiiihbeta = self.rand.uniform(oiiihbrange[0],oiiihbrange[1],nchunk)
            oiidoublet = self.rand.normal(oiidoublet_meansig[0],
                                          oiidoublet_meansig[1],nchunk)
            linesigma = self.rand.normal(linesigma_meansig[0],
                                         linesigma_meansig[1],nchunk)

            d4000 = self.basemeta['D4000'][chunkindx]
            ewoii = 10.0**(np.polyval([1.1074,-4.7338,5.6585],d4000)+ 
                           self.rand.normal(0.0,0.3)) # rest-frame, Angstrom

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
                rnorm = 10.0**(-0.4*rmag[ii])/self.rfilt.get_maggies(zwave,restflux)
                flux = restflux*rnorm # [erg/s/cm2/A, @redshift[ii]]

                # [grz]flux are in nanomaggies
                rflux = 10.0**(-0.4*(rmag[ii]-22.5))                      
                gflux = self.gfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                zflux = self.zfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                w1flux = self.w1filt.get_maggies(zwave,flux)*10**(0.4*22.5) 

                zoiiflux = oiiflux*rnorm # [erg/s/cm2]
                oiimask = [zoiiflux>minoiiflux]

                if no_colorcuts:
                    grzmask = [True]
                else:
                    grzmask = [Cuts.ELG(gflux=gflux,rflux=rflux,zflux=zflux)]

                # Not sure why this print statement doesn't work!
                #print('Building model {}/{}'.format(nobj,self.nmodel-1),end='\r')
                if all(grzmask) and all(oiimask):
                    outflux[nobj,:] = resample_flux(self.wave,zwave,flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(gflux)+22.5
                    meta['RMAG'][nobj] = rmag[ii]
                    meta['ZMAG'][nobj] = -2.5*np.log10(zflux)+22.5
                    meta['W1MAG'][nobj] = -2.5*np.log10(w1flux)+22.5
                    meta['OIIFLUX'][nobj] = zoiiflux
                    meta['EWOII'][nobj] = ewoii[ii]
                    meta['OIIIHBETA'][nobj] = oiiihbeta[ii]
                    meta['OIIDOUBLET'][nobj] = oiidoublet[ii]
                    meta['LINESIGMA'][nobj] = linesigma[ii]
                    meta['D4000'][nobj] = d4000[ii]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(self.nmodel-1):
                    break

        # Optionally write out and then return.  There's probably a smarter way
        # to do this with astropy Tables...
        if outfile is not None:
            comments = dict(
                TEMPLATEID = 'template ID',
                REDSHIFT = 'object redshift',
                GMAG = 'DECam g-band AB magnitude',
                RMAG = 'DECam r-band AB magnitude',
                ZMAG = 'DECam z-band AB magnitude',
                W1MAG = 'WISE W1-band AB magnitude',
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
            
            write_templates(outflux, self.wave, meta, self.objtype, outfile=outfile,
                            comments=comments, units=units,
                            header_comments=header_comments)

        return outflux, meta

class EMSpectrum():
    """Construct a complete nebular emission-line spectrum.

    """
    def __init__(self, minwave=3650.0, maxwave=7075.0, cdelt_kms=20.0,
                 log10wave=None, seed=None):
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
          seed (long, optional): input seed for the random numbers
        
          log10wave (numpy.ndarray, optional): Input/output wavelength array
            (log10-Angstrom, default None).

        Attributes:
          log10wave (numpy.ndarray): Wavelength array constructed from the input arguments.
          line (astropy.Table): Table containing the laboratoy (vacuum) wavelengths and nominal
            line-ratios for several dozen forbidden and recombination nebular emission lines. 
          seed (long): See Args.
          rand (numpy.RandomState): instance of numpy.random.RandomState(seed)

        Raises:
          IOError: If the required data files are not found.

        """
        from astropy.io import ascii
        from astropy.table import Table, Column, vstack

        self.seed = seed
        self.rand = np.random.RandomState(seed=self.seed)
        
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
                                          self.rand.normal(0.0,disp))
        line['ratio'][is6548] = line['ratio'][is6584]/niidoublet

        # Normalize [SII] 6716,6731.
        is6716 = np.where(line['name']=='[SII]_6716')[0]
        is6731 = np.where(line['name']=='[SII]_6731')[0]
        coeff = np.asarray([-0.64326,-0.32967,-0.23058])
        disp = 0.1 # dex

        line['ratio'][is6716] = 10**(np.polyval(coeff,oiiihbeta)+
                                          self.rand.normal(0.0,disp))
        line['ratio'][is6731] = line['ratio'][is6716]/siidoublet

        # Normalize [NeIII] 3869.
        is3869 = np.where(line['name']=='[NeIII]_3869')[0]
        coeff = np.asarray([1.0876,-1.1647])
        disp = 0.1 # dex

        line['ratio'][is3869] = 10**(np.polyval(coeff,oiiihbeta)+
                                          self.rand.normal(0.0,disp))

        # Normalize [OII] 3727, split into [OII] 3726,3729.
        is3726 = np.where(line['name']=='[OII]_3726')[0]
        is3729 = np.where(line['name']=='[OII]_3729')[0]
        coeff = np.asarray([-0.52131,-0.74810,0.44351,0.45476])
        disp = 0.1 # dex

        oiihbeta = 10**(np.polyval(coeff,oiiihbeta)+ # [OII] 3727/Hbeta
                        self.rand.normal(0.0,disp)) 

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

class LRG():
    """Generate Monte Carlo spectra of luminous red galaxies (LRGs).

    """
    def __init__(self, nmodel=50, minwave=3600.0, maxwave=10000.0,
                 cdelt=2.0, seed=None):
        """Read the LRG basis continuum templates, grzW1 filter profiles and initialize
           the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        TODO (@moustakas): Incorporate size and morphological priors.

        Args:
          objtype (str): object type
          nmodel (int, optional): Number of models to generate (default 50). 
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          seed (long, optional): input seed for the random numbers
    
        Attributes:
          objtype (str): See Args.
          nmodel (int): See Args.
          seed (long): See Args.
          rand (numpy.RandomState): instance of numpy.random.RandomState(seed)
          wave (numpy.ndarray): Output wavelength array constructed from the input
            wavelength arguments [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            LRG continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          gfilt (FILTERFUNC instance): DECam g-band filter profile class.
          rfilt (FILTERFUNC instance): DECam r-band filter profile class.
          zfilt (FILTERFUNC instance): DECam z-band filter profile class.
          w1filt (FILTERFUNC instance): WISE W1-band filter profile class.

        """
        from desisim.filterfunc import filterfunc as filt
        from desisim.io import read_base_templates

        self.objtype = 'LRG'
        self.nmodel = nmodel
        self.seed = seed
        self.rand = np.random.RandomState(seed=self.seed)

        # Initialize the output wavelength array (linear spacing)
        npix = (maxwave-minwave)/cdelt+1
        self.wave = np.linspace(minwave,maxwave,npix) 

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_base_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.gfilt = filt(filtername='decam_g.txt')
        self.rfilt = filt(filtername='decam_r.txt')
        self.zfilt = filt(filtername='decam_z.txt')
        self.w1filt = filt(filtername='wise_w1.txt')

    def make_templates(self, zrange=(0.5,1.1), zmagrange=(19.0,20.5),
                       no_colorcuts=False, header_comments=None, outfile=None):
        """Build Monte Carlo set of LRG spectra/templates.

        This function chooses random subsets of the LRG continuum spectra and
        finally normalizes the spectrum to a specific z-band magnitude.

        TODO (@moustakas): add a LINER- or AGN-like emission-line spectrum 

        Args:
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.6,1.6).
          zmagrange (float, optional): Minimum and maximum DECam z-band (AB)
            magnitude range.  Defaults to a uniform distribution between (19,20.5).
          no_colorcuts (bool, optional): Do not apply the fiducial rzW1 color-cuts
            cuts (default False).
        
          outfile (str, optional): Write the template spectra (with header information) and
            the corresponding meta-data table to this file (default None).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A]. 
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table, Column

        from desisim.templates import EMSpectrum
        from desisim.io import write_templates
        from desispec.interpolation import resample_flux

        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([self.nmodel,len(self.wave)]) # [erg/s/cm2/A]

        meta = Table()
        meta['TEMPLATEID'] = Column(np.zeros(self.nmodel,dtype='i4'))
        meta['REDSHIFT'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['GMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['RMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['ZMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['W1MAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['ZMETAL'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['AGE'] = Column(np.zeros(self.nmodel,dtype='f4'))

        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(self.nmodel,500)

        Cuts = TargetCuts()
        while nobj<=(self.nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = self.rand.randint(0,nbase-1,nchunk)

            # Assign uniform redshift and z-magnitude distributions.
            redshift = self.rand.uniform(zrange[0],zrange[1],nchunk)
            zmag = self.rand.uniform(zmagrange[0],zmagrange[1],nchunk)

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave*(1.0+redshift[ii])
                restflux = self.baseflux[iobj,:] # [erg/s/cm2/A @10pc]

                znorm = 10.0**(-0.4*zmag[ii])/self.zfilt.get_maggies(zwave,restflux)
                flux = restflux*znorm # [erg/s/cm2/A, @redshift[ii]]

                # [grzW1]flux are in nanomaggies
                zflux = 10.0**(-0.4*(zmag[ii]-22.5))                      
                gflux = self.gfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                rflux = self.rfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                w1flux = self.w1filt.get_maggies(zwave,flux)*10**(0.4*22.5) 

                if no_colorcuts:
                    rzW1mask = [True]
                else:
                    rzW1mask = [Cuts.LRG(rflux=rflux,zflux=zflux,w1flux=w1flux)]

                # Not sure why this print statement doesn't work!
                #print('Building model {}/{}'.format(nobj,self.nmodel-1),end='\r')
                if all(rzW1mask):
                    outflux[nobj,:] = resample_flux(self.wave,zwave,flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(gflux)+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(rflux)+22.5
                    meta['ZMAG'][nobj] = zmag[ii]
                    meta['W1MAG'][nobj] = -2.5*np.log10(w1flux)+22.5
                    meta['ZMETAL'][nobj] = self.basemeta['ZMETAL'][iobj]
                    meta['AGE'][nobj] = self.basemeta['AGE'][iobj]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(self.nmodel-1):
                    break

        # Optionally write out and then return.  There's probably a smarter way
        # to do this with astropy Tables...
        if outfile is not None:
            comments = dict(
                TEMPLATEID = 'template ID',
                REDSHIFT = 'object redshift',
                GMAG = 'DECam g-band AB magnitude',
                RMAG = 'DECam r-band AB magnitude',
                ZMAG = 'DECam z-band AB magnitude',
                W1MAG = 'WISE W1-band AB magnitude',
                ZMETAL = 'stellar metallicity',
                AGE = 'time since the onset of star formation'
            )
    
            units = dict(
                AGE = 'Gyr'
            )
            
            write_templates(outflux, self.wave, meta, self.objtype,
                            outfile=outfile, comments=comments, units=units,
                            header_comments=header_comments)

        return outflux, meta

class STAR():
    """Generate Monte Carlo spectra of normal stars, F-type standard stars, or white
       dwarfs.

    """
    def __init__(self, nmodel=50, minwave=3600.0, maxwave=10000.0,
                 cdelt=2.0, seed=None, FSTD=False, WD=False):
        """Read the stellar basis continuum templates, grzW1 filter profiles and
           initialize the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        Args:
          objtype (str): object type
          nmodel (int, optional): Number of models to generate (default 50). 
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          seed (long, optional): input seed for the random numbers
    
        Attributes:
          objtype (str): See Args.
          nmodel (int): See Args.
          seed (long): See Args.
          rand (numpy.RandomState): instance of numpy.random.RandomState(seed)
          wave (numpy.ndarray): Output wavelength array constructed from the input
            wavelength arguments [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            stellar continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          gfilt (FILTERFUNC instance): DECam g-band filter profile class.
          rfilt (FILTERFUNC instance): DECam r-band filter profile class.
          zfilt (FILTERFUNC instance): DECam z-band filter profile class.

        """
        from desisim.filterfunc import filterfunc as filt
        from desisim.io import read_base_templates

        if FSTD:
            self.objtype = 'FSTD'
        elif WD:
            self.objtype = 'WD'
        else:
            self.objtype = 'STAR'
        self.nmodel = nmodel
        self.seed = seed
        self.rand = np.random.RandomState(seed=self.seed)

        # Initialize the output wavelength array (linear spacing)
        npix = (maxwave-minwave)/cdelt+1
        self.wave = np.linspace(minwave,maxwave,npix) 

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_base_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.gfilt = filt(filtername='decam_g.txt')
        self.rfilt = filt(filtername='decam_r.txt')
        self.zfilt = filt(filtername='decam_z.txt')

    def make_templates(self, vrad_meansig=(0.0,200.0), rmagrange=(18.0,23.5),
                       gmagrange=(16.0,19.0), header_comments=None, outfile=None):
        """Build Monte Carlo set of spectra/templates for stars. 

        This function chooses random subsets of the continuum spectra for stars,
        adds radial velocity "jitter", then normalizes the spectrum to a
        specified r- or g-band magnitude.

        Args:
          vrad_meansig (float, optional): Mean and sigma (standard deviation) of the 
            radial velocity "jitter" (in km/s) that should be added to each
            spectrum.  Defaults to a normal distribution with a mean of zero and
            sigma of 200 km/s.
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (18,23.5).
          gmagrange (float, optional): Minimum and maximum DECam g-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,19). 
          outfile (str, optional): Write the template spectra (with header information) and
            the corresponding meta-data table to this file (default None).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A]. 
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table, Column
        from desisim.io import write_templates
        from desispec.interpolation import resample_flux

        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([self.nmodel,len(self.wave)]) # [erg/s/cm2/A]

        meta = Table()
        meta['TEMPLATEID'] = Column(np.zeros(self.nmodel,dtype='i4'))
        meta['REDSHIFT'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['GMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['RMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['ZMAG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['LOGG'] = Column(np.zeros(self.nmodel,dtype='f4'))
        meta['TEFF'] = Column(np.zeros(self.nmodel,dtype='f4'))

        if self.objtype!='WD':
            meta['FEH'] = Column(np.zeros(self.nmodel,dtype='f4'))

        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(self.nmodel,500)

        Cuts = TargetCuts()
        while nobj<=(self.nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = self.rand.randint(0,nbase-1,nchunk)

            # Assign uniform redshift and r-magnitude distributions.
            if self.objtype=='WD':
                gmag = self.rand.uniform(gmagrange[0],gmagrange[1],nchunk)
            else: 
                rmag = self.rand.uniform(rmagrange[0],rmagrange[1],nchunk)
                
            vrad = self.rand.normal(vrad_meansig[0],vrad_meansig[1],nchunk)
            redshift = vrad/2.99792458E5

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave*(1.0+redshift[ii])
                restflux = self.baseflux[iobj,:] # [erg/s/cm2/A @10pc]

                # Normalize; Note that [grz]flux are in nanomaggies
                if self.objtype=='WD':
                    gnorm = 10.0**(-0.4*gmag[ii])/self.gfilt.get_maggies(zwave,restflux)
                    flux = restflux*gnorm # [erg/s/cm2/A, @redshift[ii]]

                    gflux = 10.0**(-0.4*(gmag[ii]-22.5))                      
                    rflux = self.rfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                    zflux = self.zfilt.get_maggies(zwave,flux)*10**(0.4*22.5)
                else:
                    rnorm = 10.0**(-0.4*rmag[ii])/self.rfilt.get_maggies(zwave,restflux)
                    flux = restflux*rnorm # [erg/s/cm2/A, @redshift[ii]]

                    rflux = 10.0**(-0.4*(rmag[ii]-22.5))                      
                    gflux = self.gfilt.get_maggies(zwave,flux)*10**(0.4*22.5) 
                    zflux = self.zfilt.get_maggies(zwave,flux)*10**(0.4*22.5)

                # Color cuts on just on the standard stars.
                if self.objtype=='FSTD':
                    grzmask = [Cuts.FSTD(gflux=gflux,rflux=rflux,zflux=zflux)]
                elif self.objtype=='WD':
                    grzmask = [True]
                else:
                    grzmask = [True]

                if all(grzmask):
                    outflux[nobj,:] = resample_flux(self.wave,zwave,flux)

                    if self.objtype=='WD':
                        meta['TEMPLATEID'][nobj] = nobj
                        meta['REDSHIFT'][nobj] = redshift[ii]
                        meta['GMAG'][nobj] = gmag[ii]
                        meta['RMAG'][nobj] = -2.5*np.log10(rflux)+22.5
                        meta['ZMAG'][nobj] = -2.5*np.log10(zflux)+22.5
                        meta['LOGG'][nobj] = self.basemeta['LOGG'][iobj]
                        meta['TEFF'][nobj] = self.basemeta['TEFF'][iobj]
                    else:
                        meta['TEMPLATEID'][nobj] = nobj
                        meta['REDSHIFT'][nobj] = redshift[ii]
                        meta['GMAG'][nobj] = -2.5*np.log10(gflux)+22.5
                        meta['RMAG'][nobj] = rmag[ii]
                        meta['ZMAG'][nobj] = -2.5*np.log10(zflux)+22.5
                        meta['LOGG'][nobj] = self.basemeta['LOGG'][iobj]
                        meta['TEFF'][nobj] = self.basemeta['TEFF'][iobj]
                        meta['FEH'][nobj] = self.basemeta['FEH'][iobj]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(self.nmodel-1):
                    break

        # Optionally write out and then return.  There's probably a smarter way
        # to do this with astropy Tables...
        if outfile is not None:
            if self.objtype=='WD':
                comments = dict(
                    TEMPLATEID = 'template ID',
                    REDSHIFT = 'object redshift',
                    GMAG = 'DECam g-band AB magnitude',
                    RMAG = 'DECam r-band AB magnitude',
                    ZMAG = 'DECam z-band AB magnitude',
                    LOGG = 'log10 of the effective gravity',
                    TEFF = 'stellar effective temperature'
                    )
            else:
                comments = dict(
                    TEMPLATEID = 'template ID',
                    REDSHIFT = 'object redshift',
                    GMAG = 'DECam g-band AB magnitude',
                    RMAG = 'DECam r-band AB magnitude',
                    ZMAG = 'DECam z-band AB magnitude',
                    LOGG = 'log10 of the effective gravity',
                    TEFF = 'stellar effective temperature',
                    FEH = 'log10 iron abundance relative to solar',
                    )
                
            units = dict(
                LOGG = 'm/s^2',
                TEFF = 'K',
            )
            
            write_templates(outflux, self.wave, meta, self.objtype, outfile=outfile,
                            comments=comments, units=units,
                            header_comments=header_comments)

        return outflux, meta
