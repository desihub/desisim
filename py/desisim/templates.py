"""
desisim.templates
=================

Functions to simulate spectral templates for DESI.
"""

from __future__ import division, print_function

import os
import sys
import numpy as np

import desisim.io
from desispec.log import get_logger
log = get_logger()

LIGHT = 2.99792458E5  #- speed of light in km/s
MAG2NANO = 10.0**(0.4*22.5)

class GaussianMixtureModel():
    """Read and sample from a pre-defined Gaussian mixture model.

    """
    def __init__(self, weights, means, covars, covtype):
        self.weights = weights
        self.means = means
        self.covars = covars
        self.covtype = covtype
        self.n_components, self.n_dimensions = self.means.shape
    
    @staticmethod
    def save(model, filename):
        from astropy.io import fits
        hdus = fits.HDUList()
        hdr = fits.Header()
        hdr['covtype'] = model.covariance_type
        hdus.append(fits.ImageHDU(model.weights_, name='weights', header=hdr))
        hdus.append(fits.ImageHDU(model.means_, name='means'))
        hdus.append(fits.ImageHDU(model.covars_, name='covars'))
        hdus.writeto(filename, clobber=True)
        
    @staticmethod
    def load(filename):
        from astropy.io import fits
        hdus = fits.open(filename, memmap=False)
        hdr = hdus[0].header
        covtype = hdr['covtype']
        model = GaussianMixtureModel(
            hdus['weights'].data, hdus['means'].data, hdus['covars'].data, covtype)
        hdus.close()
        return model
    
    def sample(self, n_samples=1, random_state=None):
        
        if self.covtype != 'full':
            return NotImplementedError(
                'covariance type "{0}" not implemented yet.'.format(self.covtype))
        
        # Code adapted from sklearn's GMM.sample()
        if random_state is None:
            random_state = np.random.RandomState()

        weight_cdf = np.cumsum(self.weights)
        X = np.empty((n_samples, self.n_dimensions))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                X[comp_in_X] = random_state.multivariate_normal(
                    self.means[comp], self.covars[comp], num_comp_in_X)
        return X

class EMSpectrum():
    """Construct a complete nebular emission-line spectrum.

    """
    def __init__(self, minwave=3650.0, maxwave=7075.0, cdelt_kms=20.0, log10wave=None):
        """
        Read the requisite external data files and initialize the output wavelength array.

        The desired output wavelength array can either by passed directly using LOG10WAVE
        (note: must be a log-base10, i.e., constant-velocity pixel array!) or via the MINWAVE,
        MAXWAVE, and CDELT_KMS arguments.

        In addition, three data files are required: ${DESISIM}/data/recombination_lines.escv,
        ${DESISIM}/data/forbidden_lines.esv, and ${DESISIM}/data/forbidden_mog.fits.

        TODO (@moustakas): Incorporate AGN-like emission-line ratios.
        TODO (@moustakas): Think about how to best include dust attenuation in the lines. 

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
          forbidmog (GaussianMixtureModel): Table containing the mixture of Gaussian parameters
            encoding the forbidden emission-line priors.
          oiiidoublet (float32): Intrinsic [OIII] 5007/4959 doublet ratio (set by atomic physics).
          niidoublet (float32): Intrinsic [NII] 6584/6548 doublet ratio (set by atomic physics).

        Raises:
          IOError: If the required data files are not found.

        """
        from pkg_resources import resource_filename
        from astropy.table import Table, Column, vstack

        # Build a wavelength array if one is not given.
        if log10wave is None:
            cdelt = cdelt_kms/LIGHT/np.log(10) # pixel size [log-10 A]
            npix = (np.log10(maxwave)-np.log10(minwave))/cdelt+1
            self.log10wave = np.linspace(np.log10(minwave), np.log10(maxwave), npix)
        else:
            self.log10wave = log10wave

        # Read the files which contain the recombination and forbidden lines. 
        recombfile = resource_filename('desisim', os.path.join('..','..','data',
                                                               'recombination_lines.ecsv'))
        forbidfile = resource_filename('desisim', os.path.join('..','..','data',
                                                               'forbidden_lines.ecsv'))
        forbidmogfile = resource_filename('desisim', os.path.join('..','..','data',
                                                                  'forbidden_mogs.fits'))
    
        if not os.path.isfile(recombfile):
            log.error('Required data file {} not found!'.format(recombfile))
            raise IOError
        if not os.path.isfile(forbidfile):
            log.error('Required data file {} not found!'.format(forbidfile))
            raise IOError
        if not os.path.isfile(forbidmogfile):
            log.error('Required data file {} not found!'.format(forbidmogfile))
            raise IOError

        recombdata = Table.read(recombfile, format='ascii.ecsv', guess=False)
        forbiddata = Table.read(forbidfile, format='ascii.ecsv', guess=False)
        line = vstack([recombdata,forbiddata], metadata_conflicts='silent')

        nline = len(line)
        line['flux'] = Column(np.ones(nline), dtype='f8')  # integrated line-flux
        line['amp'] = Column(np.ones(nline), dtype='f8')   # amplitude
        self.line = line

        self.forbidmog = GaussianMixtureModel.load(forbidmogfile)

        self.oiiidoublet = 2.8875
        self.niidoublet = 2.93579

    def spectrum(self, oiiihbeta=None, oiihbeta=None, niihbeta=None,
                 siihbeta=None, oiidoublet=0.73, siidoublet=1.3,
                 linesigma=75.0, zshift=0.0, oiiflux=None, hbetaflux=None,
                 seed=None):
        """Build the actual emission-line spectrum.

        Building the emission-line spectrum involves three main steps.  First,
        the oiiihbeta, oiihbeta, and niihbeta emission-line ratios are either
        drawn from the empirical mixture of Gaussians (recommended!) or input
        values are used to construct the line-ratios of the strongest optical
        forbidden lines relative to H-beta.

        Note that all three of oiiihbeta, oiihbeta, and niihbeta must be
        specified simultaneously in order for them to be used.

        Second, the requested [OII] 3726,29 and [SII] 6716,31 doublet
        ratios are imposed.

        And finally the full emission-line spectrum is self-consistently
        normalized to *either* an integrated [OII] 3726,29 line-flux
        *or* an integrated H-beta line-flux.  Generally an ELG and LRG
        spectrum will be normalized using [OII] while the a BGS spectrum
        will be normalized using H-beta.  Note that the H-beta normalization
        trumps the [OII] normalization (in the case that both are given).

        TODO (@moustakas): Add a suitably scaled nebular continuum spectrum.
        TODO (@moustakas): Add more emission lines (e.g., [NeIII] 3869).

        Args:
          oiiihbeta (float, optional): Desired logarithmic [OIII] 5007/H-beta
            line-ratio (default -0.2).  A sensible range is [-0.5,0.2].
          oiihbeta (float, optional): Desired logarithmic [OII] 3726,29/H-beta
            line-ratio (default 0.1).  A sensible range is [0.0,0.4].
          niihbeta (float, optional): Desired logarithmic [NII] 6584/H-beta
            line-ratio (default -0.2).  A sensible range is [-0.6,0.0].
          siihbeta (float, optional): Desired logarithmic [SII] 6716/H-beta
            line-ratio (default -0.3).  A sensible range is [-0.5,0.2].
        
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
          seed (long, optional): input seed for the random numbers.

        Returns:
          emspec (numpy.ndarray): Array [npix] of flux values [erg/s/cm2/A].
          wave (numpy.ndarray): Array [npix] of vacuum wavelengths corresponding to
            FLUX [Angstrom, linear spacing].
          line (astropy.Table): Table of emission-line parameters used to generate
            the emission-line spectrum.

        """
        rand = np.random.RandomState(seed)

        line = self.line.copy()
        nline = len(line)

        # Convenience variables.
        is4959 = np.where(line['name']=='[OIII]_4959')[0]
        is5007 = np.where(line['name']=='[OIII]_5007')[0]
        is6548 = np.where(line['name']=='[NII]_6548')[0]
        is6584 = np.where(line['name']=='[NII]_6584')[0]
        is6716 = np.where(line['name']=='[SII]_6716')[0]
        is6731 = np.where(line['name']=='[SII]_6731')[0]
        #is3869 = np.where(line['name']=='[NeIII]_3869')[0]
        is3726 = np.where(line['name']=='[OII]_3726')[0]
        is3729 = np.where(line['name']=='[OII]_3729')[0]

        # Draw from the MoGs for forbidden lines.
        if oiiihbeta==None or oiihbeta==None or niihbeta==None or siihbeta==None:
            oiiihbeta, oiihbeta, niihbeta, siihbeta = \
              self.forbidmog.sample(random_state=rand)[0]

        # Normalize [OIII] 4959, 5007.
        line['ratio'][is5007] = 10**oiiihbeta # [OIII]/Hbeta
        line['ratio'][is4959] = line['ratio'][is5007]/self.oiiidoublet

        # Normalize [NII] 6548,6584.
        line['ratio'][is6584] = 10**niihbeta # [NII]/Hbeta
        line['ratio'][is6548] = line['ratio'][is6584]/self.niidoublet

        # Normalize [SII] 6716,6731.
        line['ratio'][is6716] = 10**siihbeta # [SII]/Hbeta
        line['ratio'][is6731] = line['ratio'][is6716]/siidoublet

        ## Normalize [NeIII] 3869.
        #coeff = np.asarray([1.0876,-1.1647])
        #disp = 0.1 # dex
        #line['ratio'][is3869] = 10**(np.polyval(coeff,np.log10(oiiihbeta))+
        #                             rand.normal(0.0,disp))

        # Normalize [OII] 3727, split into [OII] 3726,3729.
        factor1 = oiidoublet/(1.0+oiidoublet) # convert 3727-->3726
        factor2 = 1.0/(1.0+oiidoublet)        # convert 3727-->3729
        line['ratio'][is3726] = factor1*10**oiihbeta
        line['ratio'][is3729] = factor2*10**oiihbeta
        
        # Normalize the full spectrum to the desired integrated [OII] 3727 or
        # H-beta flux (but not both!)
        if (oiiflux is None) and (hbetaflux is None):
            line['flux'] = line['ratio']
        
        if (hbetaflux is None) and (oiiflux is not None):
            for ii in range(nline):
                line['ratio'][ii] /= line['ratio'][is3729]
                line['flux'][ii] = oiiflux*factor2*line['ratio'][ii]
                
        if (hbetaflux is not None) and (oiiflux is None):
            for ii in range(nline):
                line['flux'][ii] = hbetaflux*line['ratio'][ii]

        if (hbetaflux is not None) and (oiiflux is not None):
            log.warning('Both HBETAFLUX and OIIFLUX were given; using HBETAFLUX.')
            for ii in range(nline):
                line['flux'][ii] = hbetaflux*line['ratio'][ii]

        # Finally build the emission-line spectrum
        log10sigma = linesigma/LIGHT/np.log(10) # line-width [log-10 Angstrom]
        emspec = np.zeros(len(self.log10wave))
        for ii in range(len(line)):
            amp = line['flux'][ii]/line['wave'][ii]/np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii]*(1.0+zshift))
            line['amp'][ii] = amp/(np.sqrt(2.0*np.pi)*log10sigma)  # [erg/s/A]

            # Construct the spectrum [erg/s/cm2/A, rest]
            emspec += amp*np.exp(-0.5*(self.log10wave-thislinewave)**2/log10sigma**2)\
                      /(np.sqrt(2.0*np.pi)*log10sigma)

        return emspec, 10**self.log10wave, line

class ELG():
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None):
        """Read the ELG basis continuum templates, filter profiles and initialize the
           output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        TODO (@moustakas): Incorporate size and morphological priors.

        Args:
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].

        Attributes:
          objtype (str): 'ELG'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            ELG continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          pixbound (numpy.ndarray): Pixel boundaries of BASEWAVE [Angstrom].
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence 
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence

        """
        from speclite import filters
        from desisim.io import read_basis_templates
        from desisim import pixelsplines as pxs

        self.objtype = 'ELG'

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave)/cdelt+1
            wave = np.linspace(minwave,maxwave,npix) 
        self.wave = wave

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)

        self.ewoiicoeff = [1.34323087,-5.02866474,5.43842874]

        # Initialize the filter profiles.
        self.rfilt = filters.load_filters('decam2014-r')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

    def make_templates(self, nmodel=100, zrange=(0.6,1.6), rmagrange=(21.0,23.4),
                       oiiihbrange=(-0.5,0.2), oiidoublet_meansig=(0.73,0.05),
                       logvdisp_meansig=(1.9,0.15), minoiiflux=1E-17, seed=None,
                       nocolorcuts=False, nocontinuum=False):
        """Build Monte Carlo set of ELG spectra/templates.

        This function chooses random subsets of the ELG continuum spectra, constructs
        an emission-line spectrum, redshifts, and then finally normalizes the spectrum
        to a specific r-band magnitude.

        TODO (@moustakas): optionally normalize to a g-band magnitude

        Args:
          nmodel (int, optional): Number of models to generate (default 100). 
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.6,1.6).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (21,23.4).
          oiiihbrange (float, optional): Minimum and maximum logarithmic
            [OIII] 5007/H-beta line-ratio.  Defaults to a uniform distribution
            between (-0.5,0.2).
        
          oiidoublet_meansig (float, optional): Mean and sigma values for the (Gaussian) 
            [OII] 3726/3729 doublet ratio distribution.  Defaults to (0.73,0.05).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=1.9+/-0.15 km/s
          minoiiflux (float, optional): Minimum [OII] 3727 flux [default 1E-17 erg/s/cm2].
            Set this parameter to zero to not have a minimum flux cut.

          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial grz color-cuts
            cuts (default False).
          nocontinuum (bool, optional): Do not include the stellar continuum
            (useful for testing; default False).  Note that this option
            automatically sets NOCOLORCUTS to True.

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table
        from desisim.templates import EMSpectrum
        from desispec.interpolation import resample_flux
        from desisim import pixelsplines as pxs
        from desitarget.cuts import isELG

        if nocontinuum:
            nocolorcuts = True

        rand = np.random.RandomState(seed)

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample. 
        EM = EMSpectrum(log10wave=np.log10(self.basewave))

        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        metacols = [
            ('TEMPLATEID', 'i4'),
            ('REDSHIFT', 'f4'),
            ('GMAG', 'f4'),
            ('RMAG', 'f4'),
            ('ZMAG', 'f4'),
            ('W1MAG', 'f4'),
            ('W2MAG', 'f4'),
            ('OIIFLUX', 'f4'),
            ('EWOII', 'f4'),
            ('OIIIHBETA', 'f4'),
            ('OIIHBETA', 'f4'),
            ('NIIHBETA', 'f4'),
            ('SIIHBETA', 'f4'),
            ('OIIDOUBLET', 'f4'),
            ('D4000', 'f4'),
            ('VDISP', 'f4'), 
            ('DECAM_FLUX', 'f4', (6,)),
            ('WISE_FLUX', 'f4', (2,))]
        meta = Table(np.zeros(nmodel, dtype=metacols))

        meta['OIIFLUX'].unit = 'erg/(s*cm2)'
        meta['EWOII'].unit = 'Angstrom'
        meta['OIIIHBETA'].unit = 'dex'
        meta['OIIHBETA'].unit = 'dex'
        meta['NIIHBETA'].unit = 'dex'
        meta['SIIHBETA'].unit = 'dex'
        meta['VDISP'].unit = 'km/s'

        # Build the spectra.
        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(nmodel, 500)

        while nobj<=(nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = rand.randint(0, nbase-1, nchunk)

            # Assign uniform redshift, r-magnitude, and velocity dispersion
            # distributions.
            redshift = rand.uniform(zrange[0], zrange[1], nchunk)
            rmag = rand.uniform(rmagrange[0], rmagrange[1], nchunk)
            if logvdisp_meansig[1]>0:
                vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nchunk)
            else:
                vdisp = 10**np.repeat(logvdisp_meansig[0], nchunk)

            # Get the correct number and distribution of emission-line ratios. 
            oiihbeta = np.zeros(nchunk)
            niihbeta = np.zeros(nchunk)
            siihbeta = np.zeros(nchunk)
            oiiihbeta = np.zeros(nchunk)-99
            need = np.where(oiiihbeta==-99)[0]
            while len(need)>0:
                samp = EM.forbidmog.sample(len(need), random_state=rand)
                oiiihbeta[need] = samp[:,0]
                oiihbeta[need] = samp[:,1]
                niihbeta[need] = samp[:,2]
                siihbeta[need] = samp[:,3]
                oiiihbeta[oiiihbeta<oiiihbrange[0]] = -99
                oiiihbeta[oiiihbeta>oiiihbrange[1]] = -99
                need = np.where(oiiihbeta==-99)[0]

            # Assume the emission-line priors are uncorrelated.
            #oiiihbeta = rand.uniform(oiiihbrange[0], oiiihbrange[1], nchunk)
            oiidoublet = rand.normal(oiidoublet_meansig[0],
                                     oiidoublet_meansig[1],
                                     nchunk)
            d4000 = self.basemeta['D4000'][chunkindx]
            ewoii = 10.0**(np.polyval(self.ewoiicoeff,d4000)+
                           rand.normal(0.0,0.3, nchunk)) # rest-frame, Angstrom

            # Create a distribution of seeds for the emission-line spectra.
            emseed = rand.random_integers(0, 100*nchunk, nchunk)

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave.astype(float)*(1.0+redshift[ii])
                restflux = self.baseflux[iobj,:]

                # Normalize to [erg/s/cm2/A, @redshift[ii]]
                rnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                norm = 10.0**(-0.4*rmag[ii])/rnorm['decam2014-r'][0]
                flux = restflux*norm

                # Create an emission-line spectrum with the right [OII] flux [erg/s/cm2]. 
                oiiflux = self.basemeta['OII_CONTINUUM'][iobj]*ewoii[ii] 
                zoiiflux = oiiflux*norm # [erg/s/cm2]

                emflux, emwave, emline = EM.spectrum(
                    linesigma=vdisp[ii],
                    oiidoublet=oiidoublet[ii],
                    oiiihbeta=oiiihbeta[ii],
                    oiihbeta=oiihbeta[ii],
                    niihbeta=niihbeta[ii],
                    siihbeta=siihbeta[ii],
                    oiiflux=zoiiflux,
                    seed=emseed[ii])
                emflux /= (1+redshift[ii]) # [erg/s/cm2/A, @redshift[ii]]

                if nocontinuum:
                    flux = emflux
                else:
                    flux += emflux

                # Convert [grzW1W2]flux to nanomaggies.
                synthmaggies = self.decamwise.get_ab_maggies(flux, zwave, mask_invalid=True)
                synthnano = [ff*MAG2NANO for ff in synthmaggies[0]] # convert to nanomaggies
                
                oiimask = [zoiiflux>minoiiflux]
                if nocolorcuts:
                    colormask = [True]
                else:
                    colormask = [isELG(gflux=synthnano[1],
                                       rflux=synthnano[2],
                                       zflux=synthnano[4])]

                if all(colormask) and all(oiimask):
                    if ((nobj+1)%10)==0:
                        log.debug('Simulating {} template {}/{}'. \
                                  format(self.objtype, nobj+1, nmodel))

                    # (@moustakas) pxs.gauss_blur_matrix is producing lots of
                    # ringing in the emission lines, so deal with it later.
                    
                    # Convolve (just the stellar continuum) and resample.
                    #if nocontinuum is False:
                        #sigma = 1.0+self.basewave*vdisp[ii]/LIGHT
                        #flux = pxs.gauss_blur_matrix(self.pixbound,sigma) * flux
                        #flux = (flux-emflux)*pxs.gauss_blur_matrix(self.pixbound,sigma) + emflux
                    
                    outflux[nobj,:] = resample_flux(self.wave, zwave, flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(synthnano[1])+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(synthnano[2])+22.5
                    meta['ZMAG'][nobj] = -2.5*np.log10(synthnano[4])+22.5
                    meta['W1MAG'][nobj] = -2.5*np.log10(synthnano[6])+22.5
                    meta['W2MAG'][nobj] = -2.5*np.log10(synthnano[7])+22.5
                    meta['DECAM_FLUX'][nobj] = synthnano[:6]
                    meta['WISE_FLUX'][nobj] = synthnano[6:8]
                    meta['OIIFLUX'][nobj] = zoiiflux
                    meta['EWOII'][nobj] = ewoii[ii]
                    meta['OIIIHBETA'][nobj] = oiiihbeta[ii]
                    meta['OIIHBETA'][nobj] = oiihbeta[ii]
                    meta['NIIHBETA'][nobj] = niihbeta[ii]
                    meta['SIIHBETA'][nobj] = siihbeta[ii]
                    meta['OIIDOUBLET'][nobj] = oiidoublet[ii]
                    meta['D4000'][nobj] = d4000[ii]
                    meta['VDISP'][nobj] = vdisp[ii]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(nmodel-1):
                    break

        return outflux, self.wave, meta

class LRG():
    """Generate Monte Carlo spectra of luminous red galaxies (LRGs).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None):
        """Read the LRG basis continuum templates, filter profiles and initialize the
           output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        TODO (@moustakas): Incorporate size and morphological priors.

        Args:
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].
    
        Attributes:
          objtype (str): 'LRG'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            LRG continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          pixbound (numpy.ndarray): Pixel boundaries of BASEWAVE [Angstrom].
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          zfilt (speclite.filters instance): DECam2014 z-band FilterSequence

        """
        from speclite import filters
        from desisim.io import read_basis_templates
        from desisim import pixelsplines as pxs

        self.objtype = 'LRG'

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave)/cdelt+1
            wave = np.linspace(minwave,maxwave,npix) 
        self.wave = wave

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)

        # Initialize the filter profiles.
        self.zfilt = filters.load_filters('decam2014-z')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

    def make_templates(self, nmodel=100, zrange=(0.5,1.1), zmagrange=(19.0,20.5),
                       logvdisp_meansig=(2.3,0.1), seed=None, nocolorcuts=False):
        """Build Monte Carlo set of LRG spectra/templates.

        This function chooses random subsets of the LRG continuum spectra and
        finally normalizes the spectrum to a specific z-band magnitude.

        TODO (@moustakas): add a LINER- or AGN-like emission-line spectrum 

        Args:
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5,1.1).
          zmagrange (float, optional): Minimum and maximum DECam z-band (AB)
            magnitude range.  Defaults to a uniform distribution between (19,20.5).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=2.3+/-0.1 km/s
          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1
            color-cuts cuts (default False).
        
        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A]. 
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table
        from desispec.interpolation import resample_flux
        from desisim import pixelsplines as pxs
        from desitarget.cuts import isLRG

        rand = np.random.RandomState(seed)

        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        metacols = [
            ('TEMPLATEID', 'i4'),
            ('REDSHIFT', 'f4'),
            ('GMAG', 'f4'),
            ('RMAG', 'f4'),
            ('ZMAG', 'f4'),
            ('W1MAG', 'f4'),
            ('W2MAG', 'f4'),
            ('ZMETAL', 'f4'),
            ('AGE', 'f4'),
            ('D4000', 'f4'),
            ('VDISP', 'f4'),
            ('DECAM_FLUX', 'f4', (6,)),
            ('WISE_FLUX', 'f4', (2,))]
        meta = Table(np.zeros(nmodel, dtype=metacols))

        meta['AGE'].unit = 'Gyr'
        meta['VDISP'].unit = 'km/s'

        # Build the spectra.
        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(nmodel, 500)

        while nobj<=(nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = rand.randint(0, nbase-1, nchunk)

            # Assign uniform redshift, z-magnitude, and velocity dispersion
            # distributions.
            redshift = rand.uniform(zrange[0], zrange[1], nchunk)
            zmag = rand.uniform(zmagrange[0], zmagrange[1], nchunk)

            if logvdisp_meansig[1]>0:
                vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nchunk)
            else:
                vdisp = 10**np.repeat(logvdisp_meansig[0], nchunk)

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave.astype(float)*(1.0+redshift[ii])
                restflux = self.baseflux[iobj,:] # [erg/s/cm2/A @10pc]

                # Normalize to [erg/s/cm2/A, @redshift[ii]]
                znorm = self.zfilt.get_ab_maggies(restflux, zwave)
                norm = 10.0**(-0.4*zmag[ii])/znorm['decam2014-z'][0]
                flux = restflux*norm

                # Convert [grzW1W2]flux to nanomaggies.
                synthmaggies = self.decamwise.get_ab_maggies(flux, zwave, mask_invalid=True)
                synthnano = [ff*MAG2NANO for ff in synthmaggies[0]] # convert to nanomaggies

                if nocolorcuts:
                    colormask = [True]
                else:
                    colormask = [isLRG(rflux=synthnano[2],
                                       zflux=synthnano[4],
                                       w1flux=synthnano[6])]

                if all(colormask):
                    if ((nobj+1)%10)==0:
                        log.debug('Simulating {} template {}/{}'. \
                                  format(self.objtype, nobj+1, nmodel))

                    # Convolve and resample
                    sigma = 1.0+self.basewave*vdisp[ii]/LIGHT
                    flux = pxs.gauss_blur_matrix(self.pixbound, sigma) * flux
                        
                    outflux[nobj,:] = resample_flux(self.wave, zwave, flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(synthnano[1])+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(synthnano[2])+22.5
                    meta['ZMAG'][nobj] = -2.5*np.log10(synthnano[4])+22.5
                    meta['W1MAG'][nobj] = -2.5*np.log10(synthnano[6])+22.5
                    meta['W2MAG'][nobj] = -2.5*np.log10(synthnano[7])+22.5
                    meta['DECAM_FLUX'][nobj] = synthnano[:6]
                    meta['WISE_FLUX'][nobj] = synthnano[6:8]
                    meta['ZMETAL'][nobj] = self.basemeta['ZMETAL'][iobj]
                    meta['AGE'][nobj] = self.basemeta['AGE'][iobj]
                    meta['D4000'][nobj] = self.basemeta['D4000'][iobj]
                    meta['VDISP'][nobj] = vdisp[ii]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(nmodel-1):
                    break

        return outflux, self.wave, meta

class STAR():
    """Generate Monte Carlo spectra of normal stars, F-type standard stars, or white
       dwarfs.

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None, 
                 FSTD=False, WD=False):
        """Read the stellar basis continuum templates, filter profiles and initialize
           the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        Args:
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].
    
        Attributes:
          objtype (str): 'FSTD', 'WD', or 'STAR'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            stellar continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          gfilt (speclite.filters instance): DECam2014 g-band FilterSequence
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence

        """
        from speclite import filters
        from desisim.io import read_basis_templates

        if FSTD:
            self.objtype = 'FSTD'
        elif WD:
            self.objtype = 'WD'
        else:
            self.objtype = 'STAR'

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave)/cdelt+1
            wave = np.linspace(minwave,maxwave,npix) 
        self.wave = wave

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')
        self.gfilt = filters.load_filters('decam2014-g')
        self.rfilt = filters.load_filters('decam2014-r')

    def make_templates(self, nmodel=100, vrad_meansig=(0.0,200.0), rmagrange=(18.0,23.4),
                       gmagrange=(16.0,19.0), seed=None):
        """Build Monte Carlo set of spectra/templates for stars. 

        This function chooses random subsets of the continuum spectra for stars,
        adds radial velocity "jitter", then normalizes the spectrum to a
        specified r- or g-band magnitude.

        Args:
          nmodel (int, optional): Number of models to generate (default 100). 
          vrad_meansig (float, optional): Mean and sigma (standard deviation) of the 
            radial velocity "jitter" (in km/s) that should be added to each
            spectrum.  Defaults to a normal distribution with a mean of zero and
            sigma of 200 km/s.
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (18,23.4).
          gmagrange (float, optional): Minimum and maximum DECam g-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,19). 
          seed (long, optional): input seed for the random numbers.

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A]. 
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table
        from desispec.interpolation import resample_flux
        from desitarget.cuts import isFSTD_colors

        rand = np.random.RandomState(seed)

        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        if self.objtype=='WD':
            metacols = [
                ('TEMPLATEID', 'i4'),
                ('REDSHIFT', 'f4'),
                ('GMAG', 'f4'),
                ('RMAG', 'f4'),
                ('ZMAG', 'f4'),
                ('W1MAG', 'f4'),
                ('W2MAG', 'f4'),
                ('LOGG', 'f4'),
                ('TEFF', 'f4'),
                ('DECAM_FLUX', 'f4', (6,)),
                ('WISE_FLUX', 'f4', (2,))]
        else:
            metacols = [
                ('TEMPLATEID', 'i4'),
                ('REDSHIFT', 'f4'),
                ('GMAG', 'f4'),
                ('RMAG', 'f4'),
                ('ZMAG', 'f4'),
                ('W1MAG', 'f4'),
                ('W2MAG', 'f4'),
                ('LOGG', 'f4'),
                ('TEFF', 'f4'),
                ('FEH', 'f4'),
                ('DECAM_FLUX', 'f4', (6,)),
                ('WISE_FLUX', 'f4', (2,))]
        meta = Table(np.zeros(nmodel, dtype=metacols))

        meta['LOGG'].unit = 'm/(s**2)'
        meta['TEFF'].unit = 'K'

        # Build the spectra.
        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(nmodel, 500)

        while nobj<=(nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = rand.randint(0, nbase-1, nchunk)

            # Assign uniform redshift and r-magnitude distributions.
            if self.objtype=='WD':
                gmag = rand.uniform(gmagrange[0], gmagrange[1], nchunk)
            else: 
                rmag = rand.uniform(rmagrange[0], rmagrange[1], nchunk)
                
            vrad = rand.normal(vrad_meansig[0], vrad_meansig[1], nchunk)
            redshift = vrad/LIGHT

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave.astype(float)*(1.0+redshift[ii])
                restflux = self.baseflux[iobj,:] # [erg/s/cm2/A @10pc]

                # Normalize to [erg/s/cm2/A, @redshift[ii]]
                if self.objtype=='WD':
                    gnorm = self.gfilt.get_ab_maggies(restflux, zwave)
                    norm = 10.0**(-0.4*gmag[ii])/gnorm['decam2014-g'][0]
                else:
                    rnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                    norm = 10.0**(-0.4*rmag[ii])/rnorm['decam2014-r'][0]
                flux = restflux*norm
                    
                # Convert [grzW1W2]flux to nanomaggies.
                synthmaggies = self.decamwise.get_ab_maggies(flux, zwave, mask_invalid=True)
                if self.objtype=='WD':
                    synthmaggies['wise2010-W1'] = 0.0
                    synthmaggies['wise2010-W2'] = 0.0
                synthnano = [ff*MAG2NANO for ff in synthmaggies[0]] # convert to nanomaggies

                # Color cuts on just on the standard stars.
                if self.objtype=='FSTD':
                    colormask = [isFSTD_colors(gflux=synthnano[1],
                                               rflux=synthnano[2],
                                               zflux=synthnano[4])]
                elif self.objtype=='WD':
                    colormask = [True]
                else:
                    colormask = [True]

                if all(colormask):
                    if ((nobj+1)%10)==0:
                        log.debug('Simulating {} template {}/{}'. \
                                  format(self.objtype, nobj+1, nmodel))
                    outflux[nobj,:] = resample_flux(self.wave, zwave, flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(synthnano[1])+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(synthnano[2])+22.5
                    meta['ZMAG'][nobj] = -2.5*np.log10(synthnano[4])+22.5
                    if self.objtype!='WD':
                        meta['W1MAG'][nobj] = -2.5*np.log10(synthnano[6])+22.5
                        meta['W2MAG'][nobj] = -2.5*np.log10(synthnano[7])+22.5
                    meta['DECAM_FLUX'][nobj] = synthnano[:6]
                    meta['WISE_FLUX'][nobj] = synthnano[6:8]
                    meta['LOGG'][nobj] = self.basemeta['LOGG'][iobj]
                    meta['TEFF'][nobj] = self.basemeta['TEFF'][iobj]
                    if self.objtype!='WD':
                        meta['FEH'][nobj] = self.basemeta['FEH'][iobj]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(nmodel-1):
                    break

        return outflux, self.wave, meta

class QSO():
    """Generate Monte Carlo spectra of quasars (QSOs).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 z_wind=0.2):
        """Read the QSO basis continuum templates, filter profiles and initialize the
           output wavelength array.

        Note: 
          * Only a linearly-spaced output wavelength array is currently supported.
          * The basis templates are  only defined in the range 3500-10000 A (observed). 

        Args:
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].
          z_wind (float, optional): Redshift window for sampling

        Attributes:
          objtype (str): 'QSO'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            QSO continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          rilt (speclite.filters instance): DECam2014 r-band FilterSequence

        """
        from speclite import filters
        from desisim.io import read_basis_templates

        self.objtype = 'QSO'

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave)/cdelt+1
            wave = np.linspace(minwave,maxwave,npix) 
        self.wave = wave

        # Find the basis files.
        self.basis_file = desisim.io.find_basis_template('qso')
        self.z_wind = z_wind

        # Initialize the filter profiles.
        self.rfilt = filters.load_filters('decam2014-r')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')
        
    def make_templates(self, nmodel=100, zrange=(0.5,4.0), rmagrange=(21.0,23.0),
                       seed=None, nocolorcuts=False, old_way=False):
        """Build Monte Carlo set of QSO spectra/templates.

        This function generates a random set of QSO continua spectra and
        finally normalizes the spectrum to a specific g-band magnitude.

        Args:
          nmodel (int, optional): Number of models to generate (default 100). 
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5,4.0).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (21,23.0).
          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False) (not yet supported).
        
        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A]. 
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table, MaskedColumn
        from desispec.interpolation import resample_flux
        from desisim.qso_template import desi_qso_templ as dqt
        #from desitarget.cuts import isQSO

        rand = np.random.RandomState(seed)

        # Backwards compatiblity hack
        if desisim.io._qso_format_version(self.basis_file) == 1:
            old_way = True

        # This is a temporary hack because the QSO basis templates are
        # already in the observed frame.
        if old_way:
            flux, wave, meta = desisim.io.read_basis_templates('qso', infile=self.basis_file)
            
            keep = np.where(((meta['Z']>=zrange[0])*1)*
                            ((meta['Z']<=zrange[1])*1))[0]
            self.baseflux = flux[keep,:]
            self.basewave = wave
            self.basemeta = meta[keep]
            self.z = meta['Z'][keep]
        else:
            # Generate on-the-fly
            nzbin = (zrange[1]-zrange[0])/self.z_wind
            N_perz = int(nmodel//nzbin + 2)
            _, all_flux, redshifts = dqt.desi_qso_templates(
                zmnx=zrange, no_write=True, rebin_wave=self.wave, rstate=rand,
                N_perz=N_perz
            )
            # Cut down
            ridx = rand.choice(xrange(len(redshifts)), nmodel, replace=False)
            self.baseflux = all_flux[:,ridx]
            self.basewave = self.wave
            self.z = redshifts[ridx]

        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        metacols = [
            ('TEMPLATEID', 'i4'),
            ('REDSHIFT', 'f4'),
            ('GMAG', 'f4'),
            ('RMAG', 'f4'),
            ('ZMAG', 'f4'),
            ('W1MAG', 'f4'),
            ('W2MAG', 'f4'),
            ('DECAM_FLUX', 'f4', (6,)),
            ('WISE_FLUX', 'f4', (2,))]
        meta = Table(np.zeros(nmodel, dtype=metacols))

        nobj = 0
        if old_way:
            nbase = self.baseflux.shape[0]
            nchunk = min(nmodel,500)
        else:
            meta['REDSHIFT'] = self.z
            nbase = self.baseflux.shape[1]
            nchunk = nmodel

        while nobj<=(nmodel-1):
            # Choose a random subset of the base templates
            if old_way:
                chunkindx = rand.randint(0, nbase-1, nchunk)
            else:
                chunkindx = xrange(nchunk)

            # Assign uniform redshift and g-magnitude distributions.
            # redshift = rand.uniform(zrange[0],zrange[1],nchunk)
            rmag = rand.uniform(rmagrange[0], rmagrange[1], nchunk)
            zwave = self.basewave # Hack!

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                if old_way:
                    this = rand.randint(0, nbase-1)
                else:
                    this = ii
                if old_way:
                    obsflux = self.baseflux[this,:] # [erg/s/cm2/A]
                else:
                    obsflux = self.baseflux[:, this] # [erg/s/cm2/A]

                # Normalize to [erg/s/cm2/A, @redshift[ii]].  # Temporary hack
                # until the templates can be extended redward and blueward.
                #rnorm = self.rfilt.get_ab_maggies(obsflux, zwave)
                padflux, padzwave = self.rfilt.pad_spectrum(obsflux, zwave, method='edge')
                rnorm = self.rfilt.get_ab_maggies(padflux, padzwave)
                norm = 10.0**(-0.4*rmag[ii])/rnorm['decam2014-r'][0]
                flux = obsflux*norm

                # Convert [grzW1W2]flux to nanomaggies.  Temporary hack until
                # the templates can be extended redward!
                #synthmaggies = self.decamwise.get_ab_maggies(flux, zwave, mask_invalid=True)
                padflux, padzwave = self.decamwise.pad_spectrum(flux, zwave, method='edge')
                synthmaggies = self.decamwise.get_ab_maggies(padflux, padzwave, mask_invalid=True)
                synthmaggies['wise2010-W1'] = 0.0
                synthmaggies['wise2010-W2'] = 0.0

                synthnano = [ff*MAG2NANO for ff in synthmaggies[0]] # convert to nanomaggies

                if nocolorcuts:
                    colormask = [True]
                else:
                    #colormask = [isQSO(gflux=synthnano[1], # ToDo!
                    #                   rflux=synthnano[2], 
                    #                   zflux=synthnano[4],
                    #                   wflux=(synthnano[6],
                    #                          synthnano[7]))]
                    colormask = [True] 

                if all(colormask):
                    if ((nobj+1)%10)==0:
                        log.debug('Simulating {} template {}/{}'. \
                                  format(self.objtype, nobj+1, nmodel))
                    outflux[nobj,:] = resample_flux(self.wave, zwave, flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    if old_way:
                        meta['REDSHIFT'][nobj] = self.basemeta['Z'][this]
                    meta['GMAG'][nobj] = -2.5*np.log10(synthnano[1])+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(synthnano[2])+22.5
                    meta['ZMAG'][nobj] = -2.5*np.log10(synthnano[4])+22.5
                    #meta['W1MAG'][nobj] = -2.5*np.log10(synthnano[6])+22.5
                    #meta['W2MAG'][nobj] = -2.5*np.log10(synthnano[7])+22.5
                    meta['DECAM_FLUX'][nobj] = synthnano[:6]
                    meta['WISE_FLUX'][nobj] = synthnano[6:8]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(nmodel-1):
                    break
                
        return outflux, self.wave, meta

class BGS():
    """Generate Monte Carlo spectra of bright galaxy survey galaxies (BGSs).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None):
        """Read the BGS basis continuum templates, filter profiles and initialize the
           output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        TODO (@moustakas): Incorporate size and morphological priors.

        Args:
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].

        Attributes:
          objtype (str): 'BGS'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            ELG continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths 
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          ewhbetamog (GaussianMixtureModel): Table containing the mixture of Gaussian 
            parameters  encoding the empirical relationship between D(4000) and EW(Hbeta).
          pixbound (numpy.ndarray): Pixel boundaries of BASEWAVE [Angstrom].
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence 
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence

        Raises:
          IOError: If the required data files are not found.

        """
        from pkg_resources import resource_filename
        from speclite import filters
        from desisim.io import read_basis_templates
        from desisim import pixelsplines as pxs

        self.objtype = 'BGS'

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave)/cdelt+1
            wave = np.linspace(minwave,maxwave,npix) 
        self.wave = wave

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)

        # Read the Gaussian mixture model.
        #ewhbetamogfile = resource_filename('desisim', os.path.join('..','..',
        #                                                           'data',
        #                                                           'bgs_mogs.fits'))
        #if not os.path.isfile(ewhbetamogfile):
        #    log.error('Required data file {} not found!'.format(ewhbetamogfile))
        #    raise IOError
        #self.ewhbetamog = GaussianMixtureModel.load(ewhbetamogfile)
        self.ewhbetacoeff = [1.28520974,-4.94408026,4.9617704]

        # Initialize the filter profiles.
        self.rfilt = filters.load_filters('decam2014-r')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

    def make_templates(self, nmodel=100, zrange=(0.6,1.6), rmagrange=(15.0,19.5),
                       oiiihbrange=(-1.3,0.6), oiidoublet_meansig=(0.73,0.05),
                       logvdisp_meansig=(2.0,0.17), seed=None, nocolorcuts=False,
                       nocontinuum=False):
        """Build Monte Carlo set of BGS spectra/templates.

        This function chooses random subsets of the BGS continuum spectra, constructs
        an emission-line spectrum, redshifts, and then finally normalizes the spectrum
        to a specific r-band magnitude.

        TODO (@moustakas): Calibrate vdisp on data.

        Args:
          nmodel (int, optional): Number of models to generate (default 100). 
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.6,1.6).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (15,19.5).
          oiiihbrange (float, optional): Minimum and maximum logarithmic
            [OIII] 5007/H-beta line-ratio.  Defaults to a uniform distribution
            between (-1.3,0.6).
        
          oiidoublet_meansig (float, optional): Mean and sigma values for the (Gaussian) 
            [OII] 3726/3729 doublet ratio distribution.  Defaults to (0.73,0.05).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=2.0+/-0.17 km/s

          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial color-cuts
            cuts (default False). 
          nocontinuum (bool, optional): Do not include the stellar continuum
            (useful for testing; default False).  Note that this option
            automatically sets NOCOLORCUTS to True.

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from astropy.table import Table
        from desisim.templates import EMSpectrum
        from desispec.interpolation import resample_flux
        from desisim import pixelsplines as pxs
        from desitarget.cuts import isBGS

        if nocontinuum:
            nocolorcuts = True

        rand = np.random.RandomState(seed)

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample. 
        EM = EMSpectrum(log10wave=np.log10(self.basewave))
       
        # Initialize the output flux array and metadata Table.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        metacols = [
            ('TEMPLATEID', 'i4'),
            ('REDSHIFT', 'f4'),
            ('GMAG', 'f4'),
            ('RMAG', 'f4'),
            ('ZMAG', 'f4'),
            ('W1MAG', 'f4'),
            ('W2MAG', 'f4'),
            ('HBETAFLUX', 'f4'),
            ('EWHBETA', 'f4'),
            ('OIIIHBETA', 'f4'),
            ('OIIHBETA', 'f4'),
            ('NIIHBETA', 'f4'),
            ('SIIHBETA', 'f4'),
            ('OIIDOUBLET', 'f4'),
            ('D4000', 'f4'),
            ('VDISP', 'f4'), 
            ('DECAM_FLUX', 'f4', (6,)),
            ('WISE_FLUX', 'f4', (2,))]
        meta = Table(np.zeros(nmodel, dtype=metacols))

        meta['HBETAFLUX'].unit = 'erg/(s*cm2)'
        meta['EWHBETA'].unit = 'Angstrom'
        meta['OIIIHBETA'].unit = 'dex'
        meta['OIIHBETA'].unit = 'dex'
        meta['NIIHBETA'].unit = 'dex'
        meta['SIIHBETA'].unit = 'dex'
        meta['VDISP'].unit = 'km/s'

        # Build the spectra.
        nobj = 0
        nbase = len(self.basemeta)
        nchunk = min(nmodel, 500)

        while nobj<=(nmodel-1):
            # Choose a random subset of the base templates
            chunkindx = rand.randint(0, nbase-1, nchunk)

            # Assign uniform redshift and r-magnitude distributions.
            redshift = rand.uniform(zrange[0], zrange[1], nchunk)
            rmag = rand.uniform(rmagrange[0], rmagrange[1], nchunk)
            if logvdisp_meansig[1]>0:
                vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nchunk)
            else:
                vdisp = 10**np.repeat(logvdisp_meansig[0], nchunk)

            # Get the correct number and distribution of emission-line ratios. 
            oiihbeta = np.zeros(nchunk)
            niihbeta = np.zeros(nchunk)
            siihbeta = np.zeros(nchunk)
            oiiihbeta = np.zeros(nchunk)-99
            need = np.where(oiiihbeta==-99)[0]
            while len(need)>0:
                samp = EM.forbidmog.sample(len(need), random_state=rand)
                oiiihbeta[need] = samp[:,0]
                oiihbeta[need] = samp[:,1]
                niihbeta[need] = samp[:,2]
                siihbeta[need] = samp[:,3]
                oiiihbeta[oiiihbeta<oiiihbrange[0]] = -99
                oiiihbeta[oiiihbeta>oiiihbrange[1]] = -99
                need = np.where(oiiihbeta==-99)[0]
                
            # Assume the emission-line priors are uncorrelated.
            #oiiihbeta = rand.uniform(oiiihbrange[0], oiiihbrange[1], nchunk)
            oiidoublet = rand.normal(oiidoublet_meansig[0],
                                     oiidoublet_meansig[1],
                                     nchunk)
            d4000 = self.basemeta['D4000'][chunkindx]

            ewhbeta = 10.0**(np.polyval(self.ewhbetacoeff,d4000)+ 
                             rand.normal(0.0,0.2, nchunk)) # rest-frame, Angstrom
            #ewhbeta = self.ewhbetamog.sample(n_samples=nchunk, random_state=rand)
            ewhbeta *= (self.basemeta['HBETA_LIMIT'][chunkindx]==0)

            # Create a distribution of seeds for the emission-line spectra.
            emseed = rand.random_integers(0, 100*nchunk, nchunk)

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave.astype(float)*(1+redshift[ii])
                restflux = self.baseflux[iobj,:]

                # Normalize to [erg/s/cm2/A, @redshift[ii]]
                rnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                norm = 10.0**(-0.4*rmag[ii])/rnorm['decam2014-r'][0]
                flux = restflux*norm

                # Create an emission-line spectrum with the right [OII] flux [erg/s/cm2]. 
                hbetaflux = self.basemeta['HBETA_CONTINUUM'][iobj]*ewhbeta[ii] 
                zhbetaflux = hbetaflux*norm # [erg/s/cm2]

                emflux, emwave, emline = EM.spectrum(
                    linesigma=vdisp[ii],
                    oiidoublet=oiidoublet[ii],
                    oiiihbeta=oiiihbeta[ii],
                    hbetaflux=zhbetaflux,
                    seed=emseed[ii])
                emflux /= (1+redshift[ii]) # [erg/s/cm2/A, @redshift[ii]]

                if nocontinuum:
                    flux = emflux
                else:
                    flux += emflux

                # Convert [grzW1W2]flux to nanomaggies.
                synthmaggies = self.decamwise.get_ab_maggies(flux, zwave, mask_invalid=True)
                synthnano = [ff*MAG2NANO for ff in synthmaggies[0]] # convert to nanomaggies

                if nocolorcuts:
                    colormask = [True]
                else:
                    colormask = [isBGS(rflux=synthnano[2])]

                if all(colormask):
                    if ((nobj+1)%10)==0:
                        log.debug('Simulating {} template {}/{}'. \
                                  format(self.objtype, nobj+1, nmodel))

                    # (@moustakas) pxs.gauss_blur_matrix is producing lots of
                    # ringing in the emission lines, so deal with it later.
                    
                    # Convolve (just the stellar continuum) and resample.
                    #if nocontinuum is False:
                        #sigma = 1.0+self.basewave*vdisp[ii]/LIGHT
                        #flux = pxs.gauss_blur_matrix(self.pixbound,sigma) * flux
                        #flux = (flux-emflux)*pxs.gauss_blur_matrix(self.pixbound,sigma) + emflux
                    
                    outflux[nobj,:] = resample_flux(self.wave, zwave, flux)

                    meta['TEMPLATEID'][nobj] = nobj
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(synthnano[1])+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(synthnano[2])+22.5
                    meta['ZMAG'][nobj] = -2.5*np.log10(synthnano[4])+22.5
                    meta['W1MAG'][nobj] = -2.5*np.log10(synthnano[6])+22.5
                    meta['W2MAG'][nobj] = -2.5*np.log10(synthnano[7])+22.5
                    meta['DECAM_FLUX'][nobj] = synthnano[:6]
                    meta['WISE_FLUX'][nobj] = synthnano[6:8]
                    meta['HBETAFLUX'][nobj] = zhbetaflux
                    meta['EWHBETA'][nobj] = ewhbeta[ii]
                    meta['OIIIHBETA'][nobj] = oiiihbeta[ii]
                    meta['OIIHBETA'][nobj] = oiihbeta[ii]
                    meta['NIIHBETA'][nobj] = niihbeta[ii]
                    meta['SIIHBETA'][nobj] = siihbeta[ii]
                    meta['OIIDOUBLET'][nobj] = oiidoublet[ii]
                    meta['D4000'][nobj] = d4000[ii]
                    meta['VDISP'][nobj] = vdisp[ii]

                    nobj = nobj+1

                # If we have enough models get out!
                if nobj>=(nmodel-1):
                    break

        return outflux, self.wave, meta
