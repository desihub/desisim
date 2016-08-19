"""
desisim.templates
=================

Functions to simulate spectral templates for DESI.
"""

from __future__ import division, print_function

import os
import sys
import numpy as np

from desisim.io import empty_metatable
from desispec.log import get_logger
log = get_logger()

LIGHT = 2.99792458E5  #- speed of light in km/s

class GaussianMixtureModel(object):
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

class EMSpectrum(object):
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
        recombfile = resource_filename('desisim', 'data/recombination_lines.ecsv')
        forbidfile = resource_filename('desisim', 'data/forbidden_lines.ecsv')
        forbidmogfile = resource_filename('desisim','data/forbidden_mogs.fits')

        if not os.path.isfile(recombfile):
            log.fatal('Required data file {} not found!'.format(recombfile))
            raise IOError
        if not os.path.isfile(forbidfile):
            log.fatal('Required data file {} not found!'.format(forbidfile))
            raise IOError
        if not os.path.isfile(forbidmogfile):
            log.fatal('Required data file {} not found!'.format(forbidmogfile))
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
          seed (int, optional): input seed for the random numbers.

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
        is4959 = np.where(line['name'] == '[OIII]_4959')[0]
        is5007 = np.where(line['name'] == '[OIII]_5007')[0]
        is6548 = np.where(line['name'] == '[NII]_6548')[0]
        is6584 = np.where(line['name'] == '[NII]_6584')[0]
        is6716 = np.where(line['name'] == '[SII]_6716')[0]
        is6731 = np.where(line['name'] == '[SII]_6731')[0]
        #is3869 = np.where(line['name'] == '[NeIII]_3869')[0]
        is3726 = np.where(line['name'] == '[OII]_3726')[0]
        is3729 = np.where(line['name'] == '[OII]_3729')[0]

        # Draw from the MoGs for forbidden lines.
        if oiiihbeta is None or oiihbeta is None or niihbeta is None or siihbeta is None:
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
        factor1 = oiidoublet / (1.0+oiidoublet) # convert 3727-->3726
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
            log.warning('Both hbetaflux and oiiflux were given; using hbetaflux.')
            for ii in range(nline):
                line['flux'][ii] = hbetaflux*line['ratio'][ii]

        # Finally build the emission-line spectrum
        log10sigma = linesigma/LIGHT/np.log(10) # line-width [log-10 Angstrom]
        emspec = np.zeros(len(self.log10wave))
        for ii in range(len(line)):
            amp = line['flux'][ii] / line['wave'][ii] / np.log(10) # line-amplitude [erg/s/cm2/A]
            thislinewave = np.log10(line['wave'][ii] * (1.0+zshift))
            line['amp'][ii] = amp / (np.sqrt(2.0 * np.pi) * log10sigma)  # [erg/s/A]

            # Construct the spectrum [erg/s/cm2/A, rest]
            emspec += amp * np.exp(-0.5 * (self.log10wave-thislinewave)**2 / log10sigma**2) \
                      / (np.sqrt(2.0 * np.pi) * log10sigma)

        return emspec, 10**self.log10wave, line

class GALAXY(object):
    """Base class for generating Monte Carlo spectra of the various flavors of
       galaxies (ELG, BGS, and LRG).

    """
    def __init__(self, objtype='ELG', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, colorcuts_function=None, normfilter='decam2014-r',
                 normline='OII', add_SNeIa=False):
        """Read the appropriate basis continuum templates, filter profiles and
        initialize the output wavelength array.

        Note:
          Only a linearly-spaced output wavelength array is currently supported.

          TODO (@moustakas): Incorporate size and morphological properties.

        Args:
          objtype (str): object type (default 'ELG')
          minwave (float, optional): minimum value of the output wavelength
            array (default 3600 Angstrom).
          maxwave (float, optional): minimum value of the output wavelength
            array (default 10000 Angstrom).
          cdelt (float, optional): spacing of the output wavelength array
            (default 2 Angstrom/pixel).
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments (Angstrom).
          colorcuts_function (function name): Function to use to select
            templates that pass the color-cuts for the specified objtype
            (default None).
          normfilter (str): normalize each spectrum to the magnitude in this
            filter bandpass (default 'decam2014-r').
          normline (str): normalize the emission-line spectrum to the flux in
            this emission line.  The options are 'OII' (for ELGs, the default),
            'HBETA' (for BGS), or None (for LRGs).
          add_SNeIa (boolean, optional): optionally include a random-epoch SNe
            Ia spectrum in the integrated spectrum (default False).

        Attributes:
          wave (numpy.ndarray): Output wavelength array (Angstrom).
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            continuum spectra (erg/s/cm2/A).
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX (Angstrom).
          basemeta (astropy.Table): Table of meta-data [nbase] for each base template.
          pixbound (numpy.ndarray): Pixel boundaries of BASEWAVE (Angstrom).
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          gfilt (speclite.filters instance): DECam2014 g-band FilterSequence
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence
          zfilt (speclite.filters instance): DECam2014 z-band FilterSequence

        Optional Attributes:
          sne_baseflux (numpy.ndarray): Array [sne_nbase,sne_npix] of the base
            rest-frame SNeIa spectra interpolated onto BASEWAVE [erg/s/cm2/A].
          sne_basemeta (astropy.Table): Table of meta-data for each base SNeIa
            spectra [sne_nbase].

        """
        from speclite import filters
        from desisim.io import read_basis_templates
        from desisim import pixelsplines as pxs
        from desispec.interpolation import resample_flux

        self.objtype = objtype.upper()
        self.colorcuts_function = colorcuts_function
        self.normfilter = normfilter
        self.normline = normline

        if self.normline is not None:
            if self.normline.upper() not in ('OII', 'HBETA'):
                log.warning('Unrecognized normline input {}; setting to None.'.format(self.normline))
                self.normline = None
        
        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave) / cdelt+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Optionally read the SNe Ia basis templates and resample.
        self.add_SNeIa = add_SNeIa
        if self.add_SNeIa:
            sne_baseflux1, sne_basewave, sne_basemeta = read_basis_templates(objtype='SNE')
            sne_baseflux = np.zeros((len(sne_basemeta), len(self.basewave)))
            for ii in range(len(sne_basemeta)):
                sne_baseflux[ii, :] = resample_flux(self.basewave, sne_basewave, sne_baseflux1[ii,:])
            self.sne_baseflux = sne_baseflux
            self.sne_basemeta = sne_basemeta

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)

        # Initialize the filter profiles.
        self.gfilt = filters.load_filters('decam2014-g')
        self.rfilt = filters.load_filters('decam2014-r')
        self.zfilt = filters.load_filters('decam2014-z')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

    def vdispblur(self, flux, vdisp=150.0):
        """Convolve an input spectrum with the velocity dispersion."""
        from desisim import pixelsplines as pxs
        
        sigma = 1.0 + (self.basewave * vdisp / LIGHT)
        blurflux = pxs.gauss_blur_matrix(self.pixbound, sigma) * flux

        return blurflux

    def lineratios(self, nobj, oiiihbrange=(-0.5, 0.2), oiidoublet_meansig=(0.73, 0.05),
                   agnlike=False, rand=None):
        """Get the correct number and distribution of the forbidden and [OII] 3726/3729
           doublet emission-line ratios.  Note that the agnlike option is not
           yet supported.

        """
        if rand is None:
            rand = np.random.RandomState()

        if oiidoublet_meansig[1] > 0:
            oiidoublet = rand.normal(oiidoublet_meansig[0], oiidoublet_meansig[1], nobj)
        else:
            oiidoublet = np.repeat(oiidoublet_meansig[0], nobj)

        oiihbeta = np.zeros(nobj)
        niihbeta = np.zeros(nobj)
        siihbeta = np.zeros(nobj)
        oiiihbeta = np.zeros(nobj)-99
        need = np.where(oiiihbeta==-99)[0]
        while len(need) > 0:
            samp = EMSpectrum().forbidmog.sample(len(need), random_state=rand)
            oiiihbeta[need] = samp[:,0]
            oiihbeta[need] = samp[:,1]
            niihbeta[need] = samp[:,2]
            siihbeta[need] = samp[:,3]
            oiiihbeta[oiiihbeta<oiiihbrange[0]] = -99
            oiiihbeta[oiiihbeta>oiiihbrange[1]] = -99
            need = np.where(oiiihbeta==-99)[0]

        return oiidoublet, oiihbeta, niihbeta, siihbeta, oiiihbeta

    def make_galaxy_templates(self, nmodel=100, zrange=(0.6, 1.6), magrange=(21.0, 23.5),
                              oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
                              minlineflux=0.0, sne_rfluxratiorange=(0.01, 0.1),
                              seed=None, redshift=None, mag=None, vdisp=None, 
                              input_meta=None, nocolorcuts=False, nocontinuum=False,
                              agnlike=False):
        """Build Monte Carlo galaxy spectra/templates.

        This function chooses random subsets of the basis continuum spectra (for
        the given galaxy spectral type), constructs an emission-line spectrum
        (if desired), redshifts, convolves by the intrinsic velocity dispersion,
        and then finally normalizes each spectrum to a (generated or input)
        apparent magnitude.

        In detail, each (output) model gets randomly assigned a continuum
        (basis) template; however, if that template doesn't pass the (spectral)
        class-specific color cuts (at the specified redshift), then we iterate
        through the rest of the templates until we find one that *does* pass the
        color-cuts.

        The user also (optionally) has a lot of flexibility over the
        inputs/outputs and can specify any combination of the redshift, velocity
        dispersion, and apparent magnitude (in the normalization filter
        specified in the GALAXY.__init__ method) inputs.  Alternatively, the
        user can pass a complete metadata table, in order to easily regenerate
        spectra on-the-fly (see the documentation for the input_meta argument,
        below).

        Note:
          The default inputs are generally set to values which are appropriate
          for ELGs, so be sure to alter them when generating templates for other
          spectral classes.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.6, 1.6).
          magrange (float, optional): Minimum and maximum magnitude in the
            bandpass specified by self.normfilter.  Defaults to a uniform
            distribution between (21, 23.4) in the r-band.
          oiiihbrange (float, optional): Minimum and maximum logarithmic
            [OIII] 5007/H-beta line-ratio.  Defaults to a uniform distribution
            between (-0.5, 0.2).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=1.9+/-0.15 km/s.
          minlineflux (float, optional): Minimum emission-line flux in the line
            specified by self.normline (default 0 erg/s/cm2).
          sne_rfluxratiorange (float, optional): r-band flux ratio of the SNeIa
            spectrum with respect to the underlying galaxy.  Defaults to a
            uniform distribution between (0.01, 0.1).

          seed (int, optional): Input seed for the random numbers.
          redshift (float, optional): Input/output template redshifts.  Array
            size must equal nmodel.  Ignores zrange input.
          mag (float, optional): Input/output template magnitudes in the band
            specified by self.normfilter.  Array size must equal nmodel.
            Ignores magrange input.
          vdisp (float, optional): Input/output velocity dispersions.  Array
            size must equal nmodel.  Ignores magrange input.
        
          input_meta (astropy.Table): *Input* metadata table with the following
            required columns: TEMPLATEID, SEED, REDSHIFT, VDISP, MAG (where mag
            is specified by self.normfilter).  In addition, if add_SNeIa is True
            then the table must also contain SNE_TEMPLATEID, SNE_EPOCH, and
            SNE_RFLUXRATIO columns.  See desisim.io.empty_metatable for the
            required data type for each column.  If this table is passed then
            all other optional inputs (nmodel, redshift, vdisp, mag, zrange,
            logvdisp_meansig, etc.) are ignored.
        
          nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).
          nocontinuum (bool, optional): Do not include the stellar continuum in
            the output spectrum (useful for testing; default False).  Note that
            this option automatically sets nocolorcuts to True and add_SNeIa to
            False.
          agnlike (bool, optional): Adopt AGN-like emission-line ratios (e.g.,
            for the LRGs and some BGS galaxies) (default False, meaning we adopt
            star-formation-like line-ratios).  Option not yet supported.

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux

        # Basic error checking and some preliminaries.
        if nocontinuum:
            log.warning('Forcing nocolorcuts=True, add_SNeIa=False since nocontinuum=True.')
            nocolorcuts = True
            self.add_SNeIa = False

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        if vdisp is not None:
            if len(vdisp) != nmodel:
                log.fatal('Vdisp must be an nmodel-length array')
                raise ValueError

        npix = len(self.basewave)
        nbase = len(self.basemeta)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            templateseed = input_meta['SEED'].data
            rand = np.random.RandomState(templateseed[0])

            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data
            vdisp = input_meta['VDISP'].data

            vzero = np.where(vdisp <= 0)[0]
            if len(vzero) > 0:
                log.fatal('Velocity disperion is zero or negative in {} spectra!').format(len(vzero))
                raise ValueError

            if self.add_SNeIa:
                sne_tempid = input_meta['SNE_TEMPLATEID']
                sne_epoch = input_meta['SNE_EPOCH']
                sne_rfluxratio = input_meta['SNE_RFLUXRATIO']
                
            nchunk = 1
            nmodel = len(input_meta)
            alltemplateid_chunk = [input_meta['TEMPLATEID'].data.reshape(nmodel, 1)]

            meta = empty_metatable(nmodel, self.objtype, self.add_SNeIa)
        else:
            meta = empty_metatable(nmodel, self.objtype, self.add_SNeIa)

            # Initialize the random seed.
            rand = np.random.RandomState(seed)
            templateseed = rand.randint(2**32, size=nmodel)

            # Shuffle the basis templates and then split them into ~equal
            # chunks, so we can speed up the calculations below.
            chunksize = np.min((nbase, 50))
            nchunk = int(np.ceil(nbase / chunksize))

            alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
            for tempid in alltemplateid:
                rand.shuffle(tempid)
            alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

            # Assign redshift, magnitude, and velocity dispersion priors. 
            if redshift is None:
                redshift = rand.uniform(zrange[0], zrange[1], nmodel)

            if mag is None:
                mag = rand.uniform(magrange[0], magrange[1], nmodel).astype('f4')

            if vdisp is None:
                if logvdisp_meansig[1] > 0:
                    vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nmodel)
                else:
                    vdisp = 10**np.repeat(logvdisp_meansig[0], nmodel)

            # Generate the (optional) distribution of SNe Ia priors.
            if self.add_SNeIa:
                sne_rfluxratio = rand.uniform(sne_rfluxratiorange[0], sne_rfluxratiorange[1], nmodel)
                sne_tempid = rand.randint(0, len(self.sne_basemeta)-1, nmodel)                
                meta['SNE_TEMPLATEID'] = sne_tempid
                meta['SNE_EPOCH'] = self.sne_basemeta['EPOCH'][sne_tempid]
                meta['SNE_RFLUXRATIO'] = sne_rfluxratio

        # Populate some of the metadata table.
        for key, value in zip(('REDSHIFT', 'MAG', 'VDISP', 'SEED'),
                               (redshift, mag, vdisp, templateseed)):
            meta[key] = value

        # Optionally initialize the emission-line objects and line-ratios.
        d4000 = self.basemeta['D4000']

        if self.normline is not None:
            # Initialize the EMSpectrum object with the same wavelength array as
            # the "base" (continuum) templates so that we don't have to resample.
            from desisim.templates import EMSpectrum
            EM = EMSpectrum(log10wave=np.log10(self.basewave))
        
        # Build each spectrum in turn.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]
        for ii in range(nmodel):
            templaterand = np.random.RandomState(templateseed[ii])
                
            zwave = self.basewave.astype(float) * (1.0 + redshift[ii])

            # Optionally generate the emission-line spectrum for this model.
            if self.normline is None:
                emflux = np.zeros(npix)
                normlineflux = np.zeros(nbase)
            else:
                # For speed, build just a single emission-line spectrum for all
                # continuum templates. In detail the line-ratios should
                # correlate with D(4000) or something else.
                oiidoublet, oiihbeta, niihbeta, siihbeta, oiiihbeta = \
                  self.lineratios(nobj=1, oiiihbrange=oiiihbrange,
                                  rand=templaterand, agnlike=agnlike)

                for key, value in zip(('OIIIHBETA', 'OIIHBETA', 'NIIHBETA', 'SIIHBETA', 'OIIDOUBLET'),
                                      (oiiihbeta, oiihbeta, niihbeta, siihbeta, oiidoublet)):
                    meta[key][ii] = value

                if self.normline.upper() == 'OII':
                    ewoii = 10.0**(np.polyval(self.ewoiicoeff, d4000) + # rest-frame EW([OII]), Angstrom
                                   templaterand.normal(0.0, 0.3, nbase)) 
                    normlineflux = self.basemeta['OII_CONTINUUM'].data * ewoii
                    
                    emflux, emwave, emline = EM.spectrum(linesigma=vdisp[ii], seed=templateseed[ii],
                                                         oiidoublet=oiidoublet, oiiihbeta=oiiihbeta,
                                                         oiihbeta=oiihbeta, niihbeta=niihbeta,
                                                         siihbeta=siihbeta, oiiflux=1.0)
                    
                elif self.normline.upper() == 'HBETA':
                    ewhbeta = 10.0**(np.polyval(self.ewhbetacoeff, d4000) + \
                                     templaterand.normal(0.0, 0.2, nbase)) * \
                                     self.basemeta['HBETA_LIMIT'].data # rest-frame H-beta, Angstrom
                    normlineflux = self.basemeta['HBETA_CONTINUUM'].data * ewhbeta
                    
                    emflux, emwave, emline = EM.spectrum(linesigma=vdisp[ii], seed=templateseed[ii],
                                                         oiidoublet=oiidoublet, oiiihbeta=oiiihbeta,
                                                         oiihbeta=oiihbeta, niihbeta=niihbeta,
                                                         siihbeta=siihbeta, hbetaflux=1.0)

                emflux /= (1+redshift[ii]) # [erg/s/cm2/A, @redshift[ii]]

            # Optionally get the SN spectrum and normalization factor.
            if self.add_SNeIa:
                sne_restflux = self.sne_baseflux[sne_tempid[ii], :]
                snenorm = self.rfilt.get_ab_maggies(sne_restflux, zwave)

            for ichunk in range(nchunk):
                log.debug('Simulating {} template {}/{} in chunk {}/{}'. \
                          format(self.objtype, ii+1, nmodel, ichunk, nchunk))
                templateid = alltemplateid_chunk[ichunk][ii, :]
                nbasechunk = len(templateid)
                
                if nocontinuum:
                    restflux = np.tile(emflux, (nbasechunk, 1)) * \
                      np.tile(normlineflux[templateid], (npix, 1)).T 
                else:
                    restflux = self.baseflux[templateid, :] + np.tile(emflux, (nbasechunk, 1)) * \
                      np.tile(normlineflux[templateid], (npix, 1)).T 

                # Optionally add in the SN spectrum.
                if self.add_SNeIa:
                    galnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                    snefactor = galnorm['decam2014-r'].data * sne_rfluxratio[ii]/snenorm['decam2014-r'].data
                    restflux += np.tile(sne_restflux, (nbasechunk, 1)) * np.tile(snefactor, (npix, 1)).T

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                if nocontinuum:
                    magnorm = np.repeat(10**(-0.4*mag[ii]), nbasechunk)
                else:
                    magnorm = 10**(-0.4*mag[ii]) / np.array(maggies[self.normfilter])
                
                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = 1E9 * maggies[key] * magnorm # nanomaggies
                zlineflux = normlineflux[templateid] * magnorm

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = self.colorcuts_function(
                        gflux=synthnano[:, 1],
                        rflux=synthnano[:, 2],
                        zflux=synthnano[:, 4],
                        w1flux=synthnano[:, 6],
                        w2flux=synthnano[:, 7])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table, convolve with the
                # velocity dispersion, resample, and finish up.  Note that the
                # emission lines already have the velocity dispersion
                # line-width.
                if np.any(colormask*(zlineflux >= minlineflux)):
                    this = templaterand.choice(np.where(colormask * (zlineflux >= minlineflux))[0]) # Pick one randomly.
                    tempid = templateid[this]

                    thisemflux = emflux * normlineflux[templateid[this]]
                    blurflux = (self.vdispblur((restflux[this, :] - thisemflux), vdisp[ii]) + \
                                thisemflux) * magnorm[this]

                    outflux[ii, :] = resample_flux(self.wave, zwave, blurflux)

                    meta['TEMPLATEID'][ii] = tempid
                    meta['D4000'][ii] = d4000[tempid]
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    if self.normline is not None:
                        if self.normline == 'OII':
                            meta['OIIFLUX'][ii] = zlineflux[this]
                            meta['EWOII'][ii] = ewoii[tempid]
                        elif self.normline == 'HBETA':
                            meta['HBETAFLUX'][ii] = zlineflux[this]
                            meta['EWHBETA'][ii] = ewhbeta[tempid]
                            
                    break

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta

class ELG(GALAXY):
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs)."""
    
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 add_SNeIa=False, normfilter='decam2014-r', colorcuts_function=None):
        """Initialize the ELG class.  See the GALAXY.__init__ method for documentation
         on the arguments plus the inherited attributes.

        Note:
          We assume that the ELG templates will always be normalized in the
          DECam r-band filter and that the emission-line spectra will be
          normalized to the integrated [OII] emission-line flux.

        Args:
          
        Attributes:
          ewoiicoeff (float, array): empirically derived coefficients to map
            D(4000) to EW([OII]).
        
        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isELG as colorcuts_function
            
        super(ELG, self).__init__(objtype='ELG', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                  normfilter=normfilter, normline='OII', add_SNeIa=add_SNeIa)

        self.ewoiicoeff = [1.34323087, -5.02866474, 5.43842874]

    def make_templates(self, nmodel=100, zrange=(0.6, 1.6), rmagrange=(21.0, 23.4),
                       oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
                       minoiiflux=0.0, sne_rfluxratiorange=(0.1, 1.0), redshift=None,
                       mag=None, vdisp=None, seed=None, input_meta=None, nocolorcuts=False,
                       nocontinuum=False, agnlike=False):
        """Build Monte Carlo ELG spectra/templates.

        See the GALAXY.make_galaxy_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the ELG class.

        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (21,
            23.4).
          oiiihbrange (float, optional): Minimum and maximum logarithmic [OIII]
            5007/H-beta line-ratio.  Defaults to a uniform distribution between
            (-0.5, 0.2).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(1.9+/-0.15) km/s
          minoiiflux (float, optional): Minimum [OII] 3727 flux (default 0.0
            erg/s/cm2).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """

        outflux, wave, meta = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=rmagrange,
                                                         oiiihbrange=oiiihbrange, logvdisp_meansig=logvdisp_meansig,
                                                         minlineflux=minoiiflux, redshift=redshift, vdisp=vdisp,
                                                         mag=mag, sne_rfluxratiorange=sne_rfluxratiorange,
                                                         seed=seed, input_meta=input_meta, nocolorcuts=nocolorcuts,
                                                         nocontinuum=nocontinuum, agnlike=agnlike)

        return outflux, wave, meta

class BGS(GALAXY):
    """Generate Monte Carlo spectra of bright galaxy survey galaxies (BGSs)."""
    
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 add_SNeIa=False, normfilter='decam2014-r', colorcuts_function=None):
        """Initialize the BGS class.  See the GALAXY.__init__ method for documentation
         on the arguments plus the inherited attributes.

        Note:
          We assume that the BGS templates will always be normalized in the
          DECam r-band filter and that the emission-line spectra will be
          normalized to the integrated H-beta emission-line flux.

        Args:

        Attributes:
          ewhbetacoeff (float, array): empirically derived coefficients to map
            D(4000) to EW(H-beta).

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isBGS as colorcuts_function

        super(BGS, self).__init__(objtype='BGS', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                  normfilter=normfilter, normline='HBETA', add_SNeIa=add_SNeIa)

        self.ewhbetacoeff = [1.28520974, -4.94408026, 4.9617704]

    def make_templates(self, nmodel=100, zrange=(0.01, 0.4), rmagrange=(15.0, 19.5),
                       oiiihbrange=(-1.3, 0.6), logvdisp_meansig=(2.0, 0.17),
                       minhbetaflux=0.0, sne_rfluxratiorange=(0.1, 1.0), redshift=None,
                       mag=None, vdisp=None, seed=None, input_meta=None, nocolorcuts=False,
                       nocontinuum=False, agnlike=False):
        """Build Monte Carlo BGS spectra/templates.

         See the GALAXY.make_galaxy_templates function for documentation on the
         arguments and inherited attributes.  Here we only document the
         arguments which are specific to the BGS class.

        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (15,
            19.5).
          oiiihbrange (float, optional): Minimum and maximum logarithmic [OIII]
            5007/H-beta line-ratio.  Defaults to a uniform distribution between
            (-1.3, 0.6).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(2.0+/-0.17) km/s
          minhbetaflux (float, optional): Minimum H-beta flux (default 0.0
            erg/s/cm2).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """

        outflux, wave, meta = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=rmagrange,
                                                         oiiihbrange=oiiihbrange, logvdisp_meansig=logvdisp_meansig,
                                                         minlineflux=minhbetaflux, redshift=redshift, vdisp=vdisp,
                                                         mag=mag, sne_rfluxratiorange=sne_rfluxratiorange,
                                                         seed=seed, input_meta=input_meta, nocolorcuts=nocolorcuts,
                                                         nocontinuum=nocontinuum, agnlike=agnlike)

        return outflux, wave, meta
    
class LRG(GALAXY):
    """Generate Monte Carlo spectra of luminous red galaxies (LRGs)."""
    
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 add_SNeIa=False, normfilter='decam2014-z', colorcuts_function=None):
        """Initialize the LRG class.  See the GALAXY.__init__ method for documentation
        on the arguments plus the inherited attributes.

        Note:
          We assume that the LRG templates will always be normalized in the
          DECam z-band filter.  Emission lines (with presumably AGN-like
          line-ratios) are not yet included.

        Args:

        Attributes:

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isLRG as colorcuts_function

        super(LRG, self).__init__(objtype='LRG', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                  normfilter=normfilter, normline=None, add_SNeIa=add_SNeIa)

    def make_templates(self, nmodel=100, zrange=(0.5, 1.0), zmagrange=(19.0, 20.5),
                       logvdisp_meansig=(2.3, 0.1), sne_rfluxratiorange=(0.1, 1.0),
                       redshift=None, mag=None, vdisp=None, seed=None,
                       input_meta=None, nocolorcuts=False, agnlike=False):
        """Build Monte Carlo BGS spectra/templates.

         See the GALAXY.make_galaxy_templates function for documentation on the
         arguments and inherited attributes.  Here we only document the
         arguments which are specific to the LRG class.

        Args:
          zmagrange (float, optional): Minimum and maximum DECam z-band (AB)
            magnitude range.  Defaults to a uniform distribution between (19,
            20.5).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(2.3+/-0.1) km/s
          agnlike (bool, optional): adopt AGN-like emission-line ratios (not yet
            supported; defaults False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """

        outflux, wave, meta = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=zmagrange,
                                                         logvdisp_meansig=logvdisp_meansig, redshift=redshift,
                                                         vdisp=vdisp, mag=mag, sne_rfluxratiorange=sne_rfluxratiorange,
                                                         seed=seed, input_meta=input_meta, nocolorcuts=nocolorcuts,
                                                         agnlike=agnlike)

        # Pack into the metadata table some additional information.
        good = np.where(meta['TEMPLATEID'] != -1)[0]
        if len(good) > 0:
            meta['ZMETAL'][good] = self.basemeta[meta['TEMPLATEID'][good]]['ZMETAL']
            meta['AGE'][good] = self.basemeta[meta['TEMPLATEID'][good]]['AGE']

        return outflux, wave, meta

class SUPERSTAR(object):
    """Base class for generating Monte Carlo spectra of the various flavors of stars.""" 

    def __init__(self, objtype='STAR', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, colorcuts_function=None, normfilter='decam2014-r'):
        """Read the appropriate basis continuum templates, filter profiles and
        initialize the output wavelength array.

        Note:
          Only a linearly-spaced output wavelength array is currently supported.

        Args:
          objtype (str): type of object to simulate (default STAR)
          minwave (float, optional): minimum value of the output wavelength
            array (default 3600 Angstrom).
          maxwave (float, optional): minimum value of the output wavelength
            array (default 10000 Angstrom).
          cdelt (float, optional): spacing of the output wavelength array
            (default 2 Angstrom/pixel).
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments (Angstrom).
          colorcuts_function (function name): Function to use to select
            templates that pass the color-cuts for the specified objtype
            (default None).
          normfilter (str): normalize each spectrum to the magnitude in this
            filter bandpass (default 'decam2014-r').

        Attributes:
          wave (numpy.ndarray): Output wavelength array (Angstrom).
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            continuum spectra (erg/s/cm2/A).
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX (Angstrom).
          basemeta (astropy.Table): Table of meta-data [nbase] for each base template.
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          gfilt (speclite.filters instance): DECam2014 g-band FilterSequence
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence
          zfilt (speclite.filters instance): DECam2014 z-band FilterSequence

        """
        from speclite import filters
        from desisim.io import read_basis_templates

        self.objtype = objtype.upper()
        self.colorcuts_function = colorcuts_function
        self.normfilter = normfilter

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave) / cdelt+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        # Read the rest-frame continuum basis spectra.
        baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.gfilt = filters.load_filters('decam2014-g')
        self.rfilt = filters.load_filters('decam2014-r')
        self.zfilt = filters.load_filters('decam2014-z')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

    def make_star_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                            magrange=(18.0, 23.5), seed=None, redshift=None,
                            mag=None, input_meta=None, nocolorcuts=False):

        """Build Monte Carlo spectra/templates for various flavors of stars.

        This function chooses random subsets of the continuum spectra for the
        type of star specified by OBJTYPE, adds radial velocity jitter, applies
        the targeting color-cuts, and then normalizes the spectrum to the
        magnitude in the given filter.

        The user also (optionally) has a lot of flexibility over the
        inputs/outputs and can specify any combination of the radial velocity
        and apparent magnitude (in the normalization filter specified in the
        GALAXY.__init__ method) inputs.  Alternatively, the user can pass a
        complete metadata table, in order to easily regenerate spectra
        on-the-fly (see the documentation for the input_meta argument, below).

        Note:
          The default inputs are generally set to values which are appropriate
          for generic stars, so be sure to alter them when generating templates
          for other spectral classes.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          vrad_meansig (float, optional): Mean and sigma (standard deviation) of the
            radial velocity "jitter" (in km/s) that should be included in each
            spectrum.  Defaults to a normal distribution with a mean of zero and
            sigma of 200 km/s.
          magrange (float, optional): Minimum and maximum magnitude in the
            bandpass specified by self.normfilter.  Defaults to a uniform
            distribution between (18, 23.5) in the r-band.
          seed (int, optional): input seed for the random numbers.        
          redshift (float, optional): Input/output (dimensionless) radial
            velocity.  Array size must equal nmodel.  Ignores vrad_meansig
            input.
          mag (float, optional): Input/output template magnitudes in the band
            specified by self.normfilter.  Array size must equal nmodel.
            Ignores magrange input.

          input_meta (astropy.Table): *Input* metadata table with the following
            required columns: TEMPLATEID, SEED, REDSHIFT, and MAG (where mag is
            specified by self.normfilter).  See desisim.io.empty_metatable for
            the required data type for each column.  If this table is passed
            then all other optional inputs (nmodel, redshift, mag, vrad_meansig,
            etc.) are ignored.
        
          nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux

        # Basic error checking and some preliminaries.
        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        npix = len(self.basewave)
        nbase = len(self.basemeta)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            templateseed = input_meta['SEED'].data
            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data

            nchunk = 1
            nmodel = len(input_meta)
            alltemplateid_chunk = [input_meta['TEMPLATEID'].data.reshape(nmodel, 1)]

            meta = empty_metatable(nmodel, self.objtype)
        else:
            meta = empty_metatable(nmodel, self.objtype)
            
            # Initialize the random seed.
            rand = np.random.RandomState(seed)
            templateseed = rand.randint(2**32, size=nmodel)

            # Shuffle the basis templates and then split them into ~equal chunks, so
            # we can speed up the calculations below.
            chunksize = np.min((nbase, 50))
            nchunk = int(np.ceil(nbase / chunksize))

            alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
            for tempid in alltemplateid:
                rand.shuffle(tempid)
            alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

            # Assign radial velocity and magnitude priors.
            if redshift is None:
                if vrad_meansig[1] > 0:
                    vrad = rand.normal(vrad_meansig[0], vrad_meansig[1], nmodel)
                else:
                    vrad = np.repeat(vrad_meansig[0], nmodel)
                    
                redshift = np.array(vrad) / LIGHT

            if mag is None:
                mag = rand.uniform(magrange[0], magrange[1], nmodel).astype('f4')

        # Populate some of the metadata table.
        for key, value in zip(('REDSHIFT', 'MAG', 'SEED'),
                               (redshift, mag, templateseed)):
            meta[key] = value

        # Build each spectrum in turn.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]
        for ii in range(nmodel):
            zwave = self.basewave.astype(float)*(1.0 + redshift[ii])

            for ichunk in range(nchunk):
                log.debug('Simulating {} template {}/{} in chunk {}/{}'. \
                          format(self.objtype, ii+1, nmodel, ichunk, nchunk))
                templateid = alltemplateid_chunk[ichunk][ii, :]
                nbasechunk = len(templateid)
                
                restflux = self.baseflux[templateid, :]

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                magnorm = 10**(-0.4*mag[ii]) / np.array(maggies[self.normfilter])

                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = 1E9 * maggies[key] * magnorm

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = self.colorcuts_function(
                        gflux=synthnano[:, 1],
                        rflux=synthnano[:, 2],
                        zflux=synthnano[:, 4],
                        w1flux=synthnano[:, 6],
                        w2flux=synthnano[:, 7])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask):
                    templaterand = np.random.RandomState(templateseed[ii])
                        
                    this = templaterand.choice(np.where(colormask)[0]) # Pick one randomly.
                    tempid = templateid[this]

                    outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :]) * magnorm[this]

                    meta['TEMPLATEID'][ii] = tempid
                    meta['TEFF'][ii] = self.basemeta['TEFF'][tempid]
                    meta['LOGG'][ii] = self.basemeta['LOGG'][tempid]
                    if 'FEH' in self.basemeta.columns:
                        meta['FEH'][ii] = self.basemeta['FEH'][tempid]
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    break

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta

class STAR(SUPERSTAR):
    """Generate Monte Carlo spectra of generic stars."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None):
        """Initialize the STAR class.  See the SUPERSTAR.__init__ method for
        documentation on the arguments plus the inherited attributes.

        Note:
          We assume that the STAR templates will always be normalized in the
          DECam r-band filter.

        Args:
          
        Attributes:
        
        Raises:

        """
        super(STAR, self).__init__(objtype='STAR', minwave=minwave, maxwave=maxwave,
                                   cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                   normfilter=normfilter)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       rmagrange=(18.0, 23.5), seed=None, redshift=None,
                       mag=None, input_meta=None):
        """Build Monte Carlo spectra/templates for generic stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the STAR class.
        
        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (18,
            23.5).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta)
        return outflux, wave, meta
    
class FSTD(SUPERSTAR):
    """Generate Monte Carlo spectra of (metal-poor, main sequence turnoff) standard
    stars (FSTD).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None):
        """Initialize the FSTD class.  See the SUPERSTAR.__init__ method for
        documentation on the arguments plus the inherited attributes.

        Note:
          We assume that the FSTD templates will always be normalized in the
          DECam r-band filter.

        Args:
          
        Attributes:
        
        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isFSTD_colors as colorcuts_function
        
        super(FSTD, self).__init__(objtype='FSTD', minwave=minwave, maxwave=maxwave,
                                   cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                   normfilter=normfilter)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       rmagrange=(16.0, 19.0), seed=None, redshift=None,
                       mag=None, input_meta=None, nocolorcuts=False):
        """Build Monte Carlo spectra/templates for FSTD stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the FSTD class.
        
        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,
            19).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta, nocolorcuts=nocolorcuts)
        return outflux, wave, meta
    
class MWS_STAR(SUPERSTAR):
    """Generate Monte Carlo spectra of Milky Way Survey (magnitude-limited)
    stars.

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None):                 
        """Initialize the MWS_STAR class.  See the SUPERSTAR.__init__ method for
        documentation on the arguments plus the inherited attributes.

        Note:
          We assume that the MWS_STAR templates will always be normalized in the
          DECam r-band filter.

        Args:
          
        Attributes:
        
        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isMWSSTAR_colors as colorcuts_function
        super(MWS_STAR, self).__init__(objtype='MWS_STAR', minwave=minwave, maxwave=maxwave,
                                       cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                       normfilter=normfilter)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       rmagrange=(16.0, 20.0), seed=None, redshift=None,
                       mag=None, input_meta=None, nocolorcuts=False):
        """Build Monte Carlo spectra/templates for MWS_STAR stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the MWS_STAR class.
        
        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,
            20).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta, nocolorcuts=nocolorcuts)
        return outflux, wave, meta
    
class WD(SUPERSTAR):
    """Generate Monte Carlo spectra of white dwarfs."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 normfilter='decam2014-g', colorcuts_function=None):                 
        """Initialize the WD class.  See the SUPERSTAR.__init__ method for documentation
        on the arguments plus the inherited attributes.

        Note:
          We assume that the WD templates will always be normalized in the
          DECam g-band filter.

        Args:
          
        Attributes:
        
        Raises:

        """

        super(WD, self).__init__(objtype='WD', minwave=minwave, maxwave=maxwave,
                                 cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                 normfilter=normfilter)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       gmagrange=(16.0, 19.0), seed=None, redshift=None,
                       mag=None, input_meta=None, nocolorcuts=False):
        """Build Monte Carlo spectra/templates for WD stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the WD class.
        
        Args:
          gmagrange (float, optional): Minimum and maximum DECam g-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,
            19).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=gmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta, nocolorcuts=nocolorcuts)
        return outflux, wave, meta
    
class QSO():
    """Generate Monte Carlo spectra of quasars (QSOs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None, z_wind=0.2):
        """Read the QSO basis continuum templates, filter profiles and initialize the
           output wavelength array.

        Note:
          Only a linearly-spaced output wavelength array is currently supported
          and the basis templates are only defined in the range 3500-10000 A
          (observed).

        Args:
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].
          colorcuts_function (function name): Function to use to select
            templates that pass the color-cuts.
          normfilter (str): normalize each spectrum to the magnitude in this
            filter bandpass (default 'decam2014-r').
          z_wind (float, optional): Redshift window for sampling (defaults to
            0.2).

        Attributes:
          objtype (str): 'QSO'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            QSO continuum spectra (erg/s/cm2/A).
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX (Angstrom).
          basemeta (astropy.Table): Table of meta-data [nbase] for each base template.
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          gilt (speclite.filters instance): DECam2014 g-band FilterSequence
          rilt (speclite.filters instance): DECam2014 r-band FilterSequence
          zilt (speclite.filters instance): DECam2014 z-band FilterSequence

        """
        from speclite import filters
        from desisim.io import find_basis_template

        self.objtype = 'QSO'
        
        if colorcuts_function is None:
            from desitarget.cuts import isQSO as colorcuts_function
            self.colorcuts_function = colorcuts_function
            
        log.warning('Color-cuts not yet supported for QSOs!')
        self.colorcuts_function = None 

        self.normfilter = normfilter
        
        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave) / cdelt+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        # Find the basis files.
        self.basis_file = find_basis_template('qso')
        self.z_wind = z_wind

        # Initialize the filter profiles.
        self.gfilt = filters.load_filters('decam2014-g')
        self.rfilt = filters.load_filters('decam2014-r')
        self.zfilt = filters.load_filters('decam2014-z')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

    def make_templates(self, nmodel=100, zrange=(0.5, 4.0), rmagrange=(21.0, 23.0),
                       seed=None, redshift=None, mag=None, input_meta=None,
                       nocolorcuts=False):
        """Build Monte Carlo QSO spectra/templates.

        This function generates QSO spectra on-the-fly using PCA decomposition
        coefficients of SDSS and BOSS QSO spectra.  The default is to generate
        flat, uncorrelated priors on redshift and apparent magnitude (in the
        bandpass specified by self.normfilter).

        However, the user also (optionally) has flexibility over the
        inputs/outputs and can specify any combination of the redshift and
        output apparent magnitude.  Alternatively, the user can pass a complete
        metadata table, in order to easily regenerate spectra on-the-fly (see
        the documentation for the input_meta argument, below).

        Note:
          The templates are only defined in the range 3500-10000 A (observed)
          and we do not yet apply proper color-cuts to "select" DESI QSOs.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5, 4.0).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (21,
            23.0).
          seed (int, optional): input seed for the random numbers.
          redshift (float, optional): Input/output template redshifts.  Array
            size must equal nmodel.  Ignores zrange input.
          mag (float, optional): Input/output template magnitudes in the band
            specified by self.normfilter.  Array size must equal nmodel.
            Ignores rmagrange input.
          input_meta (astropy.Table): *Input* metadata table with the following
            required columns: SEED, REDSHIFT, and MAG (where mag is specified by
            self.normfilter).  See desisim.io.empty_metatable for the required
            data type for each column.  If this table is passed then all other
            optional inputs (nmodel, redshift, mag, zrange, rmagrange, etc.) are
            ignored.
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).
          wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux
        from desisim.qso_template import desi_qso_templ as dqt

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError
            zrange = (np.min(redshift), np.max(redshift))

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        # Optionally unpack a metadata table.
        if input_meta is not None:
            nmodel = len(input_meta)
            
            templateseed = input_meta['SEED'].data
            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data

            meta = empty_metatable(nmodel, self.objtype)
        else:
            meta = empty_metatable(nmodel, self.objtype)
            
            # Initialize the random seed.
            rand = np.random.RandomState(seed)
            templateseed = rand.randint(2**32, size=nmodel)

            # Assign redshift and magnitude priors.
            if redshift is None:
                redshift = rand.uniform(zrange[0], zrange[1], nmodel)

            if mag is None:
                mag = rand.uniform(rmagrange[0], rmagrange[1], nmodel).astype('f4')

        # Populate some of the metadata table.
        meta['TEMPLATEID'] = np.arange(nmodel)
        for key, value in zip(('REDSHIFT', 'MAG', 'SEED'),
                               (redshift, mag, templateseed)):
            meta[key] = value
            
        # Build each spectrum in turn.
        zwave = self.wave # [observed-frame, Angstrom]
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        for ii in range(nmodel):
            log.debug('Simulating {} template {}/{}.'.format(self.objtype, ii+1, nmodel))
            templaterand = np.random.RandomState(templateseed[ii])
            
            _, final_flux, redshifts = dqt.desi_qso_templates(
                z_wind=self.z_wind, N_perz=50, rstate=templaterand, 
                redshift=redshift[ii], rebin_wave=zwave, no_write=True)
            restflux = final_flux.T
            nmade = np.shape(restflux)[0]

            # Synthesize photometry to determine which models will pass the
            # color-cuts.  We have to temporarily pad because the spectra don't
            # go red enough.
            padflux, padzwave = self.rfilt.pad_spectrum(restflux, zwave, method='edge')
            maggies = self.decamwise.get_ab_maggies(padflux, padzwave, mask_invalid=True)
            magnorm = 10**(-0.4*mag[ii]) / np.array(maggies[self.normfilter])

            synthnano = np.zeros((nmade, len(self.decamwise)))
            for ff, key in enumerate(maggies.columns):
                synthnano[:, ff] = 1E9 * maggies[key] * magnorm

            if nocolorcuts or self.colorcuts_function is None:
                colormask = np.repeat(1, nmade)
            else:
                colormask = self.colorcuts_function(
                    gflux=synthnano[1],
                    rflux=synthnano[2],
                    zflux=synthnano[4],
                    w1flux=synthnano[6],
                    w2flux=synthnano[7])

            # If the color-cuts pass then populate the output flux vector
            # (suitably normalized) and metadata table and finish up.
            if np.any(colormask):
              this = templaterand.choice(np.where(colormask)[0]) # Pick one randomly.
              outflux[ii, :] = restflux[this, :] * magnorm[this]

              # Temporary hack until the models go redder.
              meta['DECAM_FLUX'][ii] = synthnano[this, :6]
              meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))
                        
        return outflux, self.wave, meta
