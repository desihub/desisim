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

LIGHT = 2.99792458E5  #- speed of light in km/s

class EMSpectrum(object):
    """Construct a complete nebular emission-line spectrum.

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
    def __init__(self, minwave=3650.0, maxwave=7075.0, cdelt_kms=20.0, log10wave=None):
        from pkg_resources import resource_filename
        from astropy.table import Table, Column, vstack
        from desiutil.sklearn import GaussianMixtureModel

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
            Tuple of (emspec, wave, line), where
            emspec is an Array [npix] of flux values [erg/s/cm2/A];
            wave is an Array [npix] of vacuum wavelengths corresponding to
            FLUX [Angstrom, linear spacing];
            line is a Table of emission-line parameters used to generate
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
    def __init__(self, objtype='ELG', minwave=3600.0, maxwave=10000.0, cdelt=0.2,
                 wave=None, colorcuts_function=None, normfilter='decam2014-r',
                 normline='OII', fracvdisp=(0.1, 40), baseflux=None, basewave=None,
                 basemeta=None, add_SNeIa=False):
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
            templates that pass the color-cuts for the specified objtype Note
            that this argument can also be a tuple of more than one selection
            function to apply (e.g., desitarget.cuts.isBGS_faint and
            desitarget.cuts.isBGS_bright) which will be applied in sequence
            (default None).
          normfilter (str): normalize each spectrum to the magnitude in this
            filter bandpass (default 'decam2014-r').
          normline (str): normalize the emission-line spectrum to the flux in
            this emission line.  The options are 'OII' (for ELGs, the default),
            'HBETA' (for BGS), or None (for LRGs).
          fracvdisp (tuple): two-element array which gives the fraction and
            absolute number of unique velocity dispersion values.  For example,
            the default (0.1, 40) means there will be either int(0.1*nmodel) or
            40 unique values, where nmodel is defined in
            GALAXY.make_galaxy_templates, below.
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
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence.
          normfilt (speclite.filters instance): FilterSequence of self.normfilter.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.

        Optional Attributes:
          sne_baseflux (numpy.ndarray): Array [sne_nbase,sne_npix] of the base
            rest-frame SNeIa spectra interpolated onto BASEWAVE [erg/s/cm2/A].
          sne_basemeta (astropy.Table): Table of meta-data for each base SNeIa
            spectra [sne_nbase].

        """
        from speclite import filters
        from desisim import pixelsplines as pxs

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

        # Read the rest-frame continuum basis spectra, if not specified.
        if baseflux is None or basewave is None or basemeta is None:
            from desisim.io import read_basis_templates
            baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Optionally read the SNe Ia basis templates and resample.
        self.add_SNeIa = add_SNeIa
        if self.add_SNeIa:
            from desispec.interpolation import resample_flux
            sne_baseflux1, sne_basewave, sne_basemeta = read_basis_templates(objtype='SNE')
            sne_baseflux = np.zeros((len(sne_basemeta), len(self.basewave)))
            for ii in range(len(sne_basemeta)):
                sne_baseflux[ii, :] = resample_flux(self.basewave, sne_basewave,
                                                    sne_baseflux1[ii, :], extrapolate=True)
            self.sne_baseflux = sne_baseflux
            self.sne_basemeta = sne_basemeta

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)
        self.fracvdisp = fracvdisp

        # Initialize the filter profiles.
        self.rfilt = filters.load_filters('decam2014-r')
        self.normfilt = filters.load_filters(self.normfilter)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

    def _blurmatrix(self, vdisp, log=None):
        """Pre-compute the blur_matrix as a dictionary keyed by each unique value of
        vdisp.

        """
        from desisim import pixelsplines as pxs

        uvdisp = list(set(vdisp))
        log.debug('Populating blur matrix with {} unique velocity dispersion values.'.format(len(uvdisp)))
        if len(uvdisp) > self.fracvdisp[1]:
            log.warning('Slow code ahead! Consider reducing the number of input velocity dispersion values from {}.'.format(
                len(uvdisp)))

        blurmatrix = dict()
        for uvv in uvdisp:
            sigma = 1.0 + (self.basewave * uvv / LIGHT)
            blurmatrix[uvv] = pxs.gauss_blur_matrix(self.pixbound, sigma).astype('f4')

        return blurmatrix

    def lineratios(self, nobj, oiiihbrange=(-0.5, 0.2), oiidoublet_meansig=(0.73, 0.05),
                   agnlike=False, rand=None):
        """Get the correct number and distribution of the forbidden and [OII] 3726/3729
        doublet emission-line ratios.  Note that the agnlike option is not yet
        supported.

        Supporting oiiihbrange needs a different (fast) approach.  Suppressing
        the code below for now until it's needed.

        """
        if agnlike:
            raise NotImplementedError('agnlike option not yet implemented')

        if rand is None:
            rand = np.random.RandomState()

        if oiidoublet_meansig[1] > 0:
            oiidoublet = rand.normal(oiidoublet_meansig[0], oiidoublet_meansig[1], nobj)
        else:
            oiidoublet = np.repeat(oiidoublet_meansig[0], nobj)

        # Sample from the MoG.  This is not strictly correct because it ignores
        # the prior on [OIII]/Hbeta, but let's revisit that later.
        samp = EMSpectrum().forbidmog.sample(nobj, random_state=rand)
        oiiihbeta = samp[:, 0]
        oiihbeta = samp[:, 1]
        niihbeta = samp[:, 2]
        siihbeta = samp[:, 3]

        return oiidoublet, oiihbeta, niihbeta, siihbeta, oiiihbeta

    def make_galaxy_templates(self, nmodel=100, zrange=(0.6, 1.6), magrange=(21.0, 23.5),
                              oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
                              minlineflux=0.0, sne_rfluxratiorange=(0.01, 0.1),
                              seed=None, redshift=None, mag=None, vdisp=None,
                              input_meta=None, nocolorcuts=False, nocontinuum=False,
                              agnlike=False, novdisp=False, restframe=False, verbose=False):
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
          novdisp (bool, optional): Do not velocity-blur the spectrum (default
            False).
          agnlike (bool, optional): Adopt AGN-like emission-line ratios (e.g.,
            for the LRGs and some BGS galaxies) (default False, meaning we adopt
            star-formation-like line-ratios).  Option not yet supported.
          restframe (bool, optional): If True, return full resolution restframe
            templates instead of resampled observer frame.
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux
        from desiutil.log import get_logger, DEBUG

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        # Basic error checking and some preliminaries.
        if nocontinuum:
            log.warning('Forcing nocolorcuts=True, add_SNeIa=False since nocontinuum=True.')
            nocolorcuts = True
            self.add_SNeIa = False

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
                log.fatal('Velocity dispersion is zero or negative!')
                raise ValueError

            if self.add_SNeIa:
                sne_tempid = input_meta['SNE_TEMPLATEID']
                sne_epoch = input_meta['SNE_EPOCH']
                sne_rfluxratio = input_meta['SNE_RFLUXRATIO']

            nchunk = 1
            nmodel = len(input_meta)
            alltemplateid_chunk = [input_meta['TEMPLATEID'].data.reshape(nmodel, 1)]

            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype, add_SNeIa=self.add_SNeIa)
        else:
            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype, add_SNeIa=self.add_SNeIa)

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
                # Limit the number of unique velocity dispersion values.
                nvdisp = int(np.max( ( np.min(
                    ( np.round(nmodel * self.fracvdisp[0]), self.fracvdisp[1] ) ), 1 ) ))
                if logvdisp_meansig[1] > 0:
                    vvdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nvdisp)
                else:
                    vvdisp = 10**np.repeat(logvdisp_meansig[0], nvdisp)
                vdisp = rand.choice(vvdisp, nmodel)

            # Generate the (optional) distribution of SNe Ia priors.
            if self.add_SNeIa:
                sne_rfluxratio = rand.uniform(sne_rfluxratiorange[0], sne_rfluxratiorange[1], nmodel)
                sne_tempid = rand.randint(0, len(self.sne_basemeta)-1, nmodel)
                meta['SNE_TEMPLATEID'] = sne_tempid
                meta['SNE_EPOCH'] = self.sne_basemeta['EPOCH'][sne_tempid]
                meta['SNE_RFLUXRATIO'] = sne_rfluxratio

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

        # Precompute the velocity dispersion convolution matrix for each unique
        # value of vdisp.
        if nocontinuum or novdisp:
            pass
        else:
            blurmatrix = self._blurmatrix(vdisp, log=log)

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
        if restframe:
            outflux = np.zeros([nmodel, len(self.basewave)])
        else:
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
                                     (self.basemeta['HBETA_LIMIT'].data == 0) # rest-frame H-beta, Angstrom
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
                if ii % 100 == 0 and ii > 0:
                    log.debug('Simulating {} template {}/{} in chunk {}/{}.'. \
                              format(self.objtype, ii, nmodel, ichunk+1, nchunk))
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
                    if self.normfilter in self.decamwise.names:
                        normmaggies = np.array(maggies[self.normfilter])
                    else:
                        normmaggies = np.array(self.normfilt.get_ab_maggies(
                            restflux, zwave, mask_invalid=True)[self.normfilter])
                    magnorm = 10**(-0.4*mag[ii]) / normmaggies

                synthnano = dict()
                for key in maggies.columns:
                    synthnano[key] = 1E9 * maggies[key] * magnorm # nanomaggies
                zlineflux = normlineflux[templateid] * magnorm

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    if isinstance(self.colorcuts_function, (tuple, list)):
                        _colormask = []
                        for cf in self.colorcuts_function:
                            _colormask.append(cf(
                                gflux=synthnano['decam2014-g'],
                                rflux=synthnano['decam2014-r'],
                                zflux=synthnano['decam2014-z'],
                                w1flux=synthnano['wise2010-W1'],
                                w2flux=synthnano['wise2010-W2']))
                        colormask = np.any( np.ma.getdata(np.vstack(_colormask)), axis=0)
                    else:
                        colormask = self.colorcuts_function(
                            gflux=synthnano['decam2014-g'],
                            rflux=synthnano['decam2014-r'],
                            zflux=synthnano['decam2014-z'],
                            w1flux=synthnano['wise2010-W1'],
                            w2flux=synthnano['wise2010-W2'])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table, convolve with the
                # velocity dispersion, resample, and finish up.  Note that the
                # emission lines already have the velocity dispersion
                # line-width.
                if np.any(colormask*(zlineflux >= minlineflux)):
                    this = templaterand.choice(np.where(colormask * (zlineflux >= minlineflux))[0]) # Pick one randomly.
                    tempid = templateid[this]

                    thisemflux = emflux * normlineflux[templateid[this]]
                    if nocontinuum or novdisp:
                        blurflux = restflux[this, :] * magnorm[this]
                    else:
                        blurflux = ((blurmatrix[vdisp[ii]] * (restflux[this, :] - thisemflux)) +
                                    thisemflux) * magnorm[this]

                    if restframe:
                        outflux[ii, :] = blurflux
                    else:
                        outflux[ii, :] = resample_flux(self.wave, zwave, blurflux, extrapolate=True)

                    meta['TEMPLATEID'][ii] = tempid
                    meta['D4000'][ii] = d4000[tempid]
                    meta['FLUX_G'][ii] = synthnano['decam2014-g'][this]
                    meta['FLUX_R'][ii] = synthnano['decam2014-r'][this]
                    meta['FLUX_Z'][ii] = synthnano['decam2014-z'][this]
                    meta['FLUX_W1'][ii] = synthnano['wise2010-W1'][this]
                    meta['FLUX_W2'][ii] = synthnano['wise2010-W2'][this]

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

        if restframe:
            return 1e17 * outflux, self.basewave, meta
        else:
            return 1e17 * outflux, self.wave, meta

class ELG(GALAXY):
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 add_SNeIa=False, normfilter='decam2014-r', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
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
                                  normfilter=normfilter, normline='OII', add_SNeIa=add_SNeIa,
                                  baseflux=baseflux, basewave=basewave, basemeta=basemeta)

        self.ewoiicoeff = [1.34323087, -5.02866474, 5.43842874]

    def make_templates(self, nmodel=100, zrange=(0.6, 1.6), rmagrange=(21.0, 23.4),
                       oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
                       minoiiflux=0.0, sne_rfluxratiorange=(0.1, 1.0), redshift=None,
                       mag=None, vdisp=None, seed=None, input_meta=None, nocolorcuts=False,
                       nocontinuum=False, agnlike=False, novdisp=False, restframe=False,
                       verbose=False):
        """Build Monte Carlo ELG spectra/templates.

        See the GALAXY.make_galaxy_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        that are specific to the ELG class.

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

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """

        outflux, wave, meta = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=rmagrange,
                                                         oiiihbrange=oiiihbrange, logvdisp_meansig=logvdisp_meansig,
                                                         minlineflux=minoiiflux, redshift=redshift, vdisp=vdisp,
                                                         mag=mag, sne_rfluxratiorange=sne_rfluxratiorange,
                                                         seed=seed, input_meta=input_meta, nocolorcuts=nocolorcuts,
                                                         nocontinuum=nocontinuum, agnlike=agnlike, novdisp=novdisp,
                                                         restframe=restframe, verbose=verbose)

        return outflux, wave, meta

class BGS(GALAXY):
    """Generate Monte Carlo spectra of bright galaxy survey galaxies (BGSs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 add_SNeIa=False, normfilter='decam2014-r', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
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
            from desitarget.cuts import isBGS_bright
            from desitarget.cuts import isBGS_faint
            colorcuts_function = (isBGS_bright, isBGS_faint)

        super(BGS, self).__init__(objtype='BGS', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                  normfilter=normfilter, normline='HBETA', add_SNeIa=add_SNeIa,
                                  baseflux=baseflux, basewave=basewave, basemeta=basemeta)

        self.ewhbetacoeff = [1.28520974, -4.94408026, 4.9617704]

    def make_templates(self, nmodel=100, zrange=(0.01, 0.4), rmagrange=(15.0, 19.5),
                       oiiihbrange=(-1.3, 0.6), logvdisp_meansig=(2.0, 0.17),
                       minhbetaflux=0.0, sne_rfluxratiorange=(0.1, 1.0), redshift=None,
                       mag=None, vdisp=None, seed=None, input_meta=None, nocolorcuts=False,
                       nocontinuum=False, agnlike=False, novdisp=False, restframe=False,
                       verbose=False):
        """Build Monte Carlo BGS spectra/templates.

         See the GALAXY.make_galaxy_templates function for documentation on the
         arguments and inherited attributes.  Here we only document the
         arguments that are specific to the BGS class.

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

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
        """
        outflux, wave, meta = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=rmagrange,
                                                         oiiihbrange=oiiihbrange, logvdisp_meansig=logvdisp_meansig,
                                                         minlineflux=minhbetaflux, redshift=redshift, vdisp=vdisp,
                                                         mag=mag, sne_rfluxratiorange=sne_rfluxratiorange,
                                                         seed=seed, input_meta=input_meta, nocolorcuts=nocolorcuts,
                                                         nocontinuum=nocontinuum, agnlike=agnlike, novdisp=novdisp,
                                                         restframe=restframe, verbose=verbose)
        
        return outflux, wave, meta

class LRG(GALAXY):
    """Generate Monte Carlo spectra of luminous red galaxies (LRGs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 add_SNeIa=False, normfilter='decam2014-z', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
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
            from desitarget.cuts import isLRG_colors as colorcuts_function

        super(LRG, self).__init__(objtype='LRG', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                  normfilter=normfilter, normline=None, add_SNeIa=add_SNeIa,
                                  baseflux=baseflux, basewave=basewave, basemeta=basemeta)

    def make_templates(self, nmodel=100, zrange=(0.5, 1.0), zmagrange=(19.0, 20.4),
                       logvdisp_meansig=(2.3, 0.1), sne_rfluxratiorange=(0.1, 1.0),
                       redshift=None, mag=None, vdisp=None, seed=None,
                       input_meta=None, nocolorcuts=False, novdisp=False, agnlike=False,
                       restframe=False, verbose=False):
        """Build Monte Carlo BGS spectra/templates.

         See the GALAXY.make_galaxy_templates function for documentation on the
         arguments and inherited attributes.  Here we only document the
         arguments that are specific to the LRG class.

        Args:
          zmagrange (float, optional): Minimum and maximum DECam z-band (AB)
            magnitude range.  Defaults to a uniform distribution between (19,
            20.5).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(2.3+/-0.1) km/s
          agnlike (bool, optional): adopt AGN-like emission-line ratios (not yet
            supported; defaults False).

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """

        outflux, wave, meta = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=zmagrange,
                                                         logvdisp_meansig=logvdisp_meansig, redshift=redshift,
                                                         vdisp=vdisp, mag=mag, sne_rfluxratiorange=sne_rfluxratiorange,
                                                         seed=seed, input_meta=input_meta, nocolorcuts=nocolorcuts,
                                                         novdisp=novdisp, agnlike=agnlike, restframe=restframe,
                                                         verbose=verbose)

        # Pack into the metadata table some additional information.
        good = np.where(meta['TEMPLATEID'] != -1)[0]
        if len(good) > 0:
            meta['ZMETAL'][good] = self.basemeta[meta['TEMPLATEID'][good]]['ZMETAL']
            meta['AGE'][good] = self.basemeta[meta['TEMPLATEID'][good]]['AGE']

        return outflux, wave, meta

class SUPERSTAR(object):
    """Base class for generating Monte Carlo spectra of the various flavors of stars."""

    def __init__(self, objtype='STAR', subtype='', minwave=3600.0, maxwave=10000.0, cdelt=0.2,
                 wave=None, colorcuts_function=None, normfilter='decam2014-r',
                 baseflux=None, basewave=None, basemeta=None):
        """Read the appropriate basis continuum templates, filter profiles and
        initialize the output wavelength array.

        Note:
          Only a linearly-spaced output wavelength array is currently supported.

        Args:
          objtype (str): type of object to simulate (default STAR).
          subtype (str, optional): stellar subtype, currently only for white
            dwarfs.  The choices are DA and DB and the default is DA.
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
          normfilt (speclite.filters instance): FilterSequence of self.normfilter.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.

        """
        from speclite import filters

        self.objtype = objtype.upper()
        self.subtype = subtype.upper()
        self.colorcuts_function = colorcuts_function
        self.normfilter = normfilter

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave) / cdelt+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        # Read the rest-frame continuum basis spectra, if not specified.
        if baseflux is None or basewave is None or basemeta is None:
            from desisim.io import read_basis_templates
            baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype,
                                                                subtype=self.subtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the filter profiles.
        self.normfilt = filters.load_filters(self.normfilter)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

    def make_star_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                            magrange=(18.0, 23.5), seed=None, redshift=None,
                            mag=None, input_meta=None, star_properties=None,
                            nocolorcuts=False, restframe=False, verbose=False):

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
        Finally, the user can pass a star_properties table in order to
        interpolate the base templates to non-gridded values of [Fe/H], logg,
        and Teff.

        Note:
          * The default inputs are generally set to values which are appropriate
            for generic stars, so be sure to alter them when generating
            templates for other spectral classes.

          * If both input_meta and star_properties are passed, then
            star_properties is ignored.

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

          star_properties (astropy.Table): *Input* table with the following
            required columns: REDSHIFT, MAG, TEFF, LOGG, and FEH (except for
            WDs, which don't need to have an FEH column).  Optionally, SEED can
            also be included in the table.  When this table is passed, the basis
            templates are interpolated to the desired physical values provided,
            enabling large numbers of mock stellar spectra to be generated with
            physically consistent properties.

          nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).
          restframe (bool, optional): If True, return full resolution restframe
            templates instead of resampled observer frame.
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux
        from desiutil.log import get_logger, DEBUG

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        npix = len(self.basewave)
        nbase = len(self.basemeta)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            nmodel = len(input_meta)
            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype, subtype=self.subtype)

            templateseed = input_meta['SEED'].data
            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data

            nchunk = 1
            alltemplateid_chunk = [input_meta['TEMPLATEID'].data.reshape(nmodel, 1)]

        else:
            if star_properties is not None:
                nmodel = len(star_properties)

                redshift = star_properties['REDSHIFT'].data
                mag = star_properties['MAG'].data

                if 'SEED' in star_properties.keys():
                    templateseed = star_properties['SEED'].data
                else:
                    rand = np.random.RandomState(seed)
                    templateseed = rand.randint(2**32, size=nmodel)

                if 'FEH' in self.basemeta.columns:
                    base_properties  = np.array([self.basemeta['LOGG'], self.basemeta['TEFF'],
                                                 self.basemeta['FEH']]).T.astype('f4')
                    input_properties = (star_properties['LOGG'].data, star_properties['TEFF'].data,
                                        star_properties['FEH'].data)
                else:
                    base_properties  = np.array([self.basemeta['LOGG'], self.basemeta['TEFF']]).T.astype('f4')
                    input_properties = (star_properties['LOGG'].data, star_properties['TEFF'].data)

                nchunk = 1
                alltemplateid_chunk = [np.arange(nmodel).reshape(nmodel, 1)]
            else:
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

            # Initialize the metadata table.
            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype, subtype=self.subtype)

        # Basic error checking and some preliminaries.
        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        # Populate some of the metadata table.
        for key, value in zip(('REDSHIFT', 'MAG', 'SEED'),
                               (redshift, mag, templateseed)):
            meta[key] = value

        # Optionally interpolate onto a non-uniform grid.
        if star_properties is None:
            baseflux = self.baseflux
        else:
            from scipy.interpolate import griddata
            baseflux = griddata(base_properties, self.baseflux,
                                input_properties, method='linear')

        # Build each spectrum in turn.
        if restframe:
            outflux = np.zeros([nmodel, len(self.basewave)]) # [erg/s/cm2/A]
        else:
            outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]
        for ii in range(nmodel):
            zwave = self.basewave.astype(float)*(1.0 + redshift[ii])

            for ichunk in range(nchunk):
                if ii % 100 == 0 and ii > 0:
                    log.debug('Simulating {} template {}/{} in chunk {}/{}.'. \
                              format(self.objtype, ii, nmodel, ichunk+1, nchunk))
                templateid = alltemplateid_chunk[ichunk][ii, :]
                nbasechunk = len(templateid)

                restflux = baseflux[templateid, :]

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                if self.normfilter in self.decamwise.names:
                    normmaggies = np.array(maggies[self.normfilter])
                else:
                    normmaggies = np.array(self.normfilt.get_ab_maggies(
                        restflux, zwave, mask_invalid=True)[self.normfilter])
                magnorm = 10**(-0.4*mag[ii]) / normmaggies

                synthnano = dict()
                for key in maggies.columns:
                    synthnano[key] = 1E9 * maggies[key] * magnorm

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = self.colorcuts_function(
                        gflux=synthnano['decam2014-g'],
                        rflux=synthnano['decam2014-r'],
                        zflux=synthnano['decam2014-z'],
                        w1flux=synthnano['wise2010-W1'],
                        w2flux=synthnano['wise2010-W2'])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask):
                    templaterand = np.random.RandomState(templateseed[ii])

                    this = templaterand.choice(np.where(colormask)[0]) # Pick one randomly.
                    tempid = templateid[this]

                    if restframe:
                        outflux[ii, :] = restflux[this, :] * magnorm[this]
                    else:
                        outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :],
                                                       extrapolate=True) * magnorm[this]

                    meta['TEMPLATEID'][ii] = tempid
                    meta['FLUX_G'][ii] = synthnano['decam2014-g'][this]
                    meta['FLUX_R'][ii] = synthnano['decam2014-r'][this]
                    meta['FLUX_Z'][ii] = synthnano['decam2014-z'][this]
                    meta['FLUX_W1'][ii] = synthnano['wise2010-W1'][this]
                    meta['FLUX_W2'][ii] = synthnano['wise2010-W2'][this]

                    if star_properties is None:
                        meta['TEFF'][ii] = self.basemeta['TEFF'][tempid]
                        meta['LOGG'][ii] = self.basemeta['LOGG'][tempid]
                        if 'FEH' in self.basemeta.columns:
                            meta['FEH'][ii] = self.basemeta['FEH'][tempid]
                    else:
                        meta['TEFF'][ii] = input_properties[1][tempid]
                        meta['LOGG'][ii] = input_properties[0][tempid]
                        if 'FEH' in self.basemeta.columns:
                            meta['FEH'][ii] = input_properties[2][tempid]

                    break

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        if restframe:
            return 1e17 * outflux, self.basewave, meta
        else:
            return 1e17 * outflux, self.wave, meta

class STAR(SUPERSTAR):
    """Generate Monte Carlo spectra of generic stars."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
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
                                   normfilter=normfilter, baseflux=baseflux, basewave=basewave,
                                   basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       rmagrange=(18.0, 23.5), seed=None, redshift=None,
                       mag=None, input_meta=None, star_properties=None,
                       restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for generic stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the STAR class.

        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (18,
            23.5).

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta,
                                                       star_properties=star_properties,
                                                       restframe=restframe, verbose=verbose)
        return outflux, wave, meta

class FSTD(SUPERSTAR):
    """Generate Monte Carlo spectra of (metal-poor, main sequence turnoff) standard
    stars (FSTD).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
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
                                   normfilter=normfilter, baseflux=baseflux, basewave=basewave,
                                   basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0), rmagrange=(16.0, 19.0),
                       seed=None, redshift=None, mag=None, input_meta=None, star_properties=None,
                       nocolorcuts=False, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for FSTD stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the FSTD class.

        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,
            19).

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta,
                                                       star_properties=star_properties,
                                                       nocolorcuts=nocolorcuts, restframe=restframe,
                                                       verbose=verbose)
        return outflux, wave, meta

class MWS_STAR(SUPERSTAR):
    """Generate Monte Carlo spectra of Milky Way Survey (magnitude-limited)
    stars.

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 normfilter='decam2014-r', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
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
                                       normfilter=normfilter, baseflux=baseflux, basewave=basewave,
                                       basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0), rmagrange=(16.0, 20.0),
                       seed=None, redshift=None, mag=None, input_meta=None, star_properties=None,
                       nocolorcuts=False, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for MWS_STAR stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the MWS_STAR class.

        Args:
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,
            20).

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:

        """
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta,
                                                       star_properties=star_properties,
                                                       nocolorcuts=nocolorcuts, restframe=restframe,
                                                       verbose=verbose)
        return outflux, wave, meta

class WD(SUPERSTAR):
    """Generate Monte Carlo spectra of white dwarfs."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 subtype='DA', normfilter='decam2014-g', colorcuts_function=None,
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the WD class.  See the SUPERSTAR.__init__ method for documentation
        on the arguments plus the inherited attributes.

        Note:
          We assume that the WD templates will always be normalized in the
          DECam g-band filter.

        Args:

        Attributes:

        Raises:

        """

        super(WD, self).__init__(objtype='WD', subtype=subtype, minwave=minwave, maxwave=maxwave,
                                 cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                 normfilter=normfilter, baseflux=baseflux, basewave=basewave,
                                 basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0), gmagrange=(16.0, 19.0),
                       seed=None, redshift=None, mag=None, input_meta=None, star_properties=None,
                       nocolorcuts=False, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for WD stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the WD class.

        Args:
          gmagrange (float, optional): Minimum and maximum DECam g-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16,
            19).

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError: If the INPUT_META or STAR_PROPERTIES table contains
            different values of SUBTYPE.

        """
        from desiutil.log import get_logger

        log = get_logger()
        
        for intable in (input_meta, star_properties):
            if intable is not None:
                if 'SUBTYPE' in intable.dtype.names:
                    if (self.subtype != '') and ~np.all(intable['SUBTYPE'] == self.subtype):
                        log.warning('WD Class initialized with subtype {}, which does not match input table.'.format(self.subtype))
                        raise ValueError
        
        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=gmagrange, seed=seed, redshift=redshift,
                                                       mag=mag, input_meta=input_meta,
                                                       star_properties=star_properties,
                                                       nocolorcuts=nocolorcuts,
                                                       restframe=restframe, verbose=verbose)
        return outflux, wave, meta

class QSO():
    """Generate Monte Carlo spectra of quasars (QSOs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
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
          cosmo (astropy.cosmology): Default cosmology object (currently
            hard-coded to FlatLCDM with H0=70, Omega0=0.3).
          normfilt (speclite.filters instance): FilterSequence of self.normfilter.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.

        """
        from astropy.io import fits
        from astropy import cosmology
        from speclite import filters
        from desisim.io import find_basis_template
        from desiutil.log import get_logger
        from desisim import lya_mock_p1d as lyamock

        log = get_logger()

        self.objtype = 'QSO'

        if colorcuts_function is None:
            try:
                from desitarget.cuts import isQSO_colors as colorcuts_function
            except ImportError:
                log.error('Please upgrade desitarget to get the latest isQSO_colors function.')
                from desitarget.cuts import isQSO as colorcuts_function

        self.colorcuts_function = colorcuts_function
        self.normfilter = normfilter

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave) / cdelt+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        self.cosmo = cosmology.core.FlatLambdaCDM(70.0, 0.3)

        self.lambda_lylimit = 911.76
        self.lambda_lyalpha = 1215.67

        # Load the PCA eigenvectors and associated data.
        infile = find_basis_template('qso')
        with fits.open(infile) as hdus:
            hdu_names = [hdus[ii].name for ii in range(len(hdus))]
            self.boss_pca_coeff = hdus[hdu_names.index('BOSS_PCA')].data
            self.sdss_pca_coeff = hdus[hdu_names.index('SDSS_PCA')].data
            self.boss_zQSO = hdus[hdu_names.index('BOSS_Z')].data
            self.sdss_zQSO = hdus[hdu_names.index('SDSS_Z')].data
            self.eigenflux = hdus[hdu_names.index('SDSS_EIGEN')].data
            self.eigenwave = hdus[hdu_names.index('SDSS_EIGEN_WAVE')].data

        self.pca_list = ['PCA0', 'PCA1', 'PCA2', 'PCA3']

        self.z_wind = z_wind

        # Iniatilize the Lyman-alpha mock maker.
        self.lyamock_maker = lyamock.MockMaker()

        # Initialize the filter profiles.
        self.normfilt = filters.load_filters(self.normfilter)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

    def _sample_pcacoeff(self, nsample, coeff, rand):
        """Draw from the distribution of PCA coefficients."""
        cdf = np.cumsum(coeff, dtype=float)
        cdf /= cdf[-1]
        x = rand.uniform(0.0, 1.0, size=nsample)
        
        return coeff[np.interp(x, cdf, np.arange(0, len(coeff), 1)).astype('int')]

    def make_templates(self, nmodel=100, zrange=(0.5, 4.0), rmagrange=(20.0, 22.5),
                       seed=None, redshift=None, mag=None, input_meta=None, N_perz=40, 
                       maxiter=20, uniform=False, lyaforest=True, nocolorcuts=False,
                       verbose=False):
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
          N_perz (int, optional): Number of templates per redshift redshift
            value to generate (default 20).
          maxiter (int): maximum number of iterations for findng a non-negative
            template that also satisfies the color-cuts (default 20).
          uniform (bool, optional): Draw uniformly from the PCA coefficients
            (default False).
          lyaforest (bool, optional): Include Lyman-alpha forest absorption
            (default True).
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False).
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux
        from desiutil.log import get_logger, DEBUG

        if uniform:
            from desiutil.stats import perc

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError
            zrange = (np.min(redshift), np.max(redshift))

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        npix = len(self.eigenwave)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            nmodel = len(input_meta)

            templateseed = input_meta['SEED'].data
            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data

            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype)
            
        else:
            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype)

            # Initialize the random seed.
            rand = np.random.RandomState(seed)
            templateseed = rand.randint(2**32, size=nmodel)

            # Assign redshift and magnitude priors.
            if redshift is None:
                redshift = rand.uniform(zrange[0], zrange[1], nmodel)

            if mag is None:
                mag = rand.uniform(rmagrange[0], rmagrange[1], nmodel).astype('f4')

        # Pre-compute the Lyman-alpha skewers.
        if lyaforest:
            for ii in range(nmodel):
                skewer_wave, skewer_flux1 = self.lyamock_maker.get_lya_skewers(
                    1, new_seed=templateseed[ii])
                if ii == 0:
                    skewer_flux = np.zeros( (nmodel, len(skewer_wave)) )
                skewer_flux[ii, :] = skewer_flux1

        # Populate some of the metadata table.
        meta['TEMPLATEID'] = np.arange(nmodel)
        for key, value in zip(('REDSHIFT', 'MAG', 'SEED'),
                               (redshift, mag, templateseed)):
            meta[key] = value
        if lyaforest:
            meta['SUBTYPE'] = 'LYA'

        # Attenuate below the Lyman-limit by the mean free path (MFP) model
        # measured by Worseck, Prochaska et al. 2014.
        mfp = np.atleast_1d(37.0 * ( (1 + redshift)/5.0)**(-5.4)) # Physical Mpc
        pix912 = np.argmin( np.abs(self.eigenwave-self.lambda_lylimit) )
        zlook = self.cosmo.lookback_distance(redshift)

        # Build each spectrum in turn.
        PCA_rand = np.zeros( (4, N_perz) )
        nonegflux = np.zeros(N_perz)
        flux = np.zeros( (N_perz, npix) )

        zwave = np.outer(self.eigenwave, 1+redshift) # [observed-frame, Angstrom]
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        for ii in range(nmodel):
            if ii % 100 == 0 and ii > 0:
                log.debug('Simulating {} template {}/{}.'.format(self.objtype, ii, nmodel))

            templaterand = np.random.RandomState(templateseed[ii])

            # BOSS or SDSS?
            if redshift[ii] > 2.15:
                zQSO = self.boss_zQSO
                pca_coeff = self.boss_pca_coeff
            else:
                zQSO = self.sdss_zQSO
                pca_coeff = self.sdss_pca_coeff

            # Interpolate the Lya forest spectrum.
            if lyaforest:
                no_forest = ( skewer_wave > self.lambda_lyalpha * (1 + redshift[ii]) )
                skewer_flux[ii, no_forest] = 1.0
                qso_skewer_flux = resample_flux(zwave[:, ii], skewer_wave, skewer_flux[ii, :],
                                                extrapolate=True)
                w = zwave[:, ii] > self.lambda_lyalpha * (1 + redshift[ii])
                qso_skewer_flux[w] = 1.0

            idx = np.where( (zQSO > redshift[ii]-self.z_wind/2) * (zQSO < redshift[ii]+self.z_wind/2) )[0]
            if len(idx) == 0:
                idx = np.where( (zQSO > redshift[ii]-self.z_wind) * (zQSO < redshift[ii]+self.z_wind) )[0]
                if len(idx) == 0:
                    log.warning('Redshift {} far from any parent BOSS/SDSS quasars; choosing closest one.')
                    idx = np.array( np.abs(zQSO-redshift[ii]).argmin() )

            # Need these arrays for the MFP, below.
            if redshift[ii] > 2.39:
                z912 = zwave[:pix912, ii] / self.lambda_lylimit - 1.0
                phys_dist = np.fabs( self.cosmo.lookback_distance(z912) - zlook[ii] ) # [Mpc]
                    
            # Iterate up to maxiter.
            makemore, itercount = True, 0
            while makemore:

                # Gather N_perz sets of coefficients.
                for jj, ipca in enumerate(self.pca_list):
                    if uniform:
                        if jj == 0:  # Use bounds for PCA0 [avoids negative values]
                            xmnx = perc(pca_coeff[ipca][idx], per=95)
                            PCA_rand[jj, :] = templaterand.uniform(xmnx[0], xmnx[1], N_perz)
                        else:
                            mn = np.mean(pca_coeff[ipca][idx])
                            sig = np.std(pca_coeff[ipca][idx])
                            PCA_rand[jj, :] = templaterand.uniform( mn - 2*sig, mn + 2*sig, N_perz)
                    else:
                        PCA_rand[jj, :] = self._sample_pcacoeff(N_perz, pca_coeff[ipca][idx], templaterand)

                # Instantiate the templates, including attenuation below the
                # Lyman-limit based on the MFP, and the Lyman-alpha forest.
                for kk in range(N_perz):
                    flux[kk, :] = np.dot(self.eigenflux.T, PCA_rand[:, kk]).flatten()
                    if redshift[ii] > 2.39:
                         flux[kk, :pix912] *= np.exp(-phys_dist.value / mfp[ii])
                    if lyaforest:
                        flux[kk, :] *= qso_skewer_flux
                    nonegflux[kk] = (np.sum(flux[kk, (zwave[:, ii] > 3000) & (zwave[:, ii] < 1E4)] < 0) == 0) * 1

                # Synthesize photometry to determine which models will pass the
                # color-cuts.  We have to temporarily pad because the spectra
                # don't go red enough.
                padflux, padzwave = self.decamwise.pad_spectrum(flux, zwave[:, ii], method='edge')
                maggies = self.decamwise.get_ab_maggies(padflux, padzwave, mask_invalid=True)

                if self.normfilter in self.decamwise.names:
                    normmaggies = np.array(maggies[self.normfilter])
                else:
                    normmaggies = np.array(self.normfilt.get_ab_maggies(
                        padflux, padzwave, mask_invalid=True)[self.normfilter])
                magnorm = 10**(-0.4*mag[ii]) / normmaggies

                synthnano = dict()
                for key in maggies.columns:
                    synthnano[key] = 1E9 * maggies[key] * magnorm

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, N_perz)
                else:
                    colormask = self.colorcuts_function(
                        gflux=synthnano['decam2014-g'],
                        rflux=synthnano['decam2014-r'],
                        zflux=synthnano['decam2014-z'],
                        w1flux=synthnano['wise2010-W1'],
                        w2flux=synthnano['wise2010-W2'],
                        optical=True)

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask * nonegflux):
                    this = templaterand.choice(np.where(colormask * nonegflux)[0]) # Pick one randomly.
                    outflux[ii, :] = resample_flux(self.wave, zwave[:, ii], flux[this, :],
                                                   extrapolate=True) * magnorm[this]

                    meta['FLUX_G'][ii] = synthnano['decam2014-g'][this]
                    meta['FLUX_R'][ii] = synthnano['decam2014-r'][this]
                    meta['FLUX_Z'][ii] = synthnano['decam2014-z'][this]
                    meta['FLUX_W1'][ii] = synthnano['wise2010-W1'][this]
                    meta['FLUX_W2'][ii] = synthnano['wise2010-W2'][this]

                    makemore = False

                itercount += 1
                if itercount == maxiter:
                    log.warning('Maximum number of iterations reached on QSO {}, z={:.5f}.'.format(ii, redshift[ii]))
                    makemore = False

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        return 1e17 * outflux, self.wave, meta

class SIMQSO():
    """Generate Monte Carlo spectra of quasars (QSOs) using simqso."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 nproc=1, normfilter='decam2014-r', colorcuts_function=None):
        """Read the QSO basis continuum templates, filter profiles and initialize the
           output wavelength array.

        Note:
          Only a linearly-spaced output wavelength array is currently supported
          although an arbitrary wavelength array is possible.

          Much of the infrastructure below is hard-coded to use the SDSS/DR9
          quasar luminosity function (see https://arxiv.org/abs/1210.6389).

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

        Attributes:
          objtype (str): 'QSO'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          procMap (map Class): Built-in map for multiprocessing (based on nproc). 
          cosmo (astropy.cosmology): Default cosmology object (currently
            hard-coded to FlatLCDM with H0=70, Omega0=0.3).
          normfilt (speclite.filters instance): FilterSequence of self.normfilter.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.

        """
        from astropy.io import fits
        from astropy import cosmology
        from speclite import filters
        from desisim.io import find_basis_template

        from desiutil.log import get_logger
        log = get_logger()

        try:
            from simqso.sqbase import ContinuumKCorr, fixed_R_dispersion
            from simqso.sqmodels import BOSS_DR9_PLEpivot
        except ImportError:
            log.fatal('External dependency simqso not found!')

        self.objtype = 'QSO'

        if colorcuts_function is None:
            try:
                from desitarget.cuts import isQSO_colors as colorcuts_function
            except ImportError:
                log.error('Please upgrade desitarget to get the latest isQSO_colors function.')
                from desitarget.cuts import isQSO as colorcuts_function

        self.colorcuts_function = colorcuts_function
        self.normfilter = normfilter

        # Initialize multiprocessing map object.
        if nproc > 1:
            pool = multiprocessing.Pool(nproc)
            self.procMap = pool.map
        else:
            self.procMap = map
        
        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = (maxwave-minwave) / cdelt+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        self.basewave = fixed_R_dispersion(900.0, 6e4, 8000)
        self.cosmo = cosmology.core.FlatLambdaCDM(70.0, 0.3)

        self.lambda_lylimit = 911.76
        self.lambda_lyalpha = 1215.67

        # Initialize the filter profiles.
        self.normfilt = filters.load_filters(self.normfilter)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')

        # Initialize the BOSS/DR9 quasar luminosity function and k-correction
        # objects.
        def _filtname(filt):
            if filt not in ('decam2014-g', 'decam2014-r', 'decam2014-z'):
                log.warning('Unrecognized normalization filter {}! Using {}'.format('decam2014-r'))
                filt = self.normfilter
            outfilt = 'DECam-{}'.format(filt[-1])
            return outfilt
            
        # Initialize the K-correction and luminosity function objects.
        self.kcorr = ContinuumKCorr(_filtname(self.normfilter), 1450, effWaveBand='SDSS-r')
        self.qlf = BOSS_DR9_PLEpivot(cosmo=self.cosmo)

    def empty_qsometa(self, meta):
        """Initialize the QSO metadata table."""

        from astropy.table import Table, Column

        nmodel = len(meta)

        qsometa = Table()
        qsometa.add_column(meta['TEMPLATEID'].copy())
        qsometa.add_column(Column(name='ABSMAG', length=nmodel, dtype='f4'))
        qsometa.add_column(Column(name='SLOPES', length=nmodel, dtype='f4',
                                  shape=(5,)))
        qsometa.add_column(Column(name='EMLINES', length=nmodel, dtype='f4',
                                  shape=(62,3)))
                                  
        return qsometa

    def make_templates(self, nmodel=100, zrange=(0.5, 4.0), rmagrange=(19.0, 23.0),
                       seed=None, redshift=None, input_meta=None, maxiter=20,
                       lyaforest=True, nocolorcuts=False, return_qsometa=False,
                       verbose=False):
        """Build Monte Carlo QSO spectra/templates.

        * This function generates QSO spectra on-the-fly using @imcgreer's
          simqso.  The default is to generate flat, uncorrelated priors on
          redshift, absolute magnitudes based on the SDSS/DR9 QSOLF, and to
          compute the corresponding apparent magnitudes using the appropriate
          per-object K-correction.

          Alternatively, the redshift can be input and the absolute and apparent
          magnitudes will again be computed self-consistently from the QSOLF.

          Providing apparent magnitudes on *input* is not supported although it
          could be if there is need.  However, one can control the apparent
          brightness of the resulting QSO spectra by specifying rmagrange.

        * The way the code is currently structured could lead to memory problems
          if one attempts to generate very large numbers of spectra
          simultaneously (>10^4, perhaps, depending on the machine).  However,
          it can easily be refactored to generate the appropriate number of
          templates in chunks at the expense of some computational speed.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5, 4.0).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (19,
            23).
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
          maxiter (int): maximum number of iterations for findng a template that
            satisfies the color-cuts (default 20).
          lyaforest (bool, optional): Include Lyman-alpha forest absorption
            (default True).
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False).
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux
        from desiutil.log import get_logger, DEBUG

        from simqso.sqgrids import generateQlfPoints, ConstSampler, DustBlackbodyVar
        from simqso.sqmodels import get_BossDr9_model_vars
        from simqso.sqrun import buildSpectraBulk

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError
            zrange = (np.min(redshift), np.max(redshift))

        # Optionally unpack a metadata table.
        if input_meta is not None:
            log.warning('Input metadata table not yet supported!')
            raise ValueError
        
        else:
            meta = empty_metatable(nmodel=nmodel, objtype=self.objtype)

            # Initialize the random seed.
            rand = np.random.RandomState(seed)
            templateseed = rand.randint(2**32, size=nmodel)

            # Assign redshift and magnitude priors.
            if redshift is None:
                redshift = rand.uniform(zrange[0], zrange[1], nmodel)
            
        # Populate some of the metadata table.
        meta['TEMPLATEID'] = np.arange(nmodel)
        meta['REDSHIFT'] = redshift
        
        # Initialize the QSO metadata table and output flux array.
        qsometa = self.empty_qsometa(meta)

        outflux = np.zeros([nmodel, len(self.wave)])

        # Generate the parameters of the spectra and the spectra themselves,
        # iterating (up to maxiter) until enough models have passed the color
        # cuts.
        need = np.where( np.sum(outflux, axis=1) == 0 )[0]

        itercount = 0
        iterseed = rand.randint(2**32, size=maxiter)

        while (len(need) > 0):

            # Sample from the QLF, using the input redshifts.
            qsos = generateQlfPoints(self.qlf, rmagrange, zrange,
                                     kcorr=self.kcorr, zin=redshift[need],
                                     qlfseed=iterseed[itercount],
                                     gridseed=iterseed[itercount])

            # Add the fiducial quasar SED model from BOSS/DR9, optionally
            # without IGM absorption. This step adds a fiducial continuum,
            # emission-line template, and an iron emission-line template.
            qsos.addVars(get_BossDr9_model_vars(qsos, self.basewave, noforest=not lyaforest))

            # Establish the desired (output) photometric system.
            qsos.loadPhotoMap([('DECam', 'DECaLS'), ('WISE', 'AllWISE')])

            # Finally, generate the spectra, iterating in order to converge on the
            # per-object K-correction (typically, after ~two steps the maximum error
            # on the absolute mags is typically <<1%).
            _, flux = buildSpectraBulk(self.basewave, qsos, maxIter=5,
                                       procMap=self.procMap, saveSpectra=True,
                                       verbose=False)

            # Synthesize photometry to determine which models will pass the
            # color cuts.
            maggies = self.decamwise.get_ab_maggies(flux, self.basewave.copy(), mask_invalid=True)

            if self.normfilter in self.decamwise.names:
                normmaggies = np.array(maggies[self.normfilter])
            else:
                normmaggies = np.array(self.normfilt.get_ab_maggies(
                    flux, self.basewave.copy(), mask_invalid=True)[self.normfilter])

            synthnano = dict()
            for key in maggies.columns:
                synthnano[key] = 1E9 * maggies[key]

            if nocolorcuts or self.colorcuts_function is None:
                colormask = np.repeat(1, len(need))
            else:
                colormask = self.colorcuts_function(
                    gflux=synthnano['decam2014-g'],
                    rflux=synthnano['decam2014-r'],
                    zflux=synthnano['decam2014-z'],
                    w1flux=synthnano['wise2010-W1'],
                    w2flux=synthnano['wise2010-W2'])

            # For objects that pass the color cuts, populate the output flux
            # vector, the metadata and qsometadata tables, and finish up.
            these = np.where(colormask)[0]
            if len(these) > 0:
                for ii in range(len(these)):
                    outflux[need[these[ii]], :] = resample_flux(
                        self.wave, self.basewave, flux[these[ii], :],
                        extrapolate=True)

                meta['SEED'][need] = iterseed[itercount]
                meta['MAG'][need[these]] = -2.5 * np.log10(normmaggies[these])
                meta['FLUX_G'][need[these]] = synthnano['decam2014-g'][these]
                meta['FLUX_R'][need[these]] = synthnano['decam2014-r'][these]
                meta['FLUX_Z'][need[these]] = synthnano['decam2014-z'][these]
                meta['FLUX_W1'][need[these]] = synthnano['wise2010-W1'][these]
                meta['FLUX_W2'][need[these]] = synthnano['wise2010-W2'][these]

                qsometa['ABSMAG'][need[these]] = qsos.data['absMag'][these].data
                qsometa['SLOPES'][need[these], :] = qsos.data['slopes'][these, :].data
                qsometa['EMLINES'][need[these], :, :] = qsos.data['emLines'][these, :, :].data

            itercount += 1
            need = np.where( np.sum(outflux, axis=1) == 0 )[0]

            if itercount == maxiter:
                log.warning('Maximum number of iterations reached.')
                break

        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        if return_qsometa:
            return 1e17 * outflux, self.wave, meta, qsometa
        else:
            return 1e17 * outflux, self.wave, meta

def specify_galparams_dict(templatetype, zrange=None, magrange=None,
                            oiiihbrange=None, logvdisp_meansig=None,
                            minlineflux=None, sne_rfluxratiorange=None,
                            redshift=None, mag=None, vdisp=None,
                            nocolorcuts=None, nocontinuum=None,
                            agnlike=None, novdisp=None, restframe=None):
    
    '''
    Creates a dictionary of keyword variables to be passed to GALAXY.make_templates (or one
    of GALAXY's child classes). Allows the user to fully define the templated spectra, via
    defining individual targets or ranges in values. Values already specified in get_targets
    are not included here. Anything not define or set to None will not be assigned and 
    CLASS.make_templates will assume the following as defaults:
        
        * nmodel=100, zrange=(0.6, 1.6), magrange=(21.0, 23.5),
        * oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
        * minlineflux=0.0, sne_rfluxratiorange=(0.01, 0.1),
        * seed=None, redshift=None, mag=None, vdisp=None,
        * input_meta=None, nocolorcuts=False, nocontinuum=False,
        * agnlike=False, novdisp=False, restframe=False, verbose=False

    Args:
        
        * nmodel (int, optional): Number of models to generate (default 100).
        * zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.6, 1.6).
        * magrange (float, optional): Minimum and maximum magnitude in the
            bandpass specified by self.normfilter.  Defaults to a uniform
            distribution between (21, 23.4) in the r-band.
        * oiiihbrange (float, optional): Minimum and maximum logarithmic
            [OIII] 5007/H-beta line-ratio.  Defaults to a uniform distribution
            between (-0.5, 0.2).
        * logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=1.9+/-0.15 km/s.
        * minlineflux (float, optional): Minimum emission-line flux in the line
            specified by self.normline (default 0 erg/s/cm2).
        * sne_rfluxratiorange (float, optional): r-band flux ratio of the SNeIa
            spectrum with respect to the underlying galaxy.  Defaults to a
            uniform distribution between (0.01, 0.1).
        * seed (int, optional): Input seed for the random numbers.
        * redshift (float, optional): Input/output template redshifts.  Array
            size must equal nmodel.  Ignores zrange input.
        * mag (float, optional): Input/output template magnitudes in the band
            specified by self.normfilter.  Array size must equal nmodel.
            Ignores magrange input.
        * vdisp (float, optional): Input/output velocity dispersions.  Array
            size must equal nmodel.  Ignores magrange input.
        * input_meta (astropy.Table): *Input* metadata table with the following
            required columns: TEMPLATEID, SEED, REDSHIFT, VDISP, MAG (where mag
            is specified by self.normfilter).  In addition, if add_SNeIa is True
            then the table must also contain SNE_TEMPLATEID, SNE_EPOCH, and
            SNE_RFLUXRATIO columns.  See desisim.io.empty_metatable for the
            required data type for each column.  If this table is passed then
            all other optional inputs (nmodel, redshift, vdisp, mag, zrange,
            logvdisp_meansig, etc.) are ignored.
        * nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).
        * nocontinuum (bool, optional): Do not include the stellar continuum in
            the output spectrum (useful for testing; default False).  Note that
            this option automatically sets nocolorcuts to True and add_SNeIa to
            False.
        * novdisp (bool, optional): Do not velocity-blur the spectrum (default False).
        * agnlike (bool, optional): Adopt AGN-like emission-line ratios (e.g.,
            for the LRGs and some BGS galaxies) (default False, meaning we adopt
            star-formation-like line-ratios).  Option not yet supported.
        * restframe (bool, optional): If True, return full resolution restframe
            templates instead of resampled observer frame.
        * verbose (bool, optional): Be verbose!

    Returns:
        
      * fulldef_dict (dict): dictionary containing all of the values passed defined
                   with variable names as the corresonding key. These are intentionally
                   identical to those passed to the make_templates classes above
                   
    '''
    
    fulldef_dict = {}
    if zrange is not None:
        fulldef_dict['zrange'] = zrange
    if magrange is not None:
        if templatetype == 'LRG':
            fulldef_dict['zmagrange'] = magrange
        else:
            fulldef_dict['rmagrange'] = magrange
    if oiiihbrange is not None:
        fulldef_dict['oiiihbrange'] = oiiihbrange
    if logvdisp_meansig is not None:
        fulldef_dict['logvdisp_meansig'] = logvdisp_meansig
    if minlineflux is not None:
        fulldef_dict['minlineflux'] = minlineflux
    if sne_rfluxratiorange is not None:
        fulldef_dict['sne_rfluxratiorange'] = sne_rfluxratiorange
    if redshift is not None:
        fulldef_dict['redshift'] = redshift
    if mag is not None:
        fulldef_dict['mag'] = mag
    if vdisp is not None:
        fulldef_dict['vdisp'] = vdisp
    if nocolorcuts is not None:
        fulldef_dict['nocolorcuts'] = nocolorcuts
    if nocontinuum is not None:
        fulldef_dict['nocontinuum'] = nocontinuum
    if agnlike is not None:
        fulldef_dict['agnlike'] = agnlike
    if novdisp is not None:
        fulldef_dict['novdisp'] = novdisp
    if restframe is not None:
        fulldef_dict['restframe'] = restframe    
    return fulldef_dict

def specify_starparams_dict(templatetype,vrad_meansig=None,
                            magrange=None, redshift=None,
                            mag=None, input_meta=None, star_properties=None,
                            nocolorcuts=None, restframe=None):
    '''
    Creates a dictionary of keyword variables to be passed to SUPERSTAR.make_templates (or one
    of SUPERSTAR's child classes). Allows the user to fully define the templated spectra, via
    defining individual targets or ranges in values. Values already specified in get_targets
    are not included here. Anything not define or set to None will not be assigned and 
    CLASS.make_templates will assume the following as defaults:    
    
        * nmodel=100, vrad_meansig=(0.0, 200.0),
        * magrange=(18.0, 23.5), seed=None, redshift=None,
        * mag=None, input_meta=None, star_properties=None,
        * nocolorcuts=False, restframe=False, verbose=False 
        
    Args:
        
        * nmodel (int, optional): Number of models to generate (default 100).
        * vrad_meansig (float, optional): Mean and sigma (standard deviation) of the
                    radial velocity "jitter" (in km/s) that should be included in each
                    spectrum.  Defaults to a normal distribution with a mean of zero and
                    sigma of 200 km/s.
        * magrange (float, optional): Minimum and maximum magnitude in the
                    bandpass specified by self.normfilter.  Defaults to a uniform
                    distribution between (18, 23.5) in the r-band.
        * seed (int, optional): input seed for the random numbers.
        * redshift (float, optional): Input/output (dimensionless) radial
                    velocity.  Array size must equal nmodel.  Ignores vrad_meansig
                    input.
        * mag (float, optional): Input/output template magnitudes in the band
                   specified by self.normfilter.  Array size must equal nmodel.
                   Ignores magrange input.
        * input_meta (astropy.Table): *Input* metadata table with the following
                   required columns: TEMPLATEID, SEED, REDSHIFT, and MAG 
                   (where mag is specified by self.normfilter).  
                   See desisim.io.empty_metatable for the required data type 
                   for each column.  If this table is passed then all other
                   optional inputs (nmodel, redshift, mag, vrad_meansig                                                                                                   etc.) are ignored.
        * star_properties (astropy.Table): *Input* table with the following
                   required columns: REDSHIFT, MAG, TEFF, LOGG, and FEH (except for
                   WDs, which don't need to have an FEH column).  Optionally, SEED can
                   also be included in the table.  When this table is passed, the basis
                   templates are interpolated to the desired physical values provided,
                   enabling large numbers of mock stellar spectra to be generated with
                   physically consistent properties.
        * nocolorcuts (bool, optional): Do not apply the color-cuts specified by
                   the self.colorcuts_function function (default False).
        * restframe (bool, optional): If True, return full resolution restframe
                   templates instead of resampled observer frame.
        * verbose (bool, optional): Be verbose!
      
    Returns:
        
        * fulldef_dict (dict): dictionary containing all of the values passed defined
                   with variable names as the corresonding key. These are intentionally
                   identical to those passed to the make_templates classes above
                   
    '''
    
    fulldef_dict = {}
    if vrad_meansig is not None:
        fulldef_dict['vrad_meansig'] = vrad_meansig
    if magrange is not None:
        if templatetype=='WD':
            fulldef_dict['gmagrange'] = magrange
        else:
            fulldef_dict['rmagrange'] = magrange
    if redshift is not None:
        fulldef_dict['redshift'] = redshift
    if mag is not None:
        fulldef_dict['mag'] = mag
    if input_meta is not None:
        fulldef_dict['input_meta'] = input_meta
    if star_properties is not None:
        fulldef_dict['star_properties'] = star_properties
    if nocolorcuts is not None:
        fulldef_dict['nocolorcuts'] = nocolorcuts
    if restframe is not None:
        fulldef_dict['restframe'] = restframe
    return fulldef_dict
