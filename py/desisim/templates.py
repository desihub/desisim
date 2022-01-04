"""
desisim.templates
=================

Functions to simulate spectral templates for DESI.
"""

from __future__ import division, print_function

import os
import sys
import numpy as np
import multiprocessing
from desiutil.log import get_logger, DEBUG
from desisim.io import empty_metatable

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

def _check_input_meta(input_meta, ignore_templateid=False):
    log = get_logger()
    cols = input_meta.colnames
    if ignore_templateid:
        required_cols = ('SEED', 'REDSHIFT', 'MAG', 'MAGFILTER')
    else:
        required_cols = ('TEMPLATEID', 'SEED', 'REDSHIFT', 'MAG', 'MAGFILTER')
    if not np.all(np.in1d(required_cols, cols)):
        log.warning('Input metadata table (input_meta) is missing one or more required columns {}'.format(
            required_cols))
        raise ValueError

def _check_star_properties(star_properties, WD=False):
    log = get_logger()
    cols = star_properties.colnames
    required_cols = ['REDSHIFT', 'MAG', 'MAGFILTER', 'TEFF', 'LOGG']
    if not WD:
        required_cols = required_cols + ['FEH']
    if not np.all(np.in1d(required_cols, cols)):
        log.warning('Input star_properties is missing one or more required columns {}'.format(
            required_cols))
        raise ValueError

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
        include_mgii (bool, optional): Include Mg II in emission (default False). 

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
    def __init__(self, minwave=3650.0, maxwave=7075.0, cdelt_kms=20.0, log10wave=None,
                 include_mgii=False):

        from pkg_resources import resource_filename
        from astropy.table import Table, Column, vstack
        from desiutil.sklearn import GaussianMixtureModel

        log = get_logger()
        
        # Build a wavelength array if one is not given.
        if log10wave is None:
            cdelt = cdelt_kms/C_LIGHT/np.log(10) # pixel size [log-10 A]
            npix = int(round((np.log10(maxwave)-np.log10(minwave))/cdelt))+1
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
        self.include_mgii = include_mgii
        if not self.include_mgii:
            forbiddata.remove_rows(np.where(forbiddata['name'] == 'MgII_2800a')[0])
            forbiddata.remove_rows(np.where(forbiddata['name'] == 'MgII_2800b')[0])
            
        line = vstack([recombdata,forbiddata], metadata_conflicts='silent')

        nline = len(line)
        line['flux'] = Column(np.ones(nline), dtype='f8')  # integrated line-flux
        line['amp'] = Column(np.ones(nline), dtype='f8')   # amplitude
        self.line = line[np.argsort(line['wave'])]

        self.forbidmog = GaussianMixtureModel.load(forbidmogfile)

        self.oiiidoublet = 2.8875   # [OIII] 5007/4959
        self.niidoublet = 2.93579   # [NII] 6584/6548
        self.oidoublet = 3.03502    # [OI] 6300/6363
        self.siiidoublet = 2.4686   # [SIII] 9532/9069
        self.ariiidoublet = 4.16988 # [ArIII] 7135/7751
        self.mgiidoublet = 1.0      # MgII 2803/2796

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
        from astropy.table import Table

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

        is6300 = np.where(line['name'] == '[OI]_6300')[0]
        is6363 = np.where(line['name'] == '[OI]_6363')[0]
        is9532 = np.where(line['name'] == '[SIII]_9532')[0]
        is9069 = np.where(line['name'] == '[SIII]_9069')[0]
        is7135 = np.where(line['name'] == '[ArIII]_7135')[0]
        is7751 = np.where(line['name'] == '[ArIII]_7751')[0]

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

        # Hack! For the following lines use constant ratios relative to H-beta--

        # Normalize [OI]
        line['ratio'][is6300] = 0.1 # [OI]6300/Hbeta
        line['ratio'][is6363] = line['ratio'][is6300]/self.oidoublet 

        # Normalize [SIII]
        line['ratio'][is9532] = 0.75 # [SIII]9532/Hbeta
        line['ratio'][is9069] = line['ratio'][is9532]/self.siiidoublet
        
        # Normalize [ArIII]
        line['ratio'][is7135] = 0.04 # [ArIII]7135/Hbeta
        line['ratio'][is7751] = line['ratio'][is7135]/self.ariiidoublet

        # Normalize MgII
        if self.include_mgii:
            is2800a = np.where(line['name'] == 'MgII_2800a')[0]
            is2800b = np.where(line['name'] == 'MgII_2800b')[0]

            line['ratio'][is2800a] = 0.3 # MgII2796/Hbeta
            line['ratio'][is2800a] = line['ratio'][is2800a]/self.mgiidoublet
        
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
        log10sigma = linesigma /C_LIGHT / np.log(10) # line-width [log-10 Angstrom]
        emspec = np.zeros_like(self.log10wave)

        loglinewave = np.log10(line['wave'])
        these = np.where( (loglinewave > self.log10wave.min()) *
                          (loglinewave < self.log10wave.max()) )[0]
        if len(these) > 0:
            theseline = line[these]
            for ii in range(len(theseline)):
                amp = theseline['flux'][ii] / theseline['wave'][ii] / np.log(10) # line-amplitude [erg/s/cm2/A]
                thislinewave = np.log10(theseline['wave'][ii] * (1.0 + zshift))
                theseline['amp'][ii] = amp / (np.sqrt(2.0 * np.pi) * log10sigma)  # [erg/s/A]

                # Construct the spectrum [erg/s/cm2/A, rest]
                jj = np.abs( self.log10wave - thislinewave ) < 6 * log10sigma
                emspec[jj] += amp * np.exp(-0.5 * (self.log10wave[jj]-thislinewave)**2 / log10sigma**2) \
                              / (np.sqrt(2.0 * np.pi) * log10sigma)
        else:
            theseline = Table()

        return emspec, 10**self.log10wave, theseline

class GALAXY(object):
    """Base class for generating Monte Carlo spectra of the various flavors of
       galaxies (ELG, BGS, and LRG).

    """
    def __init__(self, objtype='ELG', minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 transient=None, tr_fluxratio=(0.01, 1.), tr_epoch=(-10,10),
                 include_mgii=False, colorcuts_function=None,
                 normfilter_north='BASS-r', normfilter_south='decam2014-r',
                 normline='OII', fracvdisp=(0.1, 40), 
                 baseflux=None, basewave=None, basemeta=None):
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
          normfilter_north (str): normalization filter for simulated "north"
            templates.  Each spectrum is normalized to the magnitude in this
            filter bandpass (default 'BASS-r').
          normfilter_south (str): corresponding normalization filter for "south"
            (default 'decam2014-r').
          normline (str): normalize the emission-line spectrum to the flux in
            this emission line.  The options are 'OII' (for ELGs, the default),
            'HBETA' (for BGS), or None (for LRGs).
          fracvdisp (tuple): two-element array which gives the fraction and
            absolute number of unique velocity dispersion values.  For example,
            the default (0.1, 40) means there will be either int(0.1*nmodel) or
            40 unique values, where nmodel is defined in
            GALAXY.make_galaxy_templates, below.
          transient (Transient, None): optional Transient object to integrate
            into the spectrum (default None).
          tr_fluxratio (tuple): optional flux ratio range for transient
            and host spectrum. Default is (0.01, 1).
          tr_epoch (tuple): optional epoch range for uniformly sampling a
            transient spectrum, in days. Default is (-10, 10).
          include_mgii (bool, optional): Include Mg II in emission (default False).  

        Attributes:
          wave (numpy.ndarray): Output wavelength array (Angstrom).
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            continuum spectra (erg/s/cm2/A).
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX (Angstrom).
          basemeta (astropy.Table): Table of meta-data [nbase] for each base template.
          pixbound (numpy.ndarray): Pixel boundaries of BASEWAVE (Angstrom).
          normfilt_north (speclite.filters instance): FilterSequence of
            self.normfilter_north.
          normfilt_south (speclite.filters instance): FilterSequence of
            self.normfilter_south.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.
          bassmzlswise (speclite.filters instance): BASS-[g,r], MzLS-z and
            WISE2010-[W1,W2] FilterSequence.

        Optional Attributes:
          sne_baseflux (numpy.ndarray): Array [sne_nbase,sne_npix] of the base
            rest-frame SNeIa spectra interpolated onto BASEWAVE [erg/s/cm2/A].
          sne_basemeta (astropy.Table): Table of meta-data for each base SNeIa
            spectra [sne_nbase].
          rfilt_north (speclite.filters instance): BASS r-band FilterSequence.
          rfilt_south (speclite.filters instance): DECam2014 r-band FilterSequence.

        """
        from speclite import filters
        from desisim import pixelsplines as pxs

        self.objtype = objtype.upper()
        self.colorcuts_function = colorcuts_function
        self.normfilter_north = normfilter_north
        self.normfilter_south = normfilter_south
        self.normline = normline

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = int(round((maxwave-minwave) / cdelt))+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        # Read the rest-frame continuum basis spectra, if not specified.
        if baseflux is None or basewave is None or basemeta is None:
            from desisim.io import read_basis_templates
            baseflux, basewave, basemeta = read_basis_templates(objtype=self.objtype)
        self.baseflux = baseflux
        self.basewave = basewave
        self.basemeta = basemeta

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample.
        if self.normline is not None:
            if self.normline.upper() not in ('OII', 'HBETA'):
                log.warning('Unrecognized normline input {}; setting to None.'.format(self.normline))
                self.normline = None

            self.EM = EMSpectrum(log10wave=np.log10(self.basewave), include_mgii=include_mgii)

        # Optionally access a transient model.
        self.transient = transient
        self.trans_fluxratiorange = tr_fluxratio
        self.trans_epochrange = tr_epoch

        if self.transient is not None:
            self.rfilt_north = filters.load_filters('BASS-r')
            self.rfilt_south = filters.load_filters('decam2014-r')

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)
        self.fracvdisp = fracvdisp

        # Initialize the filter profiles.
        self.normfilt_north = filters.load_filters(self.normfilter_north)
        self.normfilt_south = filters.load_filters(self.normfilter_south)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')
        self.bassmzlswise = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z',
                                                 'wise2010-W1', 'wise2010-W2')

        # Default fiber fractions based on https://github.com/desihub/desisim/pull/550
        self.fiberflux_fraction = {'ELG': 0.6, 'LRG': 0.4, 'BGS': 0.3}

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
            sigma = 1.0 + (self.basewave * uvv / C_LIGHT)
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
            raise NotImplementedError('AGNLIKE option not yet implemented')

        if rand is None:
            rand = np.random.RandomState()

        if oiidoublet_meansig[1] > 0:
            oiidoublet = rand.normal(oiidoublet_meansig[0], oiidoublet_meansig[1], nobj)
        else:
            oiidoublet = np.repeat(oiidoublet_meansig[0], nobj)

        # Sample from the MoG.  This is not strictly correct because it ignores
        # the prior on [OIII]/Hbeta, but let's revisit that later.
        samp = self.EM.forbidmog.sample(nobj, random_state=rand)
        oiiihbeta = samp[:, 0]
        oiihbeta = samp[:, 1]
        niihbeta = samp[:, 2]
        siihbeta = samp[:, 3]

        return oiidoublet, oiihbeta, niihbeta, siihbeta, oiiihbeta

    def make_galaxy_templates(self, nmodel=100, zrange=(0.6, 1.6), magrange=(20.0, 22.0),
                              oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
                              minlineflux=0.0, trans_filter='decam2014-r',
                              seed=None, redshift=None, mag=None, vdisp=None,
                              input_meta=None, nocolorcuts=False,
                              nocontinuum=False, agnlike=False, novdisp=False, south=True,
                              restframe=False, verbose=False):
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
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False). Defaults to a uniform
            distribution between (20.0, 22.0).
          oiiihbrange (float, optional): Minimum and maximum logarithmic
            [OIII] 5007/H-beta line-ratio.  Defaults to a uniform distribution
            between (-0.5, 0.2).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=1.9+/-0.15 km/s.
          minlineflux (float, optional): Minimum emission-line flux in the line
            specified by self.normline (default 0 erg/s/cm2).
        
          trans_filter (str): filter corresponding to TRANS_FLUXRATIORANGE (default
            'decam2014-r').
        
          seed (int, optional): Input seed for the random numbers.
          redshift (float, optional): Input/output template redshifts.  Array
            size must equal nmodel.  Ignores zrange input.
          mag (float, optional): Input/output template magnitudes in the
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False).  Array size must equal
            nmodel.  Ignores magrange input.
          vdisp (float, optional): Input/output velocity dispersions in km/s.
            Array size must equal nmodel.
        
          input_meta (astropy.Table): *Input* metadata table with the following
            required columns: TEMPLATEID, SEED, REDSHIFT, MAG, and MAGFILTER
            (see desisim.io.empty_metatable for the expected data types).  In
            addition, in order to faithfully reproduce a previous set of
            spectra, then VDISP must also be passed (normally returned in the
            OBJMETA table).  If present, then all other optional inputs (nmodel,
            redshift, mag, zrange, logvdisp_meansig, etc.) are ignored.

          nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).
          nocontinuum (bool, optional): Do not include the stellar continuum in
            the output spectrum (useful for testing; default False).  Note that
            this option automatically sets nocolorcuts to True and transient to
            False.
          novdisp (bool, optional): Do not velocity-blur the spectrum (default
            False).
          agnlike (bool, optional): Adopt AGN-like emission-line ratios (e.g.,
            for the LRGs and some BGS galaxies) (default False, meaning we adopt
            star-formation-like line-ratios).  Option not yet supported.
          south (bool, optional): Apply "south" color-cuts using the DECaLS
            filter system, otherwise apply the "north" (MzLS+BASS) color-cuts.
            Defaults to True.
          restframe (bool, optional): If True, return full resolution restframe
            templates instead of resampled observer frame.
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:
          ValueError

        """
        from speclite import filters
        from desispec.interpolation import resample_flux
        from astropy.table import Column
        from astropy import units as u

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        # Basic error checking and some preliminaries.
        if nocontinuum:
            log.warning('Forcing nocolorcuts=True, transient=None since nocontinuum=True.')
            nocolorcuts = True
            self.transient = None

        npix = len(self.basewave)
        nbase = len(self.basemeta)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            _check_input_meta(input_meta)

            templateseed = input_meta['SEED'].data
            rand = np.random.RandomState(templateseed[0])

            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data
            magfilter = np.char.strip(input_meta['MAGFILTER'].data)
            
            nchunk = 1
            nmodel = len(input_meta)
            alltemplateid_chunk = [input_meta['TEMPLATEID'].data.reshape(nmodel, 1)]

            meta, objmeta = empty_metatable(nmodel=nmodel, objtype=self.objtype)
        else:
            meta, objmeta = empty_metatable(nmodel=nmodel, objtype=self.objtype)

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

            if south:
                magfilter = np.repeat(self.normfilter_south, nmodel)
            else:
                magfilter = np.repeat(self.normfilter_north, nmodel)

        if vdisp is None:
            # Limit the number of unique velocity dispersion values.
            nvdisp = int(np.max( ( np.min(
                ( np.round(nmodel * self.fracvdisp[0]), self.fracvdisp[1] ) ), 1 ) ))
            if logvdisp_meansig[1] > 0:
                vvdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nvdisp)
            else:
                vvdisp = 10**np.repeat(logvdisp_meansig[0], nvdisp)
            vdisp = rand.choice(vvdisp, nmodel)

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

            vzero = np.where(vdisp <= 0)[0]
            if len(vzero) > 0:
                log.fatal('Velocity dispersion is zero or negative!')
                raise ValueError

        # Generate the (optional) distribution of transient model brightness
        # and epoch priors or read them from the input table.
        if self.transient is not None:
            trans_rfluxratio = rand.uniform(self.trans_fluxratiorange[0], self.trans_fluxratiorange[1], nmodel)
            log.debug('Flux ratio range: {:g} to {:g}'.format(self.trans_fluxratiorange[0], self.trans_fluxratiorange[1]))
            log.debug('Generated ratios: {}'.format(trans_rfluxratio))

            tmin = self.trans_epochrange[0]
            if tmin < self.transient.mintime().to('day').value:
                tmin = self.transient.mintime().to('day').value
            tmin = int(tmin)

            tmax = self.trans_epochrange[1]
            if tmax > self.transient.maxtime().to('day').value:
                tmax = self.transient.maxtime().to('day').value
            tmax = int(tmax)

            trans_epoch = rand.randint(tmin, tmax, nmodel)
            log.debug('Epoch range: {:d} d to {:d} d'.format(tmin, tmax))
            log.debug('Generated epochs: {}'.format(trans_epoch))

            # Populate the object metadata table.
            objmeta['TRANSIENT_MODEL'][:] = np.full(nmodel, self.transient.model)
            objmeta['TRANSIENT_TYPE'][:] = np.full(nmodel, self.transient.type)
            objmeta['TRANSIENT_EPOCH'][:] = trans_epoch
            objmeta['TRANSIENT_RFLUXRATIO'][:] = trans_rfluxratio

        # Precompute the velocity dispersion convolution matrix for each unique
        # value of vdisp.
        if nocontinuum or novdisp:
            pass
        else:
            blurmatrix = self._blurmatrix(vdisp, log=log)

        # Populate some of the metadata table.
        objmeta['VDISP'][:] = vdisp
        for key, value in zip(('REDSHIFT', 'MAG', 'MAGFILTER', 'SEED'),
                               (redshift, mag, magfilter, templateseed)):
            meta[key][:] = value

        # Load the unique set of MAGFILTERs.  We could check against
        # self.decamwise.names and self.bassmzlswise to see if the filters have
        # already been loaded, but speed should not be an issue.
        normfilt = dict()
        for mfilter in np.unique(magfilter):
            normfilt[mfilter] = filters.load_filters(mfilter)

        # Optionally initialize the emission-line objects and line-ratios.
        d4000 = self.basemeta['D4000'].data

        # Build each spectrum in turn.
        if restframe:
            outflux = np.zeros([nmodel, len(self.basewave)])
        else:
            outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        fiberflux_fraction = self.fiberflux_fraction[self.objtype]

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
                    objmeta[key][ii] = value

                if self.normline.upper() == 'OII':
                    ewoii = 10.0**(np.polyval(self.ewoiicoeff, d4000) + # rest-frame EW([OII]), Angstrom
                                   templaterand.normal(0.0, 0.3, nbase))
                    normlineflux = self.basemeta['OII_CONTINUUM'].data * ewoii

                    emflux, emwave, emline = self.EM.spectrum(linesigma=vdisp[ii], seed=templateseed[ii],
                                                              oiidoublet=oiidoublet, oiiihbeta=oiiihbeta,
                                                              oiihbeta=oiihbeta, niihbeta=niihbeta,
                                                              siihbeta=siihbeta, oiiflux=1.0)

                elif self.normline.upper() == 'HBETA':
                    ewhbeta = 10.0**(np.polyval(self.ewhbetacoeff, d4000) + \
                                     templaterand.normal(0.0, 0.2, nbase)) * \
                                     (self.basemeta['HBETA_LIMIT'].data == 0) # rest-frame H-beta, Angstrom
                    normlineflux = self.basemeta['HBETA_CONTINUUM'].data * ewhbeta

                    emflux, emwave, emline = self.EM.spectrum(linesigma=vdisp[ii], seed=templateseed[ii],
                                                              oiidoublet=oiidoublet, oiiihbeta=oiiihbeta,
                                                              oiihbeta=oiihbeta, niihbeta=niihbeta,
                                                              siihbeta=siihbeta, hbetaflux=1.0)

                emflux /= (1+redshift[ii]) # [erg/s/cm2/A, @redshift[ii]]

            # Optionally get the transient spectrum and normalization factor.
            if self.transient is not None:
                # Evaluate the flux where the model has defined wavelengths.
                # Zero-pad all other wavelength values.
                trans_restflux = np.zeros_like(self.basewave, dtype=float)
                minw = self.transient.minwave().to('Angstrom').value
                maxw = self.transient.maxwave().to('Angstrom').value
                j = np.argwhere(self.basewave >= minw)[0,0]
                k = np.argwhere(self.basewave <= maxw)[-1,0]

                trans_restflux[j:k] = self.transient.flux(trans_epoch[ii], self.basewave[j:k] * u.Angstrom) 
                trans_norm = normfilt[magfilter[ii]].get_ab_maggies(trans_restflux, zwave)

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

                # Optionally add in the transient spectrum.
                if self.transient is not None:
                    galnorm = normfilt[magfilter[ii]].get_ab_maggies(restflux, zwave)
                    trans_factor = galnorm[magfilter[ii]].data * trans_rfluxratio[ii]/trans_norm[magfilter[ii]].data
                    restflux += np.tile(trans_restflux, (nbasechunk, 1)) * np.tile(trans_factor, (npix, 1)).T

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                if south:
                    maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                else:
                    maggies = self.bassmzlswise.get_ab_maggies(restflux, zwave, mask_invalid=True)

                if nocontinuum:
                    magnorm = np.repeat(10**(-0.4*mag[ii]), nbasechunk)
                else:
                    normmaggies = np.array(normfilt[magfilter[ii]].get_ab_maggies(
                        restflux, zwave, mask_invalid=True)[magfilter[ii]])
                    assert(np.all(normmaggies > 0))
                    magnorm = 10**(-0.4*mag[ii]) / normmaggies

                synthnano = dict()
                for key in maggies.columns:
                    synthnano[key] = 1E9 * maggies[key] * magnorm # nanomaggies
                zlineflux = normlineflux[templateid] * magnorm

                if south:
                    gflux, rflux, zflux, w1flux, w2flux = synthnano['decam2014-g'], \
                      synthnano['decam2014-r'], synthnano['decam2014-z'], \
                      synthnano['wise2010-W1'], synthnano['wise2010-W2']
                else:
                    gflux, rflux, zflux, w1flux, w2flux = synthnano['BASS-g'], \
                      synthnano['BASS-r'], synthnano['MzLS-z'], \
                      synthnano['wise2010-W1'], synthnano['wise2010-W2']

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    if self.objtype == 'BGS':
                        _colormask = []
                        for targtype in ('bright', 'faint', 'wise'):
                            _colormask.append(self.colorcuts_function(
                                gflux=gflux, rflux=rflux, zflux=zflux,
                                w1flux=w1flux, rfiberflux=fiberflux_fraction*rflux, 
                                rfibertotflux=fiberflux_fraction*rflux,
                                south=south, targtype=targtype))
                        colormask = np.any( np.ma.getdata(np.vstack(_colormask)), axis=0 )
                    else:
                        colormask = self.colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                            gfiberflux=fiberflux_fraction*gflux, 
                                                            rfiberflux=fiberflux_fraction*rflux, 
                                                            zfiberflux=fiberflux_fraction*zflux,
                                                            w1flux=w1flux, w2flux=w2flux, south=south)
                        
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
                    meta['FLUX_G'][ii] = gflux[this]
                    meta['FLUX_R'][ii] = rflux[this]
                    meta['FLUX_Z'][ii] = zflux[this]
                    meta['FLUX_W1'][ii] = w1flux[this]
                    meta['FLUX_W2'][ii] = w2flux[this]

                    objmeta['D4000'][ii] = d4000[tempid]

                    if self.normline is not None:
                        if self.normline == 'OII':
                            objmeta['OIIFLUX'][ii] = zlineflux[this]
                            objmeta['EWOII'][ii] = ewoii[tempid]
                        elif self.normline == 'HBETA':
                            objmeta['HBETAFLUX'][ii] = zlineflux[this]
                            objmeta['EWHBETA'][ii] = ewhbeta[tempid]

                    break

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        if restframe:
            outwave = self.basewave
        else:
            outwave = self.wave

        return 1e17 * outflux, outwave, meta, objmeta

class ELG(GALAXY):
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 transient=None, tr_fluxratio=(0.01, 1.), tr_epoch=(-10,10), include_mgii=False, colorcuts_function=None,
                 normfilter_north='BASS-g', normfilter_south='decam2014-g',
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the ELG class.  See the GALAXY.__init__ method for documentation
         on the arguments plus the inherited attributes.

        Note:
          By default, we assume the emission-line spectra are normalized to the
          integrated [OII] emission-line flux.

        Args:

        Attributes:
          ewoiicoeff (float, array): empirically derived coefficients to map
            D(4000) to EW([OII]).

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isELG_colors as colorcuts_function

        super(ELG, self).__init__(objtype='ELG', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, normline='OII',
                                  colorcuts_function=colorcuts_function,
                                  normfilter_north=normfilter_north, normfilter_south=normfilter_south,
                                  baseflux=baseflux, basewave=basewave, basemeta=basemeta,
                                  transient=transient, tr_fluxratio=tr_fluxratio, tr_epoch=tr_epoch, include_mgii=include_mgii)

        self.ewoiicoeff = [1.34323087, -5.02866474, 5.43842874]

    def make_templates(self, nmodel=100, zrange=(0.6, 1.6), magrange=(20.0, 23.5),
                       oiiihbrange=(-0.5, 0.2), logvdisp_meansig=(1.9, 0.15),
                       minoiiflux=0.0, trans_filter='decam2014-r',
                       redshift=None, mag=None, vdisp=None, seed=None, input_meta=None,
                       nocolorcuts=False, nocontinuum=False, agnlike=False,
                       novdisp=False, south=True, restframe=False, verbose=False):
        """Build Monte Carlo ELG spectra/templates.

        See the GALAXY.make_galaxy_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        that are specific to the ELG class.

        Args:
          oiiihbrange (float, optional): Minimum and maximum logarithmic [OIII]
            5007/H-beta line-ratio.  Defaults to a uniform distribution between
            (-0.5, 0.2).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(1.9+/-0.15) km/s
          minoiiflux (float, optional): Minimum [OII] 3727 flux (default 0.0
            erg/s/cm2).

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:

        """
        result = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=magrange,
                                            oiiihbrange=oiiihbrange, logvdisp_meansig=logvdisp_meansig,
                                            minlineflux=minoiiflux, redshift=redshift, vdisp=vdisp,
                                            mag=mag, trans_filter=trans_filter,
                                            seed=seed, input_meta=input_meta,
                                            nocolorcuts=nocolorcuts, nocontinuum=nocontinuum, agnlike=agnlike,
                                            novdisp=novdisp, south=south, restframe=restframe, verbose=verbose)
        return result

class BGS(GALAXY):
    """Generate Monte Carlo spectra of bright galaxy survey galaxies (BGSs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 transient=None, tr_fluxratio=(0.01, 1.), tr_epoch=(-10,10), include_mgii=False,
                 colorcuts_function=None, normfilter_north='BASS-r', normfilter_south='decam2014-r',
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the BGS class.  See the GALAXY.__init__ method for documentation
         on the arguments plus the inherited attributes.

        Note:
          By default, we assume the emission-line spectra are normalized to the
          integrated H-beta emission-line flux.

        Args:

        Attributes:
          ewhbetacoeff (float, array): empirically derived coefficients to map
            D(4000) to EW(H-beta).

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isBGS_colors as colorcuts_function

        super(BGS, self).__init__(objtype='BGS', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, normline='HBETA', 
                                  colorcuts_function=colorcuts_function,
                                  normfilter_north=normfilter_north, normfilter_south=normfilter_south,
                                  baseflux=baseflux, basewave=basewave, basemeta=basemeta,
                                  transient=transient, tr_fluxratio=tr_fluxratio, tr_epoch=tr_epoch, include_mgii=include_mgii)

        self.ewhbetacoeff = [1.28520974, -4.94408026, 4.9617704]

    def make_templates(self, nmodel=100, zrange=(0.01, 0.4), magrange=(15.0, 20.0),
                       oiiihbrange=(-1.3, 0.6), logvdisp_meansig=(2.0, 0.17),
                       minhbetaflux=0.0, trans_filter='decam2014-r',
                       redshift=None, mag=None, vdisp=None, seed=None, input_meta=None,
                       nocolorcuts=False, nocontinuum=False, agnlike=False,
                       novdisp=False, south=True, restframe=False, verbose=False):
        """Build Monte Carlo BGS spectra/templates.

         See the GALAXY.make_galaxy_templates function for documentation on the
         arguments and inherited attributes.  Here we only document the
         arguments that are specific to the BGS class.

        Args:
          oiiihbrange (float, optional): Minimum and maximum logarithmic [OIII]
            5007/H-beta line-ratio.  Defaults to a uniform distribution between
            (-1.3, 0.6).
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(2.0+/-0.17) km/s
          minhbetaflux (float, optional): Minimum H-beta flux (default 0.0
            erg/s/cm2).

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:

        """
        result = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=magrange,
                                            oiiihbrange=oiiihbrange, logvdisp_meansig=logvdisp_meansig,
                                            minlineflux=minhbetaflux, redshift=redshift, vdisp=vdisp,
                                            mag=mag, trans_filter=trans_filter,
                                            seed=seed, input_meta=input_meta,
                                            nocolorcuts=nocolorcuts, nocontinuum=nocontinuum, agnlike=agnlike,
                                            novdisp=novdisp, south=south, restframe=restframe, verbose=verbose)
        return result

class LRG(GALAXY):
    """Generate Monte Carlo spectra of luminous red galaxies (LRGs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 transient=None, tr_fluxratio=(0.01, 1.), tr_epoch=(-10,10), colorcuts_function=None,
                 normfilter_north='MzLS-z', normfilter_south='decam2014-z',
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the LRG class.  See the GALAXY.__init__ method for documentation
        on the arguments plus the inherited attributes.

        Note:
          Emission lines (with presumably AGN-like line-ratios) are not yet
          included.

        Args:

        Attributes:

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isLRG_colors as colorcuts_function

        super(LRG, self).__init__(objtype='LRG', minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, normline=None,
                                  colorcuts_function=colorcuts_function,
                                  normfilter_north=normfilter_north, normfilter_south=normfilter_south,
                                  baseflux=baseflux, basewave=basewave, basemeta=basemeta,
                                  transient=transient, tr_fluxratio=tr_fluxratio, tr_epoch=tr_epoch)

    def make_templates(self, nmodel=100, zrange=(0.5, 1.0), magrange=(19.0, 21.5),
                       logvdisp_meansig=(2.3, 0.1),
                       trans_filter='decam2014-r', redshift=None, mag=None, vdisp=None,
                       seed=None, input_meta=None, nocolorcuts=False,
                       novdisp=False, agnlike=False, south=True, restframe=False, verbose=False):
        """Build Monte Carlo BGS spectra/templates.

         See the GALAXY.make_galaxy_templates function for documentation on the
         arguments and inherited attributes.  Here we only document the
         arguments that are specific to the LRG class.

        Args:
          logvdisp_meansig (float, optional): Logarithmic mean and sigma values
            for the (Gaussian) stellar velocity dispersion distribution.
            Defaults to log10-sigma=(2.3+/-0.1) km/s
          agnlike (bool, optional): adopt AGN-like emission-line ratios (not yet
            supported; defaults False).

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:

        """
        result = self.make_galaxy_templates(nmodel=nmodel, zrange=zrange, magrange=magrange,
                                            logvdisp_meansig=logvdisp_meansig, redshift=redshift,
                                            vdisp=vdisp, mag=mag,
                                            trans_filter=trans_filter, seed=seed, input_meta=input_meta,
                                            nocolorcuts=nocolorcuts,
                                            agnlike=agnlike, novdisp=novdisp, south=south,
                                            restframe=restframe, verbose=verbose)

        # Pre-v2.4 templates:
        if 'ZMETAL' in self.basemeta.colnames:
            good = np.where(meta['TEMPLATEID'] != -1)[0]
            if len(good) > 0:
                meta['ZMETAL'][good] = self.basemeta[meta['TEMPLATEID'][good]]['ZMETAL']
                meta['AGE'][good] = self.basemeta[meta['TEMPLATEID'][good]]['AGE']

        return result

class SUPERSTAR(object):
    """Base class for generating Monte Carlo spectra of the various flavors of stars."""

    def __init__(self, objtype='STAR', subtype='', minwave=3600.0, maxwave=10000.0, cdelt=0.2,
                 wave=None, normfilter_north='BASS-r', normfilter_south='decam2014-r',
                 colorcuts_function=None, baseflux=None, basewave=None, basemeta=None):
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
          colorcuts_function (function name): Function to use to select targets
            (must accept a "south" Boolean argument for selecting templates that
            pass the "north" vs "south" color-cuts (default None).
          normfilter_north (str): normalization filter for simulated "north"
            templates.  Each spectrum is normalized to the magnitude in this
            filter bandpass (default 'BASS-r').
          normfilter_south (str): corresponding normalization filter for "south"
            (default 'decam2014-r').

        Attributes:
          wave (numpy.ndarray): Output wavelength array (Angstrom).
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            continuum spectra (erg/s/cm2/A).
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX (Angstrom).
          basemeta (astropy.Table): Table of meta-data [nbase] for each base template.
          normfilt_north (speclite.filters instance): FilterSequence of
            self.normfilter_north.
          normfilt_south (speclite.filters instance): FilterSequence of
            self.normfilter_south.
          sdssrfilt (speclite.filters instance): SDSS2010-r FilterSequence.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.
          bassmzlswise (speclite.filters instance): BASS-[g,r], MzLS-z and
            WISE2010-[W1,W2] FilterSequence.

        """
        from speclite import filters

        self.objtype = objtype.upper()
        self.subtype = subtype.upper()

        self.colorcuts_function = colorcuts_function
        self.normfilter_north = normfilter_north
        self.normfilter_south = normfilter_south

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = int(round((maxwave-minwave) / cdelt))+1
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
        self.normfilt_north = filters.load_filters(self.normfilter_north)
        self.normfilt_south = filters.load_filters(self.normfilter_south)
        self.sdssrfilt = filters.load_filters('sdss2010-r')
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')
        self.bassmzlswise = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z',
                                                 'wise2010-W1', 'wise2010-W2')

    def make_star_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                            magrange=(18.0, 22.0), seed=None, redshift=None,
                            mag=None, input_meta=None, star_properties=None,
                            nocolorcuts=False, south=True, restframe=False,
                            verbose=False):

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
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False). Defaults to a uniform
            distribution between (18, 22).
          seed (int, optional): input seed for the random numbers.
          redshift (float, optional): Input/output (dimensionless) radial
            velocity.  Array size must equal nmodel.  Ignores vrad_meansig
            input.
          mag (float, optional): Input/output template magnitudes in the
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False).  Array size must equal
            nmodel.  Ignores magrange input.
        
          input_meta (astropy.Table): *Input* metadata table with the following
            required columns: TEMPLATEID, SEED, REDSHIFT, MAG, and MAGFILTER
            (see desisim.io.empty_metatable for the expected data types).  If
            present, then all other optional inputs (nmodel, redshift, mag,
            zrange, vrad_meansig, etc.) are ignored.
          star_properties (astropy.Table): *Input* table with the following
            required columns: REDSHIFT, MAG, MAGFILTER, TEFF, LOGG, and FEH
            (except for WDs, which don't need to have an FEH column).
            Optionally, SEED can also be included in the table.  When this table
            is passed, the basis templates are interpolated to the desired
            physical values provided, enabling large numbers of mock stellar
            spectra to be generated with physically consistent properties.
            However, be warned that the interpolation scheme is very
            rudimentary.
        
          nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).
          south (bool, optional): Apply "south" color-cuts using the DECaLS
            filter system, otherwise apply the "north" (MzLS+BASS) color-cuts.
            Defaults to True.
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
        from speclite import filters
        from desispec.interpolation import resample_flux

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        npix = len(self.basewave)
        nbase = len(self.basemeta)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            nmodel = len(input_meta)
            _check_input_meta(input_meta)

            templateseed = input_meta['SEED'].data
            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data
            magfilter = np.char.strip(input_meta['MAGFILTER'].data)

            nchunk = 1
            alltemplateid_chunk = [input_meta['TEMPLATEID'].data.reshape(nmodel, 1)]
        else:
            if star_properties is not None:
                nmodel = len(star_properties)
                _check_star_properties(star_properties, WD=self.objtype=='WD')

                redshift = star_properties['REDSHIFT'].data
                mag = star_properties['MAG'].data
                magfilter = np.char.strip(star_properties['MAGFILTER'].data)

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

                    redshift = np.array(vrad) / C_LIGHT

                if mag is None:
                    mag = rand.uniform(magrange[0], magrange[1], nmodel).astype('f4')

                if south:
                    magfilter = np.repeat(self.normfilter_south, nmodel)
                else:
                    magfilter = np.repeat(self.normfilter_north, nmodel)

        # Basic error checking and some preliminaries.
        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        # Load the unique set of MAGFILTERs.  We could check against
        # self.decamwise.names and self.bassmzlswise to see if the filters have
        # already been loaded, but speed should not be an issue.
        normfilt = dict()
        for mfilter in np.unique(magfilter):
            normfilt[mfilter] = filters.load_filters(mfilter)

        # Initialize the metadata table.
        meta, objmeta = empty_metatable(nmodel=nmodel, objtype=self.objtype, subtype=self.subtype)

        # Populate some of the metadata table.
        for key, value in zip(('REDSHIFT', 'MAG', 'MAGFILTER', 'SEED'),
                               (redshift, mag, magfilter, templateseed)):
            meta[key][:] = value

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
                # color-cuts.  Also, with radial velocity shifts, the >v3.0
                # templates do not go red enough to cover the full r-band
                # spectral range, so pad here.
                if south:
                    padflux, padzwave = self.decamwise.pad_spectrum(restflux, zwave, method='edge')
                    maggies = self.decamwise.get_ab_maggies(padflux, padzwave, mask_invalid=True)
                else:
                    padflux, padzwave = self.bassmzlswise.pad_spectrum(restflux, zwave, method='edge')
                    maggies = self.bassmzlswise.get_ab_maggies(padflux, padzwave, mask_invalid=True)

                if 'W1-R' in self.basemeta.colnames: # new templates
                    sdssrnorm = self.sdssrfilt.get_ab_maggies(padflux, padzwave)['sdss2010-r'].data
                    maggies['wise2010-W1'] = sdssrnorm * 10**(-0.4 * self.basemeta['W1-R'][templateid].data)
                    maggies['wise2010-W2'] = sdssrnorm * 10**(-0.4 * self.basemeta['W2-R'][templateid].data)

                normmaggies = np.array(normfilt[magfilter[ii]].get_ab_maggies(
                    padflux, padzwave, mask_invalid=True)[magfilter[ii]])
                assert(np.all(normmaggies > 0))
                magnorm = 10**(-0.4*mag[ii]) / normmaggies

                synthnano = dict()
                for key in maggies.columns:
                    synthnano[key] = 1E9 * maggies[key] * magnorm

                if south:
                    gflux, rflux, zflux, w1flux, w2flux = synthnano['decam2014-g'], \
                      synthnano['decam2014-r'], synthnano['decam2014-z'], \
                      synthnano['wise2010-W1'], synthnano['wise2010-W2']
                else:
                    gflux, rflux, zflux, w1flux, w2flux = synthnano['BASS-g'], \
                      synthnano['BASS-r'], synthnano['MzLS-z'], \
                      synthnano['wise2010-W1'], synthnano['wise2010-W2']

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = self.colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                        w1flux=w1flux, w2flux=w2flux, south=south)

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
                    meta['FLUX_G'][ii] = gflux[this]
                    meta['FLUX_R'][ii] = rflux[this]
                    meta['FLUX_Z'][ii] = zflux[this]
                    meta['FLUX_W1'][ii] = w1flux[this]
                    meta['FLUX_W2'][ii] = w2flux[this]

                    if star_properties is None:
                        objmeta['TEFF'][ii] = self.basemeta['TEFF'][tempid]
                        objmeta['LOGG'][ii] = self.basemeta['LOGG'][tempid]
                        if 'FEH' in self.basemeta.columns:
                            objmeta['FEH'][ii] = self.basemeta['FEH'][tempid]
                    else:
                        objmeta['TEFF'][ii] = input_properties[1][tempid]
                        objmeta['LOGG'][ii] = input_properties[0][tempid]
                        if 'FEH' in self.basemeta.columns:
                            objmeta['FEH'][ii] = input_properties[2][tempid]

                    break

        # Check to see if any spectra could not be computed.
        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        if restframe:
            return 1e17 * outflux, self.basewave, meta, objmeta
        else:
            return 1e17 * outflux, self.wave, meta, objmeta

class STAR(SUPERSTAR):
    """Generate Monte Carlo spectra of generic stars."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the STAR class.  See the SUPERSTAR.__init__ method for
        documentation on the arguments plus the inherited attributes.

        Args:

        Attributes:

        Raises:

        """
        super(STAR, self).__init__(objtype='STAR', minwave=minwave, maxwave=maxwave,
                                   cdelt=cdelt, wave=wave, baseflux=baseflux,
                                   basewave=basewave, basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       magrange=(18.0, 23.5), seed=None, redshift=None,
                       mag=None, input_meta=None, star_properties=None,
                       south=True, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for generic stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the STAR class.

        Args:
        
        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:

        """
        result = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                          magrange=magrange, seed=seed, redshift=redshift,
                                          mag=mag, input_meta=input_meta,
                                          star_properties=star_properties,
                                          restframe=restframe, verbose=verbose)
        return result

class STD(SUPERSTAR):
    """Generate Monte Carlo spectra of (metal-poor, main sequence turnoff) standard
    stars (STD).

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 colorcuts_function=None, normfilter_north='BASS-r', normfilter_south='decam2014-r',
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the STD class.  See the SUPERSTAR.__init__ method for
        documentation on the arguments plus the inherited attributes.

        Args:

        Attributes:

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isSTD_colors as colorcuts_function

        super(STD, self).__init__(objtype='STD', minwave=minwave, maxwave=maxwave,
                                   cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                   normfilter_north=normfilter_north, normfilter_south=normfilter_south,
                                   baseflux=baseflux, basewave=basewave, basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0), magrange=(16.0, 19.0),
                       seed=None, redshift=None, mag=None, input_meta=None, star_properties=None,
                       nocolorcuts=False, south=True, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for STD stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the STD class.

        Args:

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:

        """
        result = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                          magrange=magrange, seed=seed, redshift=redshift,
                                          mag=mag, input_meta=input_meta,
                                          star_properties=star_properties,
                                          nocolorcuts=nocolorcuts, south=south,
                                          restframe=restframe, verbose=verbose)
        return result

class MWS_STAR(SUPERSTAR):
    """Generate Monte Carlo spectra of Milky Way Survey (magnitude-limited)
    stars.

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 colorcuts_function=None, normfilter_north='BASS-r', normfilter_south='decam2014-r',
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the MWS_STAR class.  See the SUPERSTAR.__init__ method for
        documentation on the arguments plus the inherited attributes.

        Args:

        Attributes:

        Raises:

        """
        if colorcuts_function is None:
            from desitarget.cuts import isMWSSTAR_colors as colorcuts_function
            
        super(MWS_STAR, self).__init__(objtype='MWS_STAR', minwave=minwave, maxwave=maxwave,
                                       cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                       normfilter_north=normfilter_north, normfilter_south=normfilter_south,
                                       baseflux=baseflux, basewave=basewave, basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0), magrange=(16.0, 20.0),
                       seed=None, redshift=None, mag=None, input_meta=None, star_properties=None,
                       nocolorcuts=False, south=True, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for MWS_STAR stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the MWS_STAR class.

        Args:

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:

        """
        result = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                          magrange=magrange, seed=seed, redshift=redshift,
                                          mag=mag, input_meta=input_meta,
                                          star_properties=star_properties,
                                          nocolorcuts=nocolorcuts, south=south,
                                          restframe=restframe, verbose=verbose)
        return result

class WD(SUPERSTAR):
    """Generate Monte Carlo spectra of white dwarfs."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 subtype='DA', colorcuts_function=None,
                 normfilter_north='BASS-g', normfilter_south='decam2014-g',
                 baseflux=None, basewave=None, basemeta=None):
        """Initialize the WD class.  See the SUPERSTAR.__init__ method for documentation
        on the arguments plus the inherited attributes.

        Args:

        Attributes:

        Raises:

        """
        super(WD, self).__init__(objtype='WD', subtype=subtype, minwave=minwave, maxwave=maxwave,
                                 cdelt=cdelt, wave=wave, colorcuts_function=colorcuts_function,
                                 normfilter_north=normfilter_north, normfilter_south=normfilter_south,
                                 baseflux=baseflux, basewave=basewave, basemeta=basemeta)

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0), magrange=(16.0, 19.0),
                       seed=None, redshift=None, mag=None, input_meta=None, star_properties=None,
                       nocolorcuts=False, south=True, restframe=False, verbose=False):
        """Build Monte Carlo spectra/templates for WD stars.

        See the SUPERSTAR.make_star_templates function for documentation on the
        arguments and inherited attributes.  Here we only document the arguments
        which are specific to the WD class.

        Args:

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:
          ValueError: If the INPUT_META or STAR_PROPERTIES table contains
            different values of SUBTYPE.

        """
        log = get_logger()
        
        for intable in (input_meta, star_properties):
            if intable is not None:
                if 'SUBTYPE' in intable.dtype.names:
                    if (self.subtype != '') and ~np.all(intable['SUBTYPE'] == self.subtype):
                        log.warning('WD Class initialized with subtype {}, which does not match input table.'.format(
                            self.subtype))
                        raise ValueError
        
        result = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                          magrange=magrange, seed=seed, redshift=redshift,
                                          mag=mag, input_meta=input_meta,
                                          star_properties=star_properties,
                                          nocolorcuts=nocolorcuts, south=south,
                                          restframe=restframe, verbose=verbose)
        return result

class QSO():
    """Generate Monte Carlo spectra of quasars (QSOs)."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 basewave_min=1200, basewave_max=2.5e4, basewave_R=8000,
                 normfilter_north='BASS-r', normfilter_south='decam2014-r', 
                 colorcuts_function=None, balqso=False, z_wind=0.2):
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
          basewave_min (float, optional): minimum output wavelength when
            noresample=True (in QSO.make_templates) [default 1200 Angstrom].
          basewave_max (float, optional): maximum output wavelength when
            noresample=True (in QSO.make_templates) [default 25000 Angstrom].
          basewave_R (float, optional): output wavelength resolution when
            noresample=True (in QSO.make_templates) [default R=8000].
          colorcuts_function (function name): Function to use to select
            templates that pass the color-cuts.
          normfilter_north (str): normalization filter for simulated "north"
            templates.  Each spectrum is normalized to the magnitude in this
            filter bandpass (default 'BASS-r').
          normfilter_south (str): corresponding normalization filter for "south"
            (default 'decam2014-r').
          balqso (bool, optional): Include broad absorption line (BAL) features
            (default False).
          z_wind (float, optional): Redshift window for sampling (defaults to
            0.2).

        Attributes:
          objtype (str): 'QSO'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          cosmo (astropy.cosmology): Default cosmology object (currently
            hard-coded to FlatLCDM with H0=70, Omega0=0.3).
          normfilt_north (speclite.filters instance): FilterSequence of
            self.normfilter_north.
          normfilt_south (speclite.filters instance): FilterSequence of
            self.normfilter_south.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.
          bassmzlswise (speclite.filters instance): BASS-[g,r], MzLS-z and
            WISE2010-[W1,W2] FilterSequence.

        """
        from astropy.io import fits
        from astropy import cosmology
        from speclite import filters
        from desisim.io import find_basis_template, read_basis_templates
        from desisim import lya_mock_p1d as lyamock

        log = get_logger()

        self.objtype = 'QSO'

        if colorcuts_function is None:
            from desitarget.cuts import isQSO_colors as colorcuts_function

        self.colorcuts_function = colorcuts_function
        self.normfilter_north = normfilter_north
        self.normfilter_south = normfilter_south

        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = int(round((maxwave-minwave) / cdelt))+1
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

        def _fixed_R_dispersion(lam1, lam2, R):
            """"""
            loglam1 = np.log(lam1)
            loglam2 = np.log(lam2)
            dloglam = R**-1
            loglam = np.arange(loglam1,loglam2+dloglam,dloglam)
            return np.exp(loglam)

        self.basewave = _fixed_R_dispersion(basewave_min, basewave_max, basewave_R)

        # Iniatilize the Lyman-alpha mock maker.
        self.lyamock_maker = lyamock.MockMaker()

        # Optionally read the BAL basis templates and resample.
        self.balqso = balqso
        if self.balqso:
            from desisim.bal import BAL
            from desispec.interpolation import resample_flux
            bal = BAL()
            bal_baseflux = np.zeros((len(bal.balmeta), len(self.eigenwave)))
            for ii in range(len(bal.balmeta)):
                bal_baseflux[ii, :] = resample_flux(self.eigenwave, bal.balwave,
                                                    bal.balflux[ii, :], extrapolate=True)
                bal_baseflux[ii, bal_baseflux[ii, :] > 1] = 1.0 # do not exceed unity
            self.bal_baseflux = bal_baseflux
            self.bal_basemeta = bal.balmeta
            self.balmeta = bal.empty_balmeta()

        # Initialize the filter profiles.
        self.normfilt_north = filters.load_filters(self.normfilter_north)
        self.normfilt_south = filters.load_filters(self.normfilter_south)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')
        self.bassmzlswise = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z',
                                                 'wise2010-W1', 'wise2010-W2')

    def _sample_pcacoeff(self, nsample, coeff, samplerand):
        """Draw from the distribution of PCA coefficients."""
        cdf = np.cumsum(coeff, dtype=float)
        cdf /= cdf[-1]
        x = samplerand.uniform(0.0, 1.0, size=nsample)
        return coeff[np.interp(x, cdf, np.arange(0, len(coeff), 1)).astype('int')]

    def make_templates(self, nmodel=100, zrange=(0.5, 4.0), magrange=(17.5, 22.7),
                       seed=None, redshift=None, mag=None, input_meta=None, N_perz=40, 
                       maxiter=20, uniform=False, balprob=0.12, lyaforest=True,
                       noresample=False, nocolorcuts=False, south=True, verbose=False):
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
          The templates are only defined in the range 3500-10000 A (observed). 

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5, 4.0).
          magrange (float, optional): Minimum and maximum magnitude in the
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False). Defaults to a uniform
            distribution between (17, 22.7).
          seed (int, optional): input seed for the random numbers.
          redshift (float, optional): Input/output template redshifts.  Array
            size must equal nmodel.  Ignores zrange input.
          mag (float, optional): Input/output template magnitudes in the
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False).  Array size must equal
            nmodel.  Ignores magrange input.
          input_meta (astropy.Table): *Input* metadata table with the following
            required columns: SEED, REDSHIFT, MAG, and MAGFILTER (see
            desisim.io.empty_metatable for the expected data types).  If
            present, then all other optional inputs (nmodel, redshift, mag,
            zrange, etc.) are ignored.  Note that this argument cannot be used
            (at this time) to precisely reproduce templates that have had BALs
            inserted.
          N_perz (int, optional): Number of templates per redshift redshift
            value to generate (default 20).
          maxiter (int): maximum number of iterations for findng a non-negative
            template that also satisfies the color-cuts (default 20).
          uniform (bool, optional): Draw uniformly from the PCA coefficients
            (default False).
          balprob (float, optional): Probability that a QSO is a BAL (default
            0.12).  Only used if QSO(balqso=True) at instantiation.
          lyaforest (bool, optional): Include Lyman-alpha forest absorption
            (default True).
          noresample (bool, optional): Do not resample the QSO spectra in
            wavelength (default False).
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False).
          south (bool, optional): Apply "south" color-cuts using the DECaLS
            filter system, otherwise apply the "north" (MzLS+BASS) color-cuts.
            Defaults to True.
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta, objmeta) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame wavelength array (Angstrom).
              If noresample=True then this is an [nmodel, npix] array (a
              different observed-frame array for each object), otherwise it's a
              one-dimensional [npix]-length array.
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

        Raises:
          ValueError

        """
        from speclite import filters
        from desispec.interpolation import resample_flux

        if uniform:
            from desiutil.stats import perc

        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        if self.balqso:
            if balprob < 0:
                log.warning('Balprob {} is negative; setting to zero.'.format(balprob))
                balprob = 0.0
            if balprob > 1:
                log.warning('Balprob {} cannot exceed unity; setting to 1.0.'.format(balprob))
                balprob = 1.0

        npix = len(self.eigenwave)

        # Optionally unpack a metadata table.
        if input_meta is not None:
            nmodel = len(input_meta)
            _check_input_meta(input_meta, ignore_templateid=True)

            templateseed = input_meta['SEED'].data
            redshift = input_meta['REDSHIFT'].data
            mag = input_meta['MAG'].data
            magfilter = np.char.strip(input_meta['MAGFILTER'].data)

            meta, objmeta = empty_metatable(nmodel=nmodel, objtype=self.objtype, simqso=False)
        else:
            meta, objmeta = empty_metatable(nmodel=nmodel, objtype=self.objtype, simqso=False)

            if self.balqso:
                from astropy.table import vstack
                balmeta = vstack([self.balmeta for ii in range(nmodel)])

            # Initialize the random seed.
            rand = np.random.RandomState(seed)
            templateseed = rand.randint(2**32, size=nmodel)

            # Assign redshift and magnitude priors.
            if redshift is None:
                redshift = rand.uniform(zrange[0], zrange[1], nmodel)
            else:
                redshift = np.atleast_1d(redshift)

            if mag is None:
                mag = rand.uniform(magrange[0], magrange[1], nmodel).astype('f4')
            else:
                mag = np.atleast_1d(mag)

            if south:
                magfilter = np.repeat(self.normfilter_south, nmodel)
            else:
                magfilter = np.repeat(self.normfilter_north, nmodel)

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('Redshift must be an nmodel-length array')
                raise ValueError
            zrange = (np.min(redshift), np.max(redshift))

        if mag is not None:
            if len(mag) != nmodel:
                log.fatal('Mag must be an nmodel-length array')
                raise ValueError

        # Pre-compute the Lyman-alpha skewers.
        if lyaforest:
            for ii in range(nmodel):
                skewer_wave, skewer_flux1 = self.lyamock_maker.get_lya_skewers(
                    1, new_seed=templateseed[ii])
                if ii == 0:
                    skewer_flux = np.zeros( (nmodel, len(skewer_wave)) )
                skewer_flux[ii, :] = skewer_flux1

        # Populate some of the metadata table.
        meta['TEMPLATEID'][:] = np.arange(nmodel)
        for key, value in zip(('REDSHIFT', 'MAG', 'MAGFILTER', 'SEED'),
                               (redshift, mag, magfilter, templateseed)):
            meta[key][:] = value

        if lyaforest: 
            meta['SUBTYPE'][:] = 'LYA'
            
        if self.balqso:
            balmeta['Z'][:] = redshift
            if lyaforest: 
                meta['SUBTYPE'][:] = 'LYA+BAL'
            else:
                meta['SUBTYPE'][:] = 'BAL'

        # Load the unique set of MAGFILTERs.  We could check against
        # self.decamwise.names and self.bassmzlswise to see if the filters have
        # already been loaded, but speed should not be an issue.
        normfilt = dict()
        for mfilter in np.unique(magfilter):
            normfilt[mfilter] = filters.load_filters(mfilter)

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
        if noresample:
            outflux = np.zeros([nmodel, len(self.eigenwave)]) # [erg/s/cm2/A]
        else:
            outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        for ii in range(nmodel):
            if ii % 100 == 0 and ii > 0:
                log.debug('Simulating {} template {}/{}.'.format(self.objtype, ii, nmodel))

            templaterand = np.random.RandomState(templateseed[ii])

            # Does this QSO have a BAL?  If so, build the spectrum here.
            hasbal = self.balqso * (templaterand.random_sample() < balprob)
            if hasbal:
                balindx = templaterand.choice(len(self.bal_basemeta))
                balflux = self.bal_baseflux[balindx, :]
                balmeta['BAL_TEMPLATEID'][ii] = balindx

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
                    if hasbal:
                        flux[kk, :] *= balflux

                    nonegflux[kk] = (np.sum(flux[kk, (zwave[:, ii] > 3000) & (zwave[:, ii] < 1E4)] < 0) == 0) * 1
                        
                # Synthesize photometry to determine which models will pass the
                # color-cuts.  We have to temporarily pad because the spectra
                # don't go red enough.
                padflux, padzwave = self.decamwise.pad_spectrum(flux, zwave[:, ii], method='edge')
                if south:
                    maggies = self.decamwise.get_ab_maggies(padflux, padzwave, mask_invalid=True)
                else:
                    maggies = self.bassmzlswise.get_ab_maggies(padflux, padzwave, mask_invalid=True)

                normmaggies = np.array(normfilt[magfilter[ii]].get_ab_maggies(
                    padflux, padzwave, mask_invalid=True)[magfilter[ii]])
                assert(np.all(normmaggies[np.where(nonegflux)[0]] > 0))
                magnorm = 10**(-0.4*mag[ii]) / normmaggies

                synthnano = dict()
                for key in maggies.columns:
                    synthnano[key] = 1E9 * maggies[key] * magnorm

                if south:
                    gflux, rflux, zflux, w1flux, w2flux = synthnano['decam2014-g'], \
                      synthnano['decam2014-r'], synthnano['decam2014-z'], \
                      synthnano['wise2010-W1'], synthnano['wise2010-W2']
                else:
                    gflux, rflux, zflux, w1flux, w2flux = synthnano['BASS-g'], \
                      synthnano['BASS-r'], synthnano['MzLS-z'], \
                      synthnano['wise2010-W1'], synthnano['wise2010-W2']
                          
                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, N_perz)
                else:
                    colormask = self.colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                        w1flux=w1flux, w2flux=w2flux, south=south,
                                                        optical=True)
                          
                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask * nonegflux):
                    this = templaterand.choice(np.where(colormask * nonegflux)[0]) # Pick one randomly.
                    if noresample:
                        outflux[ii, :] = flux[this, :] * magnorm[this]
                    else:
                        outflux[ii, :] = resample_flux(self.wave, zwave[:, ii], flux[this, :],
                                                       extrapolate=True) * magnorm[this]

                    meta['FLUX_G'][ii] = gflux[this]
                    meta['FLUX_R'][ii] = rflux[this]
                    meta['FLUX_Z'][ii] = zflux[this]
                    meta['FLUX_W1'][ii] = w1flux[this]
                    meta['FLUX_W2'][ii] = w2flux[this]

                    objmeta['PCA_COEFF'][ii, :] = PCA_rand[:, this].T
                    if hasbal:
                        objmeta['BAL_TEMPLATEID'][ii] = balindx
                    
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

        if noresample:
            outwave = zwave.T
        else:
            outwave = self.wave
    
        return 1e17 * outflux, outwave, meta, objmeta

class SIMQSO():
    """Generate Monte Carlo spectra of quasars (QSOs) using simqso."""

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=0.2, wave=None,
                 nproc=1, basewave_min=450.0, basewave_max=6e4, basewave_R=8000,
                 normfilter_north='BASS-r', normfilter_south='decam2014-r', 
                 colorcuts_function=None, restframe=False,sqmodel='default'):
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
          nproc (int, optional): number of cores to use (default 1).
          basewave_min (float, optional): minimum output wavelength when either
            restframe=True or noresample=True (in SIMQSO.make_templates)
            [default 450 Angstrom].
          basewave_max (float, optional): maximum output wavelength when either
            restframe=True or noresample=True (in SIMQSO.make_templates)
            [default 60000 Angstrom].
          basewave_R (float, optional): output wavelength resolution when either
            restframe=True or noresample=True (in SIMQSO.make_templates)
            [default R=8000].
          normfilter_north (str): normalization filter for simulated "north"
            templates.  Each spectrum is normalized to the magnitude in this
            filter bandpass (default 'BASS-r').
          normfilter_south (str): corresponding normalization filter for "south"
            (default 'decam2014-r').
          colorcuts_function (function name): Function to use to select
            templates that pass the color-cuts.
          restframe (bool, optional): If True, generate rest-frame templates.

        Attributes:
          objtype (str): 'QSO'
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          procMap (map Class): Built-in map for multiprocessing (based on nproc). 
          cosmo (astropy.cosmology): Default cosmology object (currently
            hard-coded to FlatLCDM with H0=70, Omega0=0.3).
          normfilt_north (speclite.filters instance): FilterSequence of
            self.normfilter_north.
          normfilt_south (speclite.filters instance): FilterSequence of
            self.normfilter_south.
          decamwise (speclite.filters instance): DECam2014-[g,r,z] and WISE2010-[W1,W2]
            FilterSequence.
          bassmzlswise (speclite.filters instance): BASS-[g,r], MzLS-z and
            WISE2010-[W1,W2] FilterSequence.

        """
        from astropy import cosmology
        from speclite import filters
        log = get_logger()
        try:
            from simqso.sqbase import ContinuumKCorr, fixed_R_dispersion
            #Added in order to use modified emision lines in quickquasars and select_mock_targets.
            if sqmodel=='lya_simqso_model_develop':
                #Added in order to test a different model than the one currently used in quickquasars
                from desisim.scripts.lya_simqso_model import model_PLEpivot as model_PLEpivot
                from desisim.scripts.lya_simqso_model import model_vars_develop as sqmodel_vars
                log.warning("Using simqso.sqmodel under development defined in desisim.scripts.lya_simqso_model")
            elif sqmodel=='lya_simqso_model':
                from desisim.scripts.lya_simqso_model import model_PLEpivot as model_PLEpivot
                from desisim.scripts.lya_simqso_model import model_vars as sqmodel_vars
                log.warning("Using modified simqso.sqmodel defined in desisim.scripts.lya_simqso_model")
            else:
                from simqso.sqmodels import BOSS_DR9_PLEpivot as model_PLEpivot
                from simqso.sqmodels import get_BossDr9_model_vars as sqmodel_vars
                log.warning("Using default SIMQSO model")

            self.sqmodel_vars=sqmodel_vars
        except ImportError:
            message = 'Please install https://github.com/imcgreer/simqso'
            log.error(message)
            raise(ImportError(message))

        self.objtype = 'QSO'

        if colorcuts_function is None:
            from desitarget.cuts import isQSO_colors as colorcuts_function

        self.colorcuts_function = colorcuts_function
        self.normfilter_north = normfilter_north
        self.normfilter_south = normfilter_south

        # Initialize multiprocessing map object.
        if nproc > 1:
            pool = multiprocessing.Pool(nproc)
            self.procMap = pool.map
        else:
            self.procMap = map
        
        # Initialize the output wavelength array (linear spacing) unless it is
        # already provided.
        if wave is None:
            npix = int(round((maxwave-minwave) / cdelt))+1
            wave = np.linspace(minwave, maxwave, npix)
        self.wave = wave

        self.restframe = restframe
        if restframe:
            self._zpivot = 3.0
            self.basewave = fixed_R_dispersion(basewave_min*(1+self._zpivot),
                                               basewave_max*(1+self._zpivot),
                                               basewave_R)
        else:
            self.basewave = fixed_R_dispersion(basewave_min, basewave_max, basewave_R)
            
        self.cosmo = cosmology.core.FlatLambdaCDM(70.0, 0.3)

        self.lambda_lylimit = 911.76
        self.lambda_lyalpha = 1215.67

        # Initialize the filter profiles.
        self.normfilt_north = filters.load_filters(self.normfilter_north)
        self.normfilt_south = filters.load_filters(self.normfilter_south)
        self.decamwise = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                              'wise2010-W1', 'wise2010-W2')
        self.bassmzlswise = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z',
                                                 'wise2010-W1', 'wise2010-W2')

        # Initialize the BOSS/DR9 quasar luminosity function and k-correction
        # objects.

        # Map between speclite and simqso filter names.
        filtnames = dict()
        for filt1, filt2 in zip( ('decam2014-g', 'decam2014-r', 'decam2014-z', 'BASS-g', 'BASS-r', 'MzLS-z'),
                                 ('DECam-g', 'DECam-r', 'DECam-z', 'BASS-MzLS-g', 'BASS-MzLS-r', 'MzLS-MzLS-z') ):
            filtnames[filt1] = filt2
        if normfilter_north not in filtnames.keys() or normfilter_south not in filtnames.keys():
            log.warning('Unrecognized normalization filter!')
            raise ValueError
            
        # Initialize the K-correction and luminosity function objects.
        self.kcorr_north = ContinuumKCorr(filtnames[self.normfilter_north], 1450,
                                          effWaveBand=self.normfilt_north.effective_wavelengths.value)
        self.kcorr_south = ContinuumKCorr(filtnames[self.normfilter_south], 1450,
                                          effWaveBand=self.normfilt_south.effective_wavelengths.value)
        self.qlf = model_PLEpivot(cosmo=self.cosmo)

    def empty_qsometa(self, qsometa, nmodel):
        """Initialize an empty QsoSimPoints object, which contains all the metadata
        needed to regenerate simqso spectra.
        
        """
        qsometa.data.remove_rows(np.arange(len(qsometa.data)))
        [qsometa.data.add_row() for ii in range(nmodel)]

        return qsometa

    def _make_simqso_templates(self, redshift=None, magrange=None, seed=None,                               
                               lyaforest=True, nocolorcuts=False, noresample=False,
                               input_qsometa=None, south=True):
        """Wrapper function for actually generating the templates.

        """ 
        from astropy.table import Column
        from desispec.interpolation import resample_flux
        if lyaforest:
            subtype = 'LYA'
        else:
            subtype = ''
            
        if input_qsometa:
            nmodel = len(input_qsometa.data)
        else:
            nmodel = len(redshift)
            
        meta, objmeta = empty_metatable(nmodel=nmodel, objtype='QSO',
                                        subtype=subtype, simqso=True)
        if noresample or self.restframe:
            outflux = np.zeros([nmodel, len(self.basewave)])
        else:
            outflux = np.zeros([nmodel, len(self.wave)])

        if input_qsometa:
            from simqso.sqrun import buildQsoSpectrum
            from simqso.sqgrids import SpectralFeatureVar

            flux = np.zeros([nmodel, len(self.basewave)])

            specFeatures = input_qsometa.getVars(SpectralFeatureVar)
            for ii in range(nmodel):
                flux1 = buildQsoSpectrum(self.basewave, input_qsometa.cosmo,
                                         specFeatures, input_qsometa.data[ii])
                flux[ii, :] = flux1.f_lambda

            qsometa = input_qsometa

        else:
            from simqso.sqrun import buildSpectraBulk
            from simqso.sqgrids import generateQlfPoints

            # Sample from the QLF, using the input redshifts.
            zrange = (np.min(redshift), np.max(redshift))
            if south:
                qsometa = generateQlfPoints(self.qlf, magrange, zrange, zin=redshift,
                                            kcorr=self.kcorr_south, qlfseed=seed, gridseed=seed)
            else:
                qsometa = generateQlfPoints(self.qlf, magrange, zrange, zin=redshift,
                                            kcorr=self.kcorr_north, qlfseed=seed, gridseed=seed)

            # Add the fiducial quasar SED model from BOSS/DR9, optionally
            # without IGM absorption. This step adds a fiducial continuum,
            # emission-line template, and an iron emission-line template.
            qsometa.addVars(self.sqmodel_vars(qsometa, self.basewave, noforest=not lyaforest))

            # Establish the desired (output) photometric system.
            if south:
                qsometa.loadPhotoMap([('DECam', 'DECaLS'), ('WISE', 'AllWISE')])
            else:
                qsometa.loadPhotoMap([('BASS-MzLS', 'BASS-MzLS'), ('WISE', 'AllWISE')])

            # Finally, generate the spectra, iterating in order to converge on the
            # per-object K-correction (typically, after ~two steps the maximum error
            # on the absolute mags is typically <<1%).
            _, flux = buildSpectraBulk(self.basewave, qsometa, maxIter=5, verbose=0,
                                       procMap=self.procMap, saveSpectra=True)

        # Synthesize photometry to determine which models will pass the
        # color cuts.
        if south:
            magfilt = self.normfilt_south
            magfilter = self.normfilter_south
            maggies = self.decamwise.get_ab_maggies(flux, self.basewave.copy(), mask_invalid=True)
        else:
            magfilt = self.normfilt_north
            magfilter = self.normfilter_north
            maggies = self.bassmzlswise.get_ab_maggies(flux, self.basewave.copy(), mask_invalid=True)

        normmaggies = np.array(magfilt.get_ab_maggies(flux, self.basewave.copy(), mask_invalid=True)[magfilter])
        assert(np.all(normmaggies > 0))

        synthnano = dict()
        for key in maggies.columns:
            synthnano[key] = 1E9 * maggies[key]

        if south:
            gflux, rflux, zflux, w1flux, w2flux = synthnano['decam2014-g'], \
              synthnano['decam2014-r'], synthnano['decam2014-z'], \
              synthnano['wise2010-W1'], synthnano['wise2010-W2']
        else:
            gflux, rflux, zflux, w1flux, w2flux = synthnano['BASS-g'], \
              synthnano['BASS-r'], synthnano['MzLS-z'], \
              synthnano['wise2010-W1'], synthnano['wise2010-W2']

        if nocolorcuts or self.colorcuts_function is None:
            colormask = np.repeat(1, nmodel)
        else:
            colormask = self.colorcuts_function(gflux=gflux, rflux=rflux, zflux=zflux,
                                                w1flux=w1flux, w2flux=w2flux, south=south)

        # For objects that pass the color cuts, populate the output flux
        # vector, the metadata and qsometadata tables, and finish up.
        these = np.where(colormask)[0]
        if len(these) > 0:
            for ii in range(len(these)):
                if noresample or self.restframe:
                    outflux[these[ii], :] = flux[these[ii], :]
                else:
                    outflux[these[ii], :] = resample_flux(
                        self.wave, self.basewave, flux[these[ii], :],
                        extrapolate=True)

            if input_qsometa:
                meta['SEED'][these] = input_qsometa.seed
                meta['REDSHIFT'][these] = input_qsometa.z
            else:
                meta['SEED'][these] = seed
                if self.restframe:
                    meta['REDSHIFT'][these] = 0
                else:
                    meta['REDSHIFT'][these] = redshift[these]

            meta['MAGFILTER'][these] = magfilter
            meta['MAG'][these] = -2.5 * np.log10(normmaggies[these])

            meta['FLUX_G'][these] = gflux[these]
            meta['FLUX_R'][these] = rflux[these]
            meta['FLUX_Z'][these] = zflux[these]
            meta['FLUX_W1'][these] = w1flux[these]
            meta['FLUX_W2'][these] = w2flux[these]

            objmeta['MABS_1450'][these] = qsometa.data['absMag'][these]
            objmeta['SLOPES'][these, :] = qsometa.data['slopes'][these, :]
            #Added because some emision line model tables have different lenght than default.
            if(objmeta['EMLINES'].shape!=qsometa.data['emLines'].shape):
                objmeta.replace_column('EMLINES',np.zeros((nmodel,len(qsometa.data['emLines'][0, :, 0]), 3))-1)
            objmeta['EMLINES'][these, :, :] = qsometa.data['emLines'][these, :, :]
        return outflux, meta, objmeta, qsometa

    def make_templates(self, nmodel=100, zrange=(0.5, 4.0), magrange=(17.0, 22.7),
                       seed=None, redshift=None, mag=None, maxiter=20,
                       input_qsometa=None, qsometa_extname='QSOMETA', return_qsometa=False, 
                       lyaforest=True, nocolorcuts=False, noresample=False,
                       south=True, verbose=False):
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
          brightness of the resulting QSO spectra by specifying magrange.

        * The way the code is currently structured could lead to memory problems
          if one attempts to generate very large numbers of spectra
          simultaneously (>10^4, perhaps, depending on the machine).  However,
          it can easily be refactored to generate the appropriate number of
          templates in chunks at the expense of some computational speed.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5, 4.0).
          magrange (float, optional): Minimum and maximum magnitude in the
            bandpass specified by self.normfilter_south (if south=True) or
            self.normfilter_north (if south=False). Defaults to a uniform
            distribution between (17, 22.7).
          seed (int, optional): input seed for the random numbers.
          redshift (float, optional): Input/output template redshifts.  Array
            size must equal nmodel.  Ignores zrange input.
          mag (float, optional): Not currently supported or used, but see
            magrange.  Defaults to None.
          maxiter (int): maximum number of iterations for findng a template that
            satisfies the color-cuts (default 20).
          input_qsometa (simqso.sqgrids.QsoSimPoints object or FITS filename):
            Input QsoSimPoints object or FITS filename (with a qsometa_extname
            HDU) from which to (re)generate the QSO spectra.  All other inputs
            are ignored when this optional input is present.  Please be cautious
            when using this argument, as it has not been fully tested.
          qsometa_extname (str): FITS extension name to read when input_qsometa
            is a filename.  Defaults to 'QSOMETA'.
          return_qsometa (bool, optional): Return the
            simqso.sqgrids.QsoSimPoints object, which contains all the data
            necessary to regenerate the QSO spectra.  In particular, the data
            attribute is an astropy.Table object which contains lots of useful
            info.  This object can be written to disk with the
            simqso.sqgrids.QsoSimObjects.write method (default False).
          lyaforest (bool, optional): Include Lyman-alpha forest absorption
            (default True).
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False).
          noresample (bool, optional): Do not resample the QSO spectra in
            wavelength (default False).
          south (bool, optional): Apply "south" color-cuts using the DECaLS
            filter system, otherwise apply the "north" (MzLS+BASS) color-cuts.
            Defaults to True.
          verbose (bool, optional): Be verbose!

        Returns (outflux, wave, meta, qsometa) tuple where:

          * outflux (numpy.ndarray): Array [nmodel, npix] of observed-frame
            spectra (1e-17 erg/s/cm2/A).
          * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
          * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum.
          * objmeta (astropy.Table): Additional objtype-specific table data
            [nmodel] for each spectrum.

          In addition, if return_qsometa=True then a fourth argument, qsometa,
          is returned (see the return_qsometa documentation, above).

        Raises:
          ValueError

        """
        from astropy.table import Column
        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        if self.restframe:
            log.debug('Setting nocolorcuts=True.')
            nocolorcuts = True

        # Optionally generate spectra from an input file or a
        # simqso.sqgrids.QsoSimPoints object.
        if input_qsometa:
            from simqso.sqgrids import QsoSimObjects

            if self.restframe:
                log.warning('restframe and input_qsometa inputs cannot be used together.')
                raise ValueError
            
            if isinstance(input_qsometa, QsoSimObjects):
                qsos = input_qsometa
            else:
                log.debug('Reading {} extension from {}'.format(qsometa_extname, input_qsometa))
                qsos = input_qsometa.read(input_qsometa, extname=qsometa_extname)
                
            nmodel = len(input_qsometa.data)
            outflux, meta, objmeta, qsometa = self._make_simqso_templates(
                input_qsometa=qsos, lyaforest=lyaforest, noresample=noresample,
                nocolorcuts=nocolorcuts, south=south)

            log.debug('Generated {} templates from an input qso metadata table.'.format(
                len(objmeta)))

        else:
            if self.restframe and noresample:
                log.warning('restframe and noresample inputs cannot be used together.')
                raise ValueError

            # Initialize the random seed and assign redshift priors.
            rand = np.random.RandomState(seed)

            if redshift is not None:
                if len(redshift) != nmodel:
                    log.warning('Redshift must be an nmodel-length array')
                    raise ValueError

            # Initialize the template metadata table and flux vector. 
            meta, objmeta = empty_metatable(nmodel=nmodel, objtype='QSO', simqso=True)
            qsometa = None

            if noresample or self.restframe:
                outflux = np.zeros([nmodel, len(self.basewave)])
            else:
                outflux = np.zeros([nmodel, len(self.wave)])

            if self.restframe:
                redshift = np.repeat(self._zpivot, nmodel)

            # Iterate (up to maxiter) until enough spectra pass the color cuts.
            itercount = 0
            iterseed = rand.randint(2**32, size=maxiter)

            def _need(outflux):
                return np.where( np.sum(outflux, axis=1) == 0 )[0]

            need = _need(outflux)
            
            while (len(need) > 0):
                if redshift is None:
                    zin = rand.uniform(zrange[0], zrange[1], len(need))
                else:
                    zin = redshift[need]

                iterflux, itermeta, iterobjmeta, iterqsometa = self._make_simqso_templates(
                    zin, magrange, seed=iterseed[itercount], lyaforest=lyaforest,
                    nocolorcuts=nocolorcuts, noresample=noresample, south=south)

                outflux[need, :] = iterflux
                if(objmeta['EMLINES'].shape!=iterobjmeta['EMLINES'].shape):
                    objmeta.replace_column('EMLINES',np.zeros((nmodel,len(iterobjmeta['EMLINES'][0, :, 0]), 3))-1)
                meta[need] = itermeta
                objmeta[need] = iterobjmeta
                if qsometa is None:
                    _data = iterqsometa.data.copy()
                    qsometa = self.empty_qsometa(iterqsometa, nmodel=nmodel)
                    qsometa.data[need] = _data
                else:
                    qsometa.data[need] = iterqsometa.data

                need = _need(outflux)

                itercount += 1
                if itercount == maxiter:
                    log.warning('Maximum number of iterations reached.')
                    break

            log.debug('Generated {} templates after {}/{} iterations.'.format(
                nmodel, itercount, maxiter))

        success = (np.sum(outflux, axis=1) > 0)*1
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input priors!'.\
                        format(np.sum(success == 0)))

        meta['TEMPLATEID'] = np.arange(nmodel)

        if noresample:
            outwave = self.basewave
        elif self.restframe:
            outwave = self.basewave / (1 + self._zpivot)
        else:
            outwave = self.wave

        if return_qsometa:
            return 1e17 * outflux, outwave, meta, objmeta, qsometa
        else:
            return 1e17 * outflux, outwave, meta, objmeta

def specify_galparams_dict(templatetype, zrange=None, magrange=None,
                            oiiihbrange=None, logvdisp_meansig=None,
                            minlineflux=None, trans_rfluxratiorange=None,
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
        * minlineflux=0.0, trans_rfluxratiorange=(0.01, 0.1),
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
        * trans_rfluxratiorange (float, optional): r-band flux ratio of the SNeIa
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
            is specified by self.normfilter).  In addition, if transient is True
            then the table must also contain SNE_TEMPLATEID, SNE_EPOCH, and
            SNE_RFLUXRATIO columns.  See desisim.io.empty_metatable for the
            required data type for each column.  If this table is passed then
            all other optional inputs (nmodel, redshift, vdisp, mag, zrange,
            logvdisp_meansig, etc.) are ignored.
        * nocolorcuts (bool, optional): Do not apply the color-cuts specified by
            the self.colorcuts_function function (default False).
        * nocontinuum (bool, optional): Do not include the stellar continuum in
            the output spectrum (useful for testing; default False).  Note that
            this option automatically sets nocolorcuts to True and transient to
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
        fulldef_dict['magrange'] = magrange
    if oiiihbrange is not None:
        fulldef_dict['oiiihbrange'] = oiiihbrange
    if logvdisp_meansig is not None:
        fulldef_dict['logvdisp_meansig'] = logvdisp_meansig
    if minlineflux is not None:
        fulldef_dict['minlineflux'] = minlineflux
    if trans_rfluxratiorange is not None:
        fulldef_dict['trans_rfluxratiorange'] = trans_rfluxratiorange
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
        fulldef_dict['magrange'] = magrange
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
