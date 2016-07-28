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

def _lineratios(nobj=1, EM=None, oiiihbrange=(-0.5, 0.2), rand=None):
    """Get the correct number and distribution of emission-line ratios."""
    if EM is None:
        EM = EMSpectrum()
    if rand is None:
        rand = np.random.RandomState()

    oiihbeta = np.zeros(nobj)
    niihbeta = np.zeros(nobj)
    siihbeta = np.zeros(nobj)
    oiiihbeta = np.zeros(nobj)-99
    need = np.where(oiiihbeta==-99)[0]
    while len(need) > 0:
        samp = EM.forbidmog.sample(len(need), random_state=rand)
        oiiihbeta[need] = samp[:,0]
        oiihbeta[need] = samp[:,1]
        niihbeta[need] = samp[:,2]
        siihbeta[need] = samp[:,3]
        oiiihbeta[oiiihbeta<oiiihbrange[0]] = -99
        oiiihbeta[oiiihbeta>oiiihbrange[1]] = -99
        need = np.where(oiiihbeta==-99)[0]

    return oiihbeta, niihbeta, siihbeta, oiiihbeta

def _metatable(nmodel=1, objtype='ELG', add_SNeIa=None):
    """Initialize the metadata table for each object type.""" 
    from astropy.table import Table, Column

    uobjtype = objtype.upper()

    meta = Table()
    meta.add_column(Column(name='TEMPLATEID', length=nmodel, dtype='i4'))
    meta.add_column(Column(name='REDSHIFT', length=nmodel, dtype='f4'))
    meta.add_column(Column(name='GMAG', length=nmodel, dtype='f4'))
    meta.add_column(Column(name='RMAG', length=nmodel, dtype='f4'))
    meta.add_column(Column(name='ZMAG', length=nmodel, dtype='f4'))
    meta.add_column(Column(name='W1MAG', length=nmodel, dtype='f4'))
    meta.add_column(Column(name='W2MAG', length=nmodel, dtype='f4'))
    meta.add_column(Column(name='DECAM_FLUX', shape=(6,), length=nmodel, dtype='f4'))
    meta.add_column(Column(name='WISE_FLUX', shape=(2,), length=nmodel, dtype='f4'))

    if uobjtype == 'ELG':
        meta.add_column(Column(name='OIIFLUX', length=nmodel, dtype='f4', unit='erg/(s*cm2)'))
        meta.add_column(Column(name='EWOII', length=nmodel, dtype='f4', unit='Angstrom'))
        
    if uobjtype == 'BGS':
        meta.add_column(Column(name='HBETAFLUX', length=nmodel, dtype='f4', unit='erg/(s*cm2)'))
        meta.add_column(Column(name='EWHBETA', length=nmodel, dtype='f4', unit='Angstrom'))
        
    if uobjtype == 'ELG' or uobjtype == 'BGS':
        meta.add_column(Column(name='OIIDOUBLET', length=nmodel, dtype='f4'))
        meta.add_column(Column(name='OIIIHBETA', length=nmodel, dtype='f4', unit='dex'))
        meta.add_column(Column(name='OIIHBETA', length=nmodel, dtype='f4', unit='dex'))
        meta.add_column(Column(name='NIIHBETA', length=nmodel, dtype='f4', unit='dex'))
        meta.add_column(Column(name='SIIHBETA', length=nmodel, dtype='f4', unit='dex'))
        meta.add_column(Column(name='D4000', length=nmodel, dtype='f4'))
        meta.add_column(Column(name='VDISP', length=nmodel, dtype='f4', unit='km/s'))

    if uobjtype == 'LRG':
        meta.add_column(Column(name='ZMETAL', length=nmodel, dtype='f4'))
        meta.add_column(Column(name='AGE', length=nmodel, dtype='f4', unit='Gyr'))
        meta.add_column(Column(name='D4000', length=nmodel, dtype='f4'))
        meta.add_column(Column(name='VDISP', length=nmodel, dtype='f4', unit='km/s'))

    if uobjtype == 'STAR' or uobjtype == 'MWS_STAR' or uobjtype == 'FSTD':
        meta.add_column(Column(name='TEFF', length=nmodel, dtype='f4', unit='K'))
        meta.add_column(Column(name='LOGG', length=nmodel, dtype='f4', unit='m/(s**2)'))
        meta.add_column(Column(name='FEH', length=nmodel, dtype='f4'))

    if uobjtype == 'WD':
        meta.add_column(Column(name='TEFF', length=nmodel, dtype='f4', unit='K'))
        meta.add_column(Column(name='LOGG', length=nmodel, dtype='f4', unit='m/(s**2)'))

    if add_SNeIa:
        meta.add_column(Column(name='SNE_TEMPLATEID', length=nmodel, dtype='i4'))
        meta.add_column(Column(name='SNE_RFLUXRATIO', length=nmodel, dtype='f4'))
        meta.add_column(Column(name='SNE_EPOCH', length=nmodel, dtype='f4', unit='days'))

    return meta

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

class GALAXY(object):
    """Base class for generating Monte Carlo spectra of the various flavors of
galaxies (ELG, LRG, BGS).

    """
    def __init__(self, objtype='ELG', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, add_SNeIa=False):
        """Read the appropriate basis continuum templates, filter profiles and
           initialize the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        TODO (@moustakas): Incorporate size and morphological priors.

        Args:
          objtype (str): object type (default 'ELG')
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].
          add_SNeIa (boolean, optional): optionally include a random-epoch SNe
            Ia spectrum in the integrated spectrum [default False]

        Attributes:
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          pixbound (numpy.ndarray): Pixel boundaries of BASEWAVE [Angstrom].
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

        # Optionally read the SNe Ia basis templates and resample.
        self.add_SNeIa = add_SNeIa
        if self.add_SNeIa:
            sne_baseflux1, sne_basewave, sne_basemeta = read_basis_templates(objtype='SNE')
            sne_baseflux = np.zeros((len(sne_basemeta), len(self.basewave)))
            for ii in range(len(sne_basemeta)):
                sne_baseflux[ii,:] = resample_flux(self.basewave, sne_basewave, sne_baseflux1[ii,:])
            self.sne_baseflux = sne_baseflux
            self.sne_basemeta = sne_basemeta

        # Pixel boundaries
        self.pixbound = pxs.cen2bound(basewave)

        # Initialize the filter profiles.
        self.gfilt = filters.load_filters('decam2014-g')
        self.rfilt = filters.load_filters('decam2014-r')
        self.zfilt = filters.load_filters('decam2014-z')
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')

class ELG(GALAXY):
    """Generate Monte Carlo spectra of emission-line galaxies (ELGs).

    """
    def __init__(self, objtype='ELG', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, add_SNeIa=False):
        """Read the BGS basis continuum templates, filter profiles and initialize the
           output wavelength array.
        
        Attributes:
          ewoiicoeff (float, array): empirically derived coefficients to map
            D(4000) to EW([OII]).

        """
        super(ELG, self).__init__(objtype=objtype, minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, add_SNeIa=add_SNeIa)

        self.ewoiicoeff = [1.34323087, -5.02866474, 5.43842874]
        
    def make_templates(self, nmodel=100, zrange=(0.6,1.6), rmagrange=(21.0,23.4),
                       oiiihbrange=(-0.5,0.2), oiidoublet_meansig=(0.73,0.05),
                       logvdisp_meansig=(1.9,0.15), minoiiflux=1E-18,
                       sne_rfluxratiorange=(0.1,1.0), redshift=None,
                       seed=None, nocolorcuts=False, nocontinuum=False):
        """Build Monte Carlo set of ELG spectra/templates.

        This function chooses random subsets of the ELG continuum spectra, constructs
        an emission-line spectrum, redshifts, and then finally normalizes the spectrum
        to a specific r-band magnitude.

        In detail, each (output) model gets randomly assigned a continuum
        (basis) template.  However, if that template doesn't pass the color cuts
        (at the specified redshift), then we iterate through the rest of the
        templates.  If no template passes the color cuts, then raise an
        exception.  If we don't care about color cuts, just grab one template
        for each output model.

        TODO (@moustakas): optionally normalize to a g-band magnitude.

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
          minoiiflux (float, optional): Minimum [OII] 3727 flux [default 1E-18 erg/s/cm2].
            Set this parameter to zero to not have a minimum flux cut.

          sne_rfluxratiorange (float, optional): r-band flux ratio of the SNeIa
            spectrum with respect to the underlying galaxy.

          redshift (float, optional): Input/output template redshifts.  Array
            size must equal NMODEL.  Overwrites ZRANGE input.
          seed (long, optional): Input seed for the random numbers.
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
        from desisim.templates import EMSpectrum
        from desispec.interpolation import resample_flux
        from desisim import pixelsplines as pxs
        from desitarget.cuts import isELG

        if nocontinuum:
            log.warning('NOCONTINUUM keyword set; forcing NOCOLORCUTS=True and ADD_SNEIA=False')
            nocolorcuts = True
            self.add_SNeIa = False

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('REDSHIFT must be an NMODEL-length array')
                raise ValueError

        rand = np.random.RandomState(seed)
        emseed = rand.randint(2**32, size=nmodel)

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample.
        EM = EMSpectrum(log10wave=np.log10(self.basewave))

        # Shuffle the basis templates and then split them into ~equal chunks, so
        # we can speed up the calculations below.
        nbase = len(self.basemeta)
        chunksize = np.min((nbase, 50))
        nchunk = long(np.ceil(nbase / chunksize))

        alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
        for tempid in alltemplateid:
            rand.shuffle(tempid)
        alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

        # Assign redshift, r-magnitude, and velocity dispersion priors. 
        if redshift is None:
            redshift = rand.uniform(zrange[0], zrange[1], nmodel)

        rmag = rand.uniform(rmagrange[0], rmagrange[1], nmodel)
        if logvdisp_meansig[1] > 0:
            vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nmodel)
        else:
            vdisp = 10**np.repeat(logvdisp_meansig[0], nmodel)

        # Initialize the emission line priors with varying line-ratios and the
        # appropriate relative [OII] flux.
        oiidoublet = rand.normal(oiidoublet_meansig[0], oiidoublet_meansig[1], nmodel)
        oiihbeta, niihbeta, siihbeta, oiiihbeta = _lineratios(nmodel, EM, oiiihbrange, rand)

        d4000 = self.basemeta['D4000']
        ewoii = np.tile(10.0**(np.polyval(self.ewoiicoeff, d4000)), (nmodel, 1)) + \
          rand.normal(0.0, 0.3, (nmodel, nbase)) # rest-frame, Angstrom
        oiiflux = np.tile(self.basemeta['OII_CONTINUUM'].data, (nmodel, 1)) * ewoii 

        # Populate some of the metadata table.
        meta = _metatable(nmodel, self.objtype, self.add_SNeIa)
        for key, value in zip(('REDSHIFT', 'OIIIHBETA', 'OIIHBETA', 'NIIHBETA',
                               'SIIHBETA', 'OIIDOUBLET', 'VDISP'),
                               (redshift, oiiihbeta, oiihbeta, niihbeta,
                                siihbeta, oiidoublet, vdisp)):
            meta[key] = value
        
        # Get the (optional) distribution of SNe Ia priors.  Eventually we need
        # to make this physically consistent.
        if self.add_SNeIa:
            sne_rfluxratio = rand.uniform(sne_rfluxratiorange[0], sne_rfluxratiorange[1], nmodel)
            sne_tempid = rand.randint(0, len(self.sne_basemeta)-1, nmodel)
            meta['SNE_TEMPLATEID'] = sne_tempid
            meta['SNE_EPOCH'] = self.sne_basemeta['EPOCH'][sne_tempid]
            meta['SNE_RFLUXRATIO'] = sne_rfluxratio

        # Build the spectra.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        success = np.zeros(nmodel)
        for ii in range(nmodel):
            zwave = self.basewave.astype(float)*(1.0 + redshift[ii])

            # Get the SN spectrum and normalization factor.
            if self.add_SNeIa:
                sne_restflux = self.sne_baseflux[sne_tempid[ii], :]
                snenorm = self.rfilt.get_ab_maggies(sne_restflux, zwave)

            # Generate the emission-line spectrum for this model.
            npix = len(self.basewave)
            emflux, emwave, emline = EM.spectrum(
                linesigma=vdisp[ii],
                oiidoublet=oiidoublet[ii],
                oiiihbeta=oiiihbeta[ii],
                oiihbeta=oiihbeta[ii],
                niihbeta=niihbeta[ii],
                siihbeta=siihbeta[ii],
                oiiflux=1.0,
                seed=emseed[ii])

            for ichunk in range(nchunk):
                log.debug('Simulating {} template {}/{} in chunk {}/{}'. \
                          format(self.objtype, ii+1, nmodel, ichunk, nchunk))
                templateid = alltemplateid_chunk[ichunk][ii, :]
                nbasechunk = len(templateid)
                
                if nocontinuum:
                    restflux = np.tile(emflux, (nbasechunk, 1)) * \
                      np.tile(oiiflux[ii, templateid], (1, npix)).reshape(nbasechunk, npix)
                else:
                    restflux = self.baseflux[templateid, :] + np.tile(emflux, (nbasechunk, 1)) * \
                      np.tile(oiiflux[ii, templateid], (1, npix)).reshape(nbasechunk, npix)

                # Add in the SN spectrum.
                if self.add_SNeIa:
                    galnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                    snenorm = np.tile(galnorm['decam2014-r'].data, (1, npix)).reshape(nbasechunk, npix) * \
                      np.tile(sne_rfluxratio[ii]/snenorm['decam2014-r'].data, (nbasechunk, npix))
                    restflux += np.tile(sne_restflux, (nbasechunk, 1)) * snenorm

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = maggies[key] * 10**(-0.4*(rmag[ii]-22.5)) / maggies['decam2014-r']

                zoiiflux = oiiflux[ii, templateid] * 10**(-0.4*rmag[ii]) / np.array(maggies['decam2014-r'])

                if nocolorcuts:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = isELG(gflux=synthnano[:, 1], 
                                      rflux=synthnano[:, 2], 
                                      zflux=synthnano[:, 4])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask*(zoiiflux > minoiiflux)):
                    success[ii] = 1

                    # (@moustakas) pxs.gauss_blur_matrix is producing lots of
                    # ringing in the emission lines, so deal with it later.

                    # Convolve (just the stellar continuum) and resample.
                    #if nocontinuum is False:
                        #sigma = 1.0+self.basewave*vdisp[ii]/LIGHT
                        #flux = pxs.gauss_blur_matrix(self.pixbound,sigma) * flux
                        #flux = (flux-emflux)*pxs.gauss_blur_matrix(self.pixbound,sigma) + emflux
                        
                    this = rand.choice(np.where(colormask*(zoiiflux > minoiiflux))[0]) # Pick one randomly.
                    tempid = templateid[this]
                    outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :] * \
                                                   10**(-0.4*rmag[ii])/maggies['decam2014-r'][this])

                    meta['TEMPLATEID'][ii] = tempid
                    meta['OIIFLUX'][ii] = zoiiflux[this]
                    meta['EWOII'][ii] = ewoii[ii, tempid]
                    meta['D4000'][ii] = d4000[tempid]
                    for magkey, magindx in zip(('GMAG','RMAG','ZMAG','W1MAG','W2MAG'), (1, 2, 4, 6, 7)):
                        meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    break

        # Check to see if any spectra could not be computed.
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input redshifts (or redshift priors)!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta

class LRG(GALAXY):
    """Generate Monte Carlo spectra of luminous red galaxies (LRGs).

    """
    def __init__(self, objtype='LRG', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, add_SNeIa=False):
        """Read the LRG basis continuum templates, filter profiles and initialize the
           output wavelength array.
        
        Attributes:

        """
        super(LRG, self).__init__(objtype=objtype, minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, add_SNeIa=add_SNeIa)

    def make_templates(self, nmodel=100, zrange=(0.5,1.1), zmagrange=(19.0,20.5),
                       logvdisp_meansig=(2.3,0.1), sne_rfluxratiorange=(0.1,1.0),
                       redshift=None, seed=None, nocolorcuts=False):
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

          sne_rfluxratiorange (float, optional): r-band flux ratio of the SNeIa
            spectrum with respect to the underlying galaxy.

          redshift (float, optional): Input/output template redshifts.  Array
            size must equal NMODEL.  Overwrites ZRANGE input.
          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1
            color-cuts cuts (default False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from desispec.interpolation import resample_flux
        from desisim import pixelsplines as pxs
        from desitarget.cuts import isLRG

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('REDSHIFT must be an NMODEL-length array')
                raise ValueError
                
        rand = np.random.RandomState(seed)

        # Shuffle the basis templates and then split them into ~equal chunks, so
        # we can speed up the calculations below.
        nbase = len(self.basemeta)
        chunksize = np.min((nbase, 50))
        nchunk = long(np.ceil(nbase / chunksize))

        alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
        for tempid in alltemplateid:
            rand.shuffle(tempid)
        alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

        # Assign redshift, z-magnitude, and velocity dispersion priors. 
        if redshift is None:
            redshift = rand.uniform(zrange[0], zrange[1], nmodel)

        zmag = rand.uniform(zmagrange[0], zmagrange[1], nmodel)
        if logvdisp_meansig[1] > 0:
            vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nmodel)
        else:
            vdisp = 10**np.repeat(logvdisp_meansig[0], nmodel)

        # Populate some of the metadata table.
        meta = _metatable(nmodel, self.objtype, self.add_SNeIa)
        for key, value in zip(('REDSHIFT', 'VDISP'),
                               (redshift, vdisp)):
            meta[key] = value
        
        # Get the (optional) distribution of SNe Ia priors.  Eventually we need
        # to make this physically consistent.
        if self.add_SNeIa:
            sne_rfluxratio = rand.uniform(sne_rfluxratiorange[0], sne_rfluxratiorange[1], nmodel)
            sne_tempid = rand.randint(0, len(self.sne_basemeta)-1, nmodel)
            meta['SNE_TEMPLATEID'] = sne_tempid
            meta['SNE_EPOCH'] = self.sne_basemeta['EPOCH'][sne_tempid]
            meta['SNE_RFLUXRATIO'] = sne_rfluxratio

        # Build the spectra.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        success = np.zeros(nmodel)
        for ii in range(nmodel):
            zwave = self.basewave.astype(float)*(1.0 + redshift[ii])

            # Get the SN spectrum and normalization factor.
            if self.add_SNeIa:
                sne_restflux = self.sne_baseflux[sne_tempid[ii], :]
                snenorm = self.rfilt.get_ab_maggies(sne_restflux, zwave)

            for ichunk in range(nchunk):
                log.debug('Simulating {} template {}/{} in chunk {}/{}'. \
                          format(self.objtype, ii+1, nmodel, ichunk, nchunk))
                templateid = alltemplateid_chunk[ichunk][ii, :]
                nbasechunk = len(templateid)
                
                restflux = self.baseflux[templateid, :]

                # Add in the SN spectrum.
                if self.add_SNeIa:
                    galnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                    snenorm = np.tile(galnorm['decam2014-r'].data, (1, npix)).reshape(nbasechunk, npix) * \
                      np.tile(sne_rfluxratio[ii]/snenorm['decam2014-r'].data, (nbasechunk, npix))
                    restflux += np.tile(sne_restflux, (nbasechunk, 1)) * snenorm

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = maggies[key] * 10**(-0.4*(zmag[ii]-22.5)) / maggies['decam2014-z']

                if nocolorcuts:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = isLRG(rflux=synthnano[:, 2], 
                                      zflux=synthnano[:, 4], 
                                      w1flux=synthnano[:, 6])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask):
                    success[ii] = 1
                        
                    this = rand.choice(np.where(colormask)[0]) # Pick one randomly.
                    tempid = templateid[this]
                    outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :] * \
                                                   10**(-0.4*zmag[ii])/maggies['decam2014-z'][this])

                    meta['TEMPLATEID'][ii] = tempid
                    meta['D4000'][ii] = self.basemeta['D4000'][tempid]
                    meta['AGE'][ii] = self.basemeta['AGE'][tempid]
                    meta['ZMETAL'][ii] = self.basemeta['ZMETAL'][tempid]
                    for magkey, magindx in zip(('GMAG','RMAG','ZMAG','W1MAG','W2MAG'), (1, 2, 4, 6, 7)):
                        meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    break

        # Check to see if any spectra could not be computed.
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input redshifts (or redshift priors)!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta

class SUPERSTAR(object):
    """Base Class for generating Monte Carlo spectra of the various flavors of DESI
       stellar targets.

    """
    def __init__(self, objtype='STAR', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, colorcuts_function=None, normfilter='decam2014-r'):
        """Read the stellar basis continuum templates, filter profiles and initialize
           the output wavelength array.

        Only a linearly-spaced output wavelength array is currently supported.

        Args:
          objtype (str): type of object to simulate (default STAR)
          minwave (float, optional): minimum value of the output wavelength
            array [default 3600 Angstrom].
          maxwave (float, optional): minimum value of the output wavelength
            array [default 10000 Angstrom].
          cdelt (float, optional): spacing of the output wavelength array
            [default 2 Angstrom/pixel].
          wave (numpy.ndarray): Input/output observed-frame wavelength array,
            overriding the minwave, maxwave, and cdelt arguments [Angstrom].
          colorcuts_function (function): Function (object) to use to select
            templates that pass the color-cuts for the specified objtype
            (default None).
          normfilter (str): normalization filter name; the spectra are
            normalized to the magnitude in this bandpass (default
            'decam2014-r').

        Attributes:
          objtype (str): see Args
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            stellar continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
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
        self.zfilt = filters.load_filters('decam2014-z')

    def make_star_templates(self, nmodel=100, vrad_meansig=(0.0,200.0), magrange=(18.0, 23.5),
                            vrad=None, seed=None, nocolorcuts=False):

        """Build Monte Carlo set of spectra/templates for stars.

        This function chooses random subsets of the continuum spectra for the
        type of star specified by OBJTYPE, adds radial velocity "jitter" (or
        uses give radial velocities), then normalizes the spectrum to the
        magnitude in the given filter.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          vrad_meansig (float, optional): Mean and sigma (standard deviation) of the
            radial velocity "jitter" (in km/s) that should be added to each
            spectrum.  Defaults to a normal distribution with a mean of zero and
            sigma of 200 km/s.
          magrange (float, optional): Minimum and maximum magnitude in the
            bandpass specified by self.normfilter.  Defaults to a uniform
            distribution between (18, 23.5) in the r-band.
          vrad (float, optional): Input/output radial velocities.  Array
            size must equal NMODEL.  Overwrites VRAD_MEANSIG input.
          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial color-cuts
            (default False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:
          ValueError

        """
        from desispec.interpolation import resample_flux

        if vrad is not None:
            if len(vrad) != nmodel:
                log.fatal('VRAD must be an NMODEL-length array')
                raise ValueError

        rand = np.random.RandomState(seed)

        # Shuffle the basis templates and then split them into ~equal chunks, so
        # we can speed up the calculations below.
        nbase = len(self.basemeta)
        chunksize = np.min((nbase, 50))
        nchunk = long(np.ceil(nbase / chunksize))

        alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
        for tempid in alltemplateid:
            rand.shuffle(tempid)
        alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

        # Assign radial velocity and magnitude priors.
        mag = rand.uniform(magrange[0], magrange[1], nmodel)

        if vrad is None:
            if vrad_meansig[1] > 0:
                vrad = rand.normal(vrad_meansig[0], vrad_meansig[1], nmodel)
            else:
                vrad = np.repeat(vrad_meansig[0], nmodel)
        redshift = np.array(vrad) / LIGHT

        # Populate some of the metadata table.
        meta = _metatable(nmodel, self.objtype)
        meta['REDSHIFT'] = redshift

        # Build the spectra.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        success = np.zeros(nmodel)
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
                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = maggies[key] * 10**(-0.4*(mag[ii]-22.5)) / maggies[self.normfilter]

                if nocolorcuts or self.colorcuts_function is None:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = self.colorcuts_function(
                        gflux=synthnano[:, 1],
                        rflux=synthnano[:, 2],
                        zflux=synthnano[:, 4],
                        W1flux=synthnano[:, 6],
                        W2flux=synthnano[:, 7])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask):
                    success[ii] = 1
                        
                    this = rand.choice(np.where(colormask)[0]) # Pick one randomly.
                    tempid = templateid[this]
                    outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :] * \
                                                   10**(-0.4*mag[ii])/maggies[self.normfilter][this])

                    meta['TEMPLATEID'][ii] = tempid
                    meta['TEFF'][ii] = self.basemeta['TEFF'][tempid]
                    meta['LOGG'][ii] = self.basemeta['LOGG'][tempid]
                    if self.objtype != 'WD':
                        meta['FEH'][ii] = self.basemeta['FEH'][tempid]
                    for magkey, magindx in zip(('GMAG','RMAG','ZMAG','W1MAG','W2MAG'), (1, 2, 4, 6, 7)):
                        meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    break

        # Check to see if any spectra could not be computed.
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input radial velocities (or rv priors)!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta

class STAR(object):
    """Base Class for generating Monte Carlo spectra of the various flavors of DESI stellar targets.

    """
    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None, WD=False):
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
          objtype (str): set to STAR or WD here, subclasses can set to specific stellar types
          wave (numpy.ndarray): Output wavelength array [Angstrom].
          baseflux (numpy.ndarray): Array [nbase,npix] of the base rest-frame
            stellar continuum spectra [erg/s/cm2/A].
          basewave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX [Angstrom].
          basemeta (astropy.Table): Table of meta-data for each base template [nbase].
          decamwise (speclite.filters instance): DECam2014-* and WISE2010-* FilterSequence
          gfilt (speclite.filters instance): DECam2014 g-band FilterSequence
          rfilt (speclite.filters instance): DECam2014 r-band FilterSequence
          zfilt (speclite.filters instance): DECam2014 z-band FilterSequence

        """
        from speclite import filters
        from desisim.io import read_basis_templates

        if WD:
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
        self.zfilt = filters.load_filters('decam2014-z')

    def make_templates(self, nmodel=100, vrad_meansig=(0.0,200.0), rmagrange=(18.0,23.5),
                       gmagrange=(16.0,19.0), vrad=None, seed=None, nocolorcuts=False):

        """Build Monte Carlo set of spectra/templates for WDs or generic stars.

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
          vrad (float, optional): Input/output radial velocities.  Array
            size must equal NMODEL.  Overwrites VRAD_MEANSIG input.
          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial color-cuts
            (default False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from desispec.interpolation import resample_flux

        if vrad is not None:
            if len(vrad) != nmodel:
                log.fatal('VRAD must be an NMODEL-length array')
                raise ValueError

        rand = np.random.RandomState(seed)

        # Shuffle the basis templates and then split them into ~equal chunks, so
        # we can speed up the calculations below.
        nbase = len(self.basemeta)
        chunksize = np.min((nbase, 50))
        nchunk = long(np.ceil(nbase / chunksize))

        alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
        for tempid in alltemplateid:
            rand.shuffle(tempid)
        alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

        # Assign radial velocity and magnitude priors.
        print('NEED gmag for white dwarfs!')
        rmag = rand.uniform(rmagrange[0], rmagrange[1], nmodel)

        if vrad is None:
            if vrad_meansig[1] > 0:
                vrad = rand.normal(vrad_meansig[0], vrad_meansig[1], nmodel)
            else:
                vrad = np.repeat(vrad_meansig[0], nmodel)
        redshift = np.array(vrad) / LIGHT

        # Populate some of the metadata table.
        meta = _metatable(nmodel, self.objtype)
        meta['REDSHIFT'] = redshift

        # Build the spectra.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        success = np.zeros(nmodel)
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
                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = maggies[key] * 10**(-0.4*(rmag[ii]-22.5)) / maggies['decam2014-r']

                if nocolorcuts:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = np.repeat(1, nbasechunk)

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.any(colormask):
                    success[ii] = 1
                        
                    this = rand.choice(np.where(colormask)[0]) # Pick one randomly.
                    tempid = templateid[this]
                    outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :] * \
                                                   10**(-0.4*rmag[ii])/maggies['decam2014-r'][this])

                    meta['TEMPLATEID'][ii] = tempid
                    meta['TEFF'][ii] = self.basemeta['TEFF'][tempid]
                    meta['LOGG'][ii] = self.basemeta['LOGG'][tempid]
                    meta['FEH'][ii] = self.basemeta['FEH'][tempid]
                    for magkey, magindx in zip(('GMAG','RMAG','ZMAG','W1MAG','W2MAG'), (1, 2, 4, 6, 7)):
                        meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    break

        # Check to see if any spectra could not be computed.
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input radial velocities (or rv priors)!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta

class FSTD(SUPERSTAR):
    """Generate Monte Carlos spectra of DESI metal-poor main sequence turnoff stars (FSTD).

    """

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None):

        from desitarget.cuts import isFSTD_colors
        super(FSTD, self).__init__(objtype='FSTD', minwave=minwave, maxwave=maxwave,
                                   cdelt=cdelt, wave=wave, colorcuts_function=isFSTD_colors,
                                   normfilter='decam2014-r')

    def make_templates(self, nmodel=100, vrad_meansig=(0.0, 200.0),
                       rmagrange=(16.0, 19.0), vrad=None, seed=None):
        """Build Monte Carlo set of spectra/templates for FSTD stars.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          vrad_meansig (float, optional): Mean and sigma (standard deviation) of the
          radial velocity "jitter" (in km/s) that should be added to each
            spectrum.  Defaults to a normal distribution with a mean of zero and
            sigma of 200 km/s.
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (16, 19).
          vrad (float, optional): Input/output radial velocities.  Array
            size must equal NMODEL.  Overwrites VRAD_MEANSIG input.
          seed (long, optional): input seed for the random numbers.

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """

        outflux, wave, meta = self.make_star_templates(nmodel=nmodel, vrad_meansig=vrad_meansig,
                                                       magrange=rmagrange, vrad=vrad, seed=seed)
        return outflux, wave, meta
    
class MWS_STAR(STAR):

    """Generate Monte Carlos spectra of DESI MWS mag-selected targets.

    """

    def __init__(self, minwave=3600.0, maxwave=10000.0, cdelt=2.0, wave=None):
        super(MWS_STAR, self).__init__(minwave=minwave, maxwave=maxwave, cdelt=cdelt, wave=wave)
        self.objtype = 'MWS_STAR'

    def make_templates(self, nmodel=100, vrad_meansig=(0.0,200.0), rmagrange=(16.0,20.0),
                       seed=None):
        """Build Monte Carlo set of spectra/templates for DESI MWS Magnitude-selected Survey

        This function chooses random subsets of the continuum spectra for stars,
        adds realistic spread in radial velocity, then normalizes the spectrum to a
        specified r- or g-band magnitude.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          vrad_meansig (float, optional): Mean and sigma (standard deviation) of the
          radial velocity "jitter" (in km/s) that should be added to each
            spectrum.  Defaults to a normal distribution with a mean of zero and
            sigma of 200 km/s.
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range, here 16-20
          gmagrange (float, optional): Minimum and maximum DECam g-band (AB)
            magnitude range.
            seed (long, optional): input seed for the random numbers.

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """

        from astropy.table import Table
        from desispec.interpolation import resample_flux
        from desitarget.cuts import isMWSSTAR_colors

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
            rmag = rand.uniform(rmagrange[0], rmagrange[1], nchunk)
            vrad = rand.normal(vrad_meansig[0], vrad_meansig[1], nchunk)
            redshift = vrad/LIGHT

            # Unfortunately we have to loop here.
            for ii, iobj in enumerate(chunkindx):
                zwave = self.basewave.astype(float)*(1.0+redshift[ii])
                restflux = self.baseflux[iobj,:] # [erg/s/cm2/A @10pc]

                # Normalize to [erg/s/cm2/A, @redshift[ii]]
                rnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                norm = 10.0**(-0.4*rmag[ii])/rnorm['decam2014-r'][0]
                flux = restflux*norm

                # Convert [grzW1W2]flux to nanomaggies.
                synthmaggies = self.decamwise.get_ab_maggies(flux, zwave, mask_invalid=True)
                synthnano = np.array([ff*MAG2NANO for ff in synthmaggies[0]]) # convert to nanomaggies
                synthnano[synthnano == 0] = 10**(0.4*(22.5-99)) # if flux==0 then set mag==99 (below)

                # Color cuts on just on the standard stars.
                colormask = [isMWSSTAR_colors(gflux=synthnano[1], rflux=synthnano[2])]

                if all(colormask):
                    if ((nobj+1)%10)==0:
                        log.debug('Simulating {} template {}/{}'. \
                                format(self.objtype, nobj+1, nmodel))
                    outflux[nobj,:] = resample_flux(self.wave, zwave, flux)
    
                    meta['TEMPLATEID'][nobj] = chunkindx[ii]
                    meta['REDSHIFT'][nobj] = redshift[ii]
                    meta['GMAG'][nobj] = -2.5*np.log10(synthnano[1])+22.5
                    meta['RMAG'][nobj] = -2.5*np.log10(synthnano[2])+22.5
                    meta['ZMAG'][nobj] = -2.5*np.log10(synthnano[4])+22.5
                    meta['W1MAG'][nobj] = -2.5*np.log10(synthnano[6])+22.5
                    meta['W2MAG'][nobj] = -2.5*np.log10(synthnano[7])+22.5
                    meta['DECAM_FLUX'][nobj] = synthnano[:6]
                    meta['WISE_FLUX'][nobj] = synthnano[6:8]
                    meta['LOGG'][nobj] = self.basemeta['LOGG'][iobj]
                    meta['TEFF'][nobj] = self.basemeta['TEFF'][iobj]
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
          gilt (speclite.filters instance): DECam2014 g-band FilterSequence
          rilt (speclite.filters instance): DECam2014 r-band FilterSequence
          zilt (speclite.filters instance): DECam2014 z-band FilterSequence

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
        self.decamwise = filters.load_filters('decam2014-*', 'wise2010-W1', 'wise2010-W2')
        self.gfilt = filters.load_filters('decam2014-g')
        self.rfilt = filters.load_filters('decam2014-r')
        self.zfilt = filters.load_filters('decam2014-z')

    def make_templates(self, nmodel=100, zrange=(0.5, 4.0), rmagrange=(21.0, 23.0),
                       redshift=None, seed=None, nocolorcuts=False):
        """Build Monte Carlo set of QSO spectra/templates.

        This function generates a random set of QSO continua spectra and
        finally normalizes the spectrum to a specific g-band magnitude.

        Args:
          nmodel (int, optional): Number of models to generate (default 100).
          zrange (float, optional): Minimum and maximum redshift range.  Defaults
            to a uniform distribution between (0.5,4.0).
          rmagrange (float, optional): Minimum and maximum DECam r-band (AB)
            magnitude range.  Defaults to a uniform distribution between (21,23.0).
          redshift (float, optional): Input/output template redshifts.  Array
            size must equal NMODEL.  Overwrites ZRANGE input.
          seed (long, optional): input seed for the random numbers.
          nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
            cuts (default False).

        Returns:
          outflux (numpy.ndarray): Array [nmodel,npix] of observed-frame spectra [erg/s/cm2/A].
          wave (numpy.ndarray): Observed-frame [npix] wavelength array [Angstrom].
          meta (astropy.Table): Table of meta-data for each output spectrum [nmodel].

        Raises:

        """
        from desispec.interpolation import resample_flux
        from desisim.qso_template import desi_qso_templ as dqt
        #from desitarget.cuts import isQSO

        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('REDSHIFT must be an NMODEL-length array')
                raise ValueError
            zrange = (np.min(redshift), np.max(redshift))

        log.warning('Color-cuts not yet supported; forcing nocolorcuts=True')
        nocolorcuts = True

        rand = np.random.RandomState(seed)

        # Assign redshift and r-magnitude priors.
        if redshift is None:
            redshift = rand.uniform(zrange[0], zrange[1], nmodel)

        rmag = rand.uniform(rmagrange[0], rmagrange[1], nmodel)

        # Populate some of the metadata table.
        meta = _metatable(nmodel, self.objtype)
        meta['TEMPLATEID'] = np.arange(nmodel)
        meta['REDSHIFT'] = redshift
            
        # Build the spectra on-the-fly in chunks until enough models pass the
        # color cuts.
        nzbin = (zrange[1]-zrange[0])/self.z_wind                                                        
        N_perz = int(nmodel//nzbin + 2)                                                                  

        zwave = self.wave # [observed-frame, Angstrom]
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        success = np.zeros(nmodel)
        for ii in range(nmodel):
            log.debug('Simulating {} template {}/{}.'.format(self.objtype, ii+1, nmodel))
                      
            _, final_flux, redshifts = dqt.desi_qso_templates(
                no_write=True, rebin_wave=zwave, rstate=rand,
                N_perz=N_perz, redshift=redshift[ii])
            restflux = final_flux.T
            nmade = np.shape(restflux)[0]

            # Synthesize photometry to determine which models will pass the
            # color-cuts.  We have to temporarily pad because the spectra don't
            # go red enough.
            padflux, padzwave = self.rfilt.pad_spectrum(restflux, zwave, method='edge')
            maggies = self.decamwise.get_ab_maggies(padflux, padzwave, mask_invalid=True)

            synthnano = np.zeros((nmade, len(self.decamwise)))
            for ff, key in enumerate(maggies.columns):
                synthnano[:, ff] = maggies[key] * 10**(-0.4*(rmag[ii]-22.5)) / maggies['decam2014-r']

            if nocolorcuts:
                colormask = np.repeat(1, nmade)
            else:
                colormask = [isQSO(gflux=synthnano[1], # ToDo!
                                   rflux=synthnano[2],
                                   zflux=synthnano[4],
                                   wflux=(synthnano[6],
                                          synthnano[7]))]

            # If the color-cuts pass then populate the output flux vector
            # (suitably normalized) and metadata table and finish up.
            if np.any(colormask):
              success[ii] = 1

              this = rand.choice(np.where(colormask)[0]) # Pick one randomly.
              outflux[ii, :] = restflux[this, :]*10**(-0.4*rmag[ii])/maggies['decam2014-r'][this]

              # Temporary hack until the models go redder.
              for magkey, magindx in zip(('GMAG','RMAG','ZMAG'), (1, 2, 4)):
                  meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
              #for magkey, magindx in zip(('GMAG','RMAG','ZMAG','W1MAG','W2MAG'), (1, 2, 4, 6, 7)):
              #    meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
              meta['DECAM_FLUX'][ii] = synthnano[this, :6]
              meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

        # Check to see if any spectra could not be computed.
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input redshifts (or redshift priors)!'.\
                        format(np.sum(success == 0)))
                        
        return outflux, self.wave, meta

class BGS(GALAXY):
    """Generate Monte Carlo spectra of bright galaxy survey galaxies (BGSs).

    """
    def __init__(self, objtype='BGS', minwave=3600.0, maxwave=10000.0, cdelt=2.0,
                 wave=None, add_SNeIa=False):
        """Read the BGS basis continuum templates, filter profiles and initialize the
           output wavelength array.
        
        Attributes:
          ewhbetamog (GaussianMixtureModel): Table containing the mixture of
            Gaussian parameters encoding the empirical relationship between
            D(4000) and EW(Hbeta).

        """
        super(BGS, self).__init__(objtype=objtype, minwave=minwave, maxwave=maxwave,
                                  cdelt=cdelt, wave=wave, add_SNeIa=add_SNeIa)

        self.ewhbetacoeff = [1.28520974, -4.94408026, 4.9617704]

    def make_templates(self, nmodel=100, zrange=(0.01,0.4), rmagrange=(15.0,19.5),
                       oiiihbrange=(-1.3,0.6), oiidoublet_meansig=(0.73,0.05),
                       logvdisp_meansig=(2.0,0.17), sne_rfluxratiorange=(0.1,1.0),
                       redshift=None, seed=None, nocolorcuts=False, nocontinuum=False):
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

          sne_rfluxratiorange (float, optional): r-band flux ratio of the SNeIa
            spectrum with respect to the underlying galaxy.

          redshift (float, optional): Input/output template redshifts.  Array
            size must equal NMODEL.  Overwrites ZRANGE input.
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
        from desisim.templates import EMSpectrum
        from desispec.interpolation import resample_flux
        from desisim import pixelsplines as pxs
        from desitarget.cuts import isBGS

        if nocontinuum:
            log.warning('NOCONTINUUM keyword set; forcing NOCOLORCUTS=True and ADD_SNEIA=False')
            nocolorcuts = True
            self.add_SNeIa = False
            
        if redshift is not None:
            if len(redshift) != nmodel:
                log.fatal('REDSHIFT must be an NMODEL-length array')
                raise ValueError
                
        rand = np.random.RandomState(seed)
        emseed = rand.randint(2**32, size=nmodel)

        # Initialize the EMSpectrum object with the same wavelength array as
        # the "base" (continuum) templates so that we don't have to resample.
        EM = EMSpectrum(log10wave=np.log10(self.basewave))

        # Shuffle the basis templates and then split them into ~equal chunks, so
        # we can speed up the calculations below.
        nbase = len(self.basemeta)
        chunksize = np.min((nbase, 50))
        nchunk = long(np.ceil(nbase / chunksize))

        alltemplateid = np.tile(np.arange(nbase), (nmodel, 1))
        for tempid in alltemplateid:
            rand.shuffle(tempid)
        alltemplateid_chunk = np.array_split(alltemplateid, nchunk, axis=1)

        # Assign redshift, r-magnitude, and velocity dispersion priors. 
        if redshift is None:
            redshift = rand.uniform(zrange[0], zrange[1], nmodel)

        rmag = rand.uniform(rmagrange[0], rmagrange[1], nmodel)
        if logvdisp_meansig[1] > 0:
            vdisp = 10**rand.normal(logvdisp_meansig[0], logvdisp_meansig[1], nmodel)
        else:
            vdisp = 10**np.repeat(logvdisp_meansig[0], nmodel)

        # Initialize the emission line priors with varying line-ratios and the
        # appropriate relative H-beta flux.  Zero out emission lines for the
        # passive galaxies.
        oiidoublet = rand.normal(oiidoublet_meansig[0], oiidoublet_meansig[1], nmodel)
        oiihbeta, niihbeta, siihbeta, oiiihbeta = _lineratios(nmodel, EM, oiiihbrange, rand)

        d4000 = self.basemeta['D4000']
        ewhbeta = np.tile(10.0**(np.polyval(self.ewhbetacoeff, d4000)), (nmodel, 1)) + \
          rand.normal(0.0, 0.2, (nmodel, nbase)) # rest-frame, Angstrom
        #ewhbeta = self.ewhbetamog.sample(n_samples=(nmodel, nbase), random_state=rand)
        ewhbeta *= np.tile(self.basemeta['HBETA_LIMIT'], (nmodel, 1))
        hbetaflux = np.tile(self.basemeta['HBETA_CONTINUUM'].data, (nmodel, 1)) * ewhbeta

        # Populate some of the metadata table.
        meta = _metatable(nmodel, self.objtype, self.add_SNeIa)
        for key, value in zip(('REDSHIFT', 'OIIIHBETA', 'OIIHBETA', 'NIIHBETA',
                               'SIIHBETA', 'OIIDOUBLET', 'VDISP'),
                               (redshift, oiiihbeta, oiihbeta, niihbeta,
                                siihbeta, oiidoublet, vdisp)):
            meta[key] = value

        # Get the (optional) distribution of SNe Ia priors.  Eventually we need
        # to make this physically consistent.
        if self.add_SNeIa:
            sne_rfluxratio = rand.uniform(sne_rfluxratiorange[0], sne_rfluxratiorange[1], nmodel)
            sne_tempid = rand.randint(0, len(self.sne_basemeta)-1, nmodel)
            meta['SNE_TEMPLATEID'] = sne_tempid
            meta['SNE_EPOCH'] = self.sne_basemeta['EPOCH'][sne_tempid]
            meta['SNE_RFLUXRATIO'] = sne_rfluxratio

        # Build the spectra.
        outflux = np.zeros([nmodel, len(self.wave)]) # [erg/s/cm2/A]

        success = np.zeros(nmodel)
        for ii in range(nmodel):
            zwave = self.basewave.astype(float)*(1.0 + redshift[ii])

            # Get the SN spectrum and normalization factor.
            if self.add_SNeIa:
                sne_restflux = self.sne_baseflux[sne_tempid[ii], :]
                snenorm = self.rfilt.get_ab_maggies(sne_restflux, zwave)

            # Generate the emission-line spectrum for this model.
            npix = len(self.basewave)
            emflux, emwave, emline = EM.spectrum(
                linesigma=vdisp[ii],
                oiidoublet=oiidoublet[ii],
                oiiihbeta=oiiihbeta[ii],
                oiihbeta=oiihbeta[ii],
                niihbeta=niihbeta[ii],
                siihbeta=siihbeta[ii],
                hbetaflux=1.0,
                seed=emseed[ii])

            for ichunk in range(nchunk):
                log.debug('Simulating {} template {}/{} in chunk {}/{}'. \
                          format(self.objtype, ii+1, nmodel, ichunk, nchunk))
                templateid = alltemplateid_chunk[ichunk][ii, :]
                nbasechunk = len(templateid)
                
                if nocontinuum:
                    restflux = np.tile(emflux, (nbasechunk, 1)) * \
                      np.tile(hbetaflux[ii, templateid], (1, npix)).reshape(nbasechunk, npix)
                else:
                    restflux = self.baseflux[templateid, :] + np.tile(emflux, (nbasechunk, 1)) * \
                      np.tile(hbetaflux[ii, templateid], (1, npix)).reshape(nbasechunk, npix)

                # Add in the SN spectrum.
                if self.add_SNeIa:
                    galnorm = self.rfilt.get_ab_maggies(restflux, zwave)
                    snenorm = np.tile(galnorm['decam2014-r'].data, (1, npix)).reshape(nbasechunk, npix) * \
                      np.tile(sne_rfluxratio[ii]/snenorm['decam2014-r'].data, (nbasechunk, npix))
                    restflux += np.tile(sne_restflux, (nbasechunk, 1)) * snenorm

                # Synthesize photometry to determine which models will pass the
                # color-cuts.
                maggies = self.decamwise.get_ab_maggies(restflux, zwave, mask_invalid=True)
                synthnano = np.zeros((nbasechunk, len(self.decamwise)))
                for ff, key in enumerate(maggies.columns):
                    synthnano[:, ff] = maggies[key] * 10**(-0.4*(rmag[ii]-22.5)) / maggies['decam2014-r']

                zhbetaflux = hbetaflux[ii, templateid] * 10**(-0.4*rmag[ii]) / np.array(maggies['decam2014-r'])

                if nocolorcuts:
                    colormask = np.repeat(1, nbasechunk)
                else:
                    colormask = isBGS(rflux=synthnano[:, 2])

                # If the color-cuts pass then populate the output flux vector
                # (suitably normalized) and metadata table and finish up.
                if np.all(colormask):
                    success[ii] = 1

                    # (@moustakas) pxs.gauss_blur_matrix is producing lots of
                    # ringing in the emission lines, so deal with it later.

                    # Convolve (just the stellar continuum) and resample.
                    #if nocontinuum is False:
                        #sigma = 1.0+self.basewave*vdisp[ii]/LIGHT
                        #flux = pxs.gauss_blur_matrix(self.pixbound,sigma) * flux
                        #flux = (flux-emflux)*pxs.gauss_blur_matrix(self.pixbound,sigma) + emflux
                        
                    this = rand.choice(np.where(colormask)[0]) # Pick one randomly.
                    tempid = templateid[this]
                    outflux[ii, :] = resample_flux(self.wave, zwave, restflux[this, :] * \
                                                   10**(-0.4*rmag[ii])/maggies['decam2014-r'][this])

                    meta['TEMPLATEID'][ii] = tempid
                    meta['HBETAFLUX'][ii] = zhbetaflux[this]
                    meta['EWHBETA'][ii] = ewhbeta[ii, tempid]
                    meta['D4000'][ii] = d4000[tempid]
                    for magkey, magindx in zip(('GMAG','RMAG','ZMAG','W1MAG','W2MAG'), (1, 2, 4, 6, 7)):
                        meta[magkey][ii] = 22.5-2.5*np.log10(synthnano[this, magindx])
                    meta['DECAM_FLUX'][ii] = synthnano[this, :6]
                    meta['WISE_FLUX'][ii] = synthnano[this, 6:8]

                    break

        # Check to see if any spectra could not be computed.
        if ~np.all(success):
            log.warning('{} spectra could not be computed given the input redshifts (or redshift priors)!'.\
                        format(np.sum(success == 0)))

        return outflux, self.wave, meta
