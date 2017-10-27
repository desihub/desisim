"""
desisim.lya_spectra
===================

Function to simulate a QSO spectrum including Lyman-alpha absorption.
"""
from __future__ import division, print_function

import numpy as np
from scipy.special import wofz

from astropy import constants as const

from pkg_resources import resource_filename

c_cgs = const.c.to('cm/s').value


def get_spectra(lyafile, nqso=None, wave=None, templateid=None, normfilter='sdss2010-g',
                seed=None, rand=None, qso=None, add_dlas=False, debug=False, nocolorcuts=False):
    """Generate a QSO spectrum which includes Lyman-alpha absorption.

    Args:
        lyafile (str): name of the Lyman-alpha spectrum file to read.
        nqso (int, optional): number of spectra to generate (starting from the
          first spectrum; if more flexibility is needed use TEMPLATEID).
        wave (numpy.ndarray, optional): desired output wavelength vector.
        templateid (int numpy.ndarray, optional): indices of the spectra
          (0-indexed) to read from LYAFILE (default is to read everything).  If
          provided together with NQSO, TEMPLATEID wins.
        normfilter (str, optional): normalization filter
        seed (int, optional): Seed for random number generator.
        rand (numpy.RandomState, optional): RandomState object used for the
          random number generation.  If provided together with SEED, this
          optional input superseeds the numpy.RandomState object instantiated by
          SEED.
        qso (desisim.templates.QSO, optional): object with which to generate
          individual spectra/templates.
        add_dlas (bool): Inject damped Lya systems into the Lya forest
          These are done according to the current best estimates for the incidence dN/dz
          (Prochaska et al. 2008, ApJ, 675, 1002)
          Set in calc_lz
          These are *not* inserted according to overdensity along the sightline
        nocolorcuts (bool, optional): Do not apply the fiducial rzW1W2 color-cuts
          cuts (default False).

    Returns:
        flux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra
          (erg/s/cm2/A).
        wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
        meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum
          with columns defined in desisim.io.empty_metatable *plus* RA, DEC.
        dla_meta (astropy.Table): Table of meta-data [ndla] for the DLAs injected
          into the spectra.  Only returned if add_dlas=True

    """
    from scipy.interpolate import interp1d

    import fitsio

    from speclite.filters import load_filters
    from desisim.templates import QSO
    from desisim.io import empty_metatable

    h = fitsio.FITS(lyafile)
    if templateid is None:
        if nqso is None:
            nqso = len(h)-1
        templateid = np.arange(nqso)
    else:
        templateid = np.asarray(templateid)
        nqso = len(templateid)

    if rand is None:
        rand = np.random.RandomState(seed)
    templateseed = rand.randint(2**32, size=nqso)

    #heads = [head.read_header() for head in h[templateid + 1]]
    heads = []
    for indx in templateid:
        heads.append(h[indx + 1].read_header())

    zqso = np.array([head['ZQSO'] for head in heads])
    ra = np.array([head['RA'] for head in heads])
    dec = np.array([head['DEC'] for head in heads])
    mag_g = np.array([head['MAG_G'] for head in heads])

    # Hard-coded filtername!
    normfilt = load_filters(normfilter)

    if qso is None:
        qso = QSO(normfilter=normfilter, wave=wave)

    wave = qso.wave
    flux = np.zeros([nqso, len(wave)], dtype='f4')

    meta = empty_metatable(objtype='QSO', nmodel=nqso)
    meta['TEMPLATEID'] = templateid
    meta['REDSHIFT'] = zqso
    meta['MAG'] = mag_g
    meta['SEED'] = templateseed
    meta['RA'] = ra
    meta['DEC'] = dec


    # Lists for DLA meta data
    if add_dlas:
        dla_NHI, dla_z, dla_id = [], [], []

    # Loop on quasars
    for ii, indx in enumerate(templateid):
        flux1, _, meta1 = qso.make_templates(nmodel=1, redshift=np.atleast_1d(zqso[ii]), 
                                             mag=np.atleast_1d(mag_g[ii]), seed=templateseed[ii],
                                             nocolorcuts=nocolorcuts)
        flux1 *= 1e-17
        for col in meta1.colnames:
            meta[col][ii] = meta1[col][0]

        # read lambda and forest transmission
        data = h[indx + 1].read()
        la = data['LAMBDA'][:]
        tr = data['FLUX'][:]

        if len(tr):
            # Interpolate the transmission at the spectral wavelengths, if
            # outside the forest, the transmission is 1.
            itr = interp1d(la, tr, bounds_error=False, fill_value=1.0)
            flux1 *= itr(wave)

        # Inject a DLA?
        if add_dlas:
            if np.min(wave/1215.67 - 1) < zqso[ii]: # Any forest?
                dlas, dla_model = insert_dlas(wave, zqso[ii], seed=templateseed[ii])
                ndla = len(dlas)
                if ndla > 0:
                    flux1 *= dla_model
                    # Meta
                    dla_z += [idla['z'] for idla in dlas]
                    dla_NHI += [idla['N'] for idla in dlas]
                    dla_id += [indx]*ndla

        padflux, padwave = normfilt.pad_spectrum(flux1, wave, method='edge')
        normmaggies = np.array(normfilt.get_ab_maggies(padflux, padwave,
                                                       mask_invalid=True)[normfilter])

        factor = 10**(-0.4 * mag_g[ii]) / normmaggies
        flux1 *= factor
        for key in ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
            meta[key][ii] *= factor
        flux[ii, :] = flux1[:]

    h.close()

    # Finish
    if add_dlas:
        ndla = len(dla_id)
        if ndla > 0:
            from astropy.table import Table
            dla_meta = Table()
            dla_meta['NHI'] = dla_NHI  # log NHI values
            dla_meta['z'] = dla_z
            dla_meta['ID'] = dla_id
        else:
            dla_meta = None
        return flux*1e17, wave, meta, dla_meta
    else:
        return flux*1e17, wave, meta


def insert_dlas(wave, zem, rstate=None, seed=None, fNHI=None, debug=False, **kwargs):
    """ Insert zero, one or more DLAs into a given spectrum towards a source
    with a given redshift
    Args:
        wave (ndarray):  wavelength array in Ang
        zem (float): quasar emission redshift
        rstate (numpy.random.rstate, optional): for random numberes
        seed (int, optional):
        fNHI (spline): f_NHI object
        **kwargs: Passed to init_fNHI()

    Returns:
        dlas (list): List of DLA dict's with keys z,N
        dla_model (ndarray): normalized specrtrum with DLAs inserted

    """
    from scipy import interpolate

    # Init
    if rstate is None:
        rstate = np.random.RandomState(seed)
    if fNHI is None:
        fNHI = init_fNHI(**kwargs)

    # Allowed redshift placement
    ## Cut on zem and 910A rest-frame
    zlya = wave/1215.67 - 1
    dz = np.roll(zlya,-1)-zlya
    dz[-1] = dz[-2]
    gdz = (zlya < zem) & (wave > 910.*(1+zem))

    # l(z) -- Uses DLA for SLLS too which is fine
    lz = calc_lz(zlya[gdz])
    cum_lz = np.cumsum(lz*dz[gdz])
    tot_lz = cum_lz[-1]
    fzdla = interpolate.interp1d(cum_lz/tot_lz, zlya[gdz],
                                 bounds_error=False,fill_value=np.min(zlya[gdz]))#

    # n DLA
    nDLA = rstate.poisson(tot_lz, 1)

    # Generate DLAs
    dlas = []
    for jj in range(nDLA[0]):
        # Random z
        zabs = float(fzdla(rstate.random_sample()))
        # Random NHI
        NHI = float(fNHI(rstate.random_sample()))
        # Generate and append
        dla = dict(z=zabs, N=NHI)
        dlas.append(dla)

    # Generate model of DLAs
    dla_model = dla_spec(wave, dlas)

    # Return
    return dlas, dla_model


def dla_spec(wave, dlas):
    """ Generate spectrum absorbed by dlas
    Args:
        wave (ndarray):  observed wavelengths
        dlas (list):  DLA dicts

    Returns:
        abs_flux: ndarray of absorbed flux

    """
    flya = 0.4164
    gamma_lya = 626500000.0
    lyacm = 1215.6700 / 1e8
    wavecm = wave / 1e8
    tau = np.zeros(wave.size)
    for dla in dlas:
        par = [dla['N'],
               dla['z'],
               30*1e5,  # b value
               lyacm,
               flya,
               gamma_lya]
        tau += voigt_tau(wavecm, par)
    # Flux
    flux = np.exp(-1.0*tau)
    # Return
    return flux


def voigt_tau(wave, par):
    """ Find the optical depth at input wavelengths
    Taken from linetools.analysis.voigt

    This is a stripped down routine for calculating a tau array for an
    input line. Built for speed, not utility nor with much error
    checking.  Use wisely.  And take careful note of the expected
    units of the inputs (cgs)

    Parameters
    ----------
    wave : ndarray
      Assumed to be in cm
    parm : list
      Line parameters.  All are input unitless and should be in cgs
        par[0] = logN (cm^-2)
        par[1] = z
        par[2] = b in cm/s
        par[3] = wrest in cm
        par[4] = f value
        par[5] = gamma (s^-1)

    Returns
    -------
    tau : ndarray
      Optical depth at input wavelengths
    """
    cold = 10.0 ** par[0]  # / u.cm / u.cm
    zp1 = par[1] + 1.0
    # wv=line.wrest.to(u.cm) #*1.0e-8
    nujk = c_cgs / par[3]
    dnu = par[2] / par[3]  # (line.attrib['b'].to(u.km/u.s) / wv).to('Hz')
    avoigt = par[5] / (4 * np.pi * dnu)
    uvoigt = ((c_cgs / (wave / zp1)) - nujk) / dnu
    # Voigt
    cne = 0.014971475 * cold * par[4]  # line.data['f'] * u.cm * u.cm * u.Hz
    tau = cne * voigt_wofz(uvoigt, avoigt) / dnu
    #
    return tau


def voigt_wofz(vin,a):
    """Uses scipy function for calculation.
    Taken from linetools.analysis.voigt

    Parameters
    ----------
    vin : ndarray
      u parameter
    a : float
      a parameter

    Returns
    -------
    voigt : ndarray
    """
    return wofz(vin + 1j * a).real


def init_fNHI(slls=False, mix=True):
    """
    Args:
        slls (bool): SLLS only?
        mix (bool):  Mix of DLAs and SLLS?

    Returns:
        model: fNHI model

    """
    from astropy.io import fits
    from scipy import interpolate as scii
    # f(N)
    fN_file = resource_filename('desisim','data/fN_spline_z24.fits.gz')
    hdu = fits.open(fN_file)
    fN_data = hdu[1].data
    # Instantiate
    pivots=np.array(fN_data['LGN']).flatten()
    param = dict(sply=np.array(fN_data['FN']).flatten())
    fNHI_model = scii.PchipInterpolator(pivots, param['sply'], extrapolate=True)  # scipy 0.16

    # Integrate on NHI
    if slls:
        lX, cum_lX, lX_NHI = calculate_lox(fNHI_model, 19.5, NHI_max=20.3, cumul=True)
    elif mix:
        lX, cum_lX, lX_NHI = calculate_lox(fNHI_model, 19.5, NHI_max=22.5, cumul=True)
    else:
        lX, cum_lX, lX_NHI = calculate_lox(fNHI_model, 20.3, NHI_max=22.5, cumul=True)
    # Interpolator
    cum_lX /= cum_lX[-1] # Normalize
    fNHI = scii.interp1d(cum_lX, lX_NHI, bounds_error=False,fill_value=lX_NHI[0])
    # Return
    return fNHI


def calc_lz(z, boost=1.6):
    """
    Args:
        z (ndarray): redshift values for evaluation
        boost (float): boost for SLLS (should be 1 if only running DLAs)

    Returns:
        ndarray:  l(z) aka dN/dz values of DLAs

    """
    lz = boost * 0.6 * np.exp(-7./z**2)  # Prochaska et al. 2008, ApJ, 675, 1002
    return lz


def evaluate_fN(model, NHI):
    """ Evaluate an f(N,X) model at a set of NHI values

    Parameters
    ----------
    NHI : array
      log NHI values


    Returns
    -------
    log_fN : array
      f(NHI,X) values

    """
    # Evaluate without z dependence
    log_fNX = model.__call__(NHI)

    return log_fNX


def calculate_lox(model, NHI_min, NHI_max=None, neval=10000, cumul=False):
    """ Calculate l(X) over an N_HI interval

    Parameters
    ----------
    z : float
      Redshift for evaluation
    NHI_min : float
      minimum log NHI value
    NHI_max : float, optional
      maximum log NHI value for evaluation (Infinity)
    neval : int, optional
      Discretization parameter (10000)
    cumul : bool, optional
      Return a cumulative array? (False)
    cosmo : astropy.cosmology, optional
      Cosmological model to adopt (as needed)

    Returns
    -------
    lX : float
      l(X) value
    """
    # Initial
    if NHI_max is None:
        NHI_max = 23.
    # Brute force (should be good to ~0.5%)
    lgNHI = np.linspace(NHI_min,NHI_max,neval)  #NHI_min + (NHI_max-NHI_min)*np.arange(neval)/(neval-1.)
    dlgN = lgNHI[1]-lgNHI[0]
    # Evaluate f(N,X)
    lgfNX = evaluate_fN(model, lgNHI)
    # Sum
    lX = np.sum(10.**(lgfNX+lgNHI)) * dlgN * np.log(10.)
    if cumul is True:
        cum_sum = np.cumsum(10.**(lgfNX+lgNHI)) * dlgN * np.log(10.)
    # Return
    return lX, cum_sum, lgNHI
