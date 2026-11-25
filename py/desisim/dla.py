"""
desisim.dla
===========

Functions and methods for inserting DLAs into QSO spectra.

"""
import numpy as np
from scipy.special import wofz

from importlib import resources

from astropy import constants as const
from desiutil.log import get_logger

c_cgs = const.c.to('cm/s').value

def insert_dlas(wave, zem, rstate=None, seed=None, fNHI=None, debug=False, dlaplus=False, **kwargs):
    """ Insert zero, one or more DLAs into a given spectrum towards a source
    with a given redshift
    Args:
        wave (ndarray):  wavelength array in Ang
        zem (float): quasar emission redshift
        rstate (numpy.random.rstate, optional): for random numberes
        seed (int, optional):
        fNHI (spline): f_NHI object
        dlaplus (bool): if True, include higher order Lyman lines
        **kwargs: Passed to init_fNHI()

    Returns:
        dlas (list): List of DLA dict's with keys z,N
        dla_model (ndarray): normalized specrtrum with DLAs inserted

    """
    from scipy import interpolate
    log = get_logger()
    # Init
    if rstate is None:
        rstate = np.random.RandomState(seed) # this is breaking the chain of randoms if seed is None
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
    if len(cum_lz)<2:
       log.warning('WARNING: cum_lz in insert_dla  has only {} element. skyped add DLA.'.format(len(cum_lz)))
       dlas,dla_model=[],[]
       return dlas,dla_model
      
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
        dla = dict(z=zabs, N=NHI,dlaid=jj)
        dlas.append(dla)

    # Generate model of DLAs
    dla_model = dla_spec(wave, dlas, dlaplus=dlaplus)

    # Return
    return dlas, dla_model


def dla_spec(wave, dlas, dlaplus=False):
    """ Generate spectrum absorbed by dlas
    Args:
        wave (ndarray):  observed wavelengths
        dlas (list):  DLA dicts
        dlaplus (bool): if True, include higher order Lyman lines

    Returns:
        abs_flux: ndarray of absorbed flux

    """
    
    # Lyman series (Ly1 = Lya ... Ly20) laboratory vacuum wavelengths in Angstroms,
    #   from Morton (2003, ApJS 149, 205), the same values as Appendix D of
    #   Lee, Webb, and Carswell (2020) and NIST
    lya_wave = np.array([1215.6701,
        1025.7223,
        972.5368,
        949.7431,
        937.8035,
        930.7483,
        926.2257,
        923.1504,
        920.9631,
        919.3514,
        918.1294,
        917.1806,
        916.4291,
        915.8238,
        915.3290,
        914.9192,
        914.5762,
        914.2861,
        914.0385,
        913.8256])

    # Oscillator Strengths
    #   Values from Baker (2008) NIST report
    #   These also match Appendix D of Lee, Webb, and Carswell (2020) 
    lya_f = np.array([0.41641,
        0.079142,
        0.029006,
        0.013945,
        0.0078035,
        0.0048164,
        0.0031850,
        0.0022172,
        0.0016062,
        0.0012011,
        0.00092190,
        0.00072310, 
        0.00057769, 
        0.00046886,
        0.00038577,
        0.00032124,
        0.00027035,
        0.00022967,
        0.00019677,
        0.00016987])

    # Damping constants from Lee et al. (2020)
    # These have units of s^-1
    lya_gamma = np.array([
        6.265e8,
        1.897e8,
        8.127e7,
        4.204e7,
        2.450e7,
        1.551e7,
        1.043e7,
        7.344e6,
        5.366e6,
        4.038e6,
        3.115e6,
        2.453e6,
        1.966e6,
        1.600e6,
        1.319e6,
        1.100e6,
        9.275e5,
        7.890e5,
        6.767e5,
        5.848e5]) 
    
    # Lya parameters, used with and without dlaplus: identical to the original dla_spec,
    # so that the output without dlaplus is unchanged and the Lya profile of a DLA is the
    # same with dlaplus (which then only adds absorption). 1215.67 A is the laboratory
    # value (1215.6701 A, entry 0 of lya_wave) to the precision of lambda_RF_LYA, which
    # quickquasars uses for the forest and the DLA redshifts.
    lyacm = 1215.6700 / 1e8 # cm
    flya = 0.4164
    gamma_lya = 626500000.0 # s^-1

    tau = np.zeros(wave.size)
    wavecm = wave / 1e8
    b_value = 30*1e5 # cm/s
    # Lyman limit (series limit of the lines above), Morton (2003)
    lyman_limit = 911.7536 # Angstrom
    # HI photoionization cross-section at the Lyman limit (e.g. Osterbrock & Ferland 2006);
    # below the limit it falls as sigma(lambda) = sigma_0 (lambda / lambda_LL)^3
    sigma_0_cont = 6.304e-18  # cm^2

    for dla in dlas:
        # Cache DLA properties to avoid repeated dictionary lookups
        dla_z = dla['z']
        dla_N = dla['N']

        # Lya, always included (with dlaplus too), because its damping wings can reach
        # the spectrum even when the line centre is outside it
        par = [dla_N,
               dla_z,
               b_value,
               lyacm,
               flya,
               gamma_lya]
        tau += voigt_tau(wavecm, par)

        if dlaplus: # add higher order Lyman series
            # The higher-order lines (Lyb to Ly20) are only computed when their centre
            # is in the wavelength range.
            relevant_lines = []
            for n in range(1, lya_wave.size):
                line_wave_obs = lya_wave[n] * (1 + dla_z)  # observed wavelength
                if line_wave_obs >= wave[0] and line_wave_obs <= wave[-1]:
                    relevant_lines.append(n)

            for n in relevant_lines:
                par = [dla_N, dla_z, b_value, lya_wave[n] / 1e8, lya_f[n], lya_gamma[n]]
                tau += voigt_tau(wavecm, par)

            # Lyman continuum (bound-free) absorption, tau = sigma_0 N (lambda / lambda_LL)^3,
            # below the Lyman limit.
            # The lines above the last tabulated one (upper level n = 22, 23, ...,
            # 913.64 A to the limit) are replaced by a flat quasi-continuum tau = sigma_0 N
            # between the Lyman limit and the midpoint between the last tabulated line and
            # the next one. The mean oscillator-strength density of the series,
            # df/dlambda ~ 0.8 / lambda_LL for large n (f_n ~ 1.6 n^-3, line spacing
            # 2 lambda_LL n^-3), corresponds to sigma ~ 6.4e-18 cm^2, i.e. it joins on to
            # the photoionization cross-section at the limit. The lines blend into a
            # continuum for n >~ 28, where the spacing is smaller than the Doppler width;
            # between n = 22 and 28 the gaps between the line cores (~0.1-0.2 A) are not
            # resolved by DESI.
            n_next = lya_wave.size + 2 # upper level of the first line not tabulated
            wave_next = lyman_limit / (1. - 1. / n_next**2)
            wave_cut_obs = 0.5 * (lya_wave[-1] + wave_next) * (1 + dla_z)
            wave_ll_obs = lyman_limit * (1 + dla_z)
            if wave[0] < wave_cut_obs:
                below = wave < wave_cut_obs
                tau[below] += sigma_0_cont * 10**dla_N * np.minimum(wave[below] / wave_ll_obs, 1.)**3
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
    fN_file = str(resources.files('desisim').joinpath('data', 'fN_spline_z24.fits.gz'))
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
