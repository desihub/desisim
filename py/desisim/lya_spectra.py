"""
desisim.lya_spectra
===================

Function to simulate a QSO spectrum including Lyman-alpha absorption.

"""
from __future__ import division, print_function

import numpy as np
from desisim.dla import insert_dlas
from desiutil.log import get_logger

absorber_IGM = {
    'Halpha'      : { 'LRF':6562.8, 'COEF':None },
    'Hbeta'       : { 'LRF':4862.68, 'COEF':None },
    'MgI(2853)'   : { 'LRF':2852.96, 'COEF':None },
    'MgII(2804)'  : { 'LRF':2803.5324, 'COEF':None },
    'MgII(2796)'  : { 'LRF':2796.3511, 'COEF':None },
    'FeII(2600)'  : { 'LRF':2600.1724835, 'COEF':None },
    'FeII(2587)'  : { 'LRF':2586.6495659, 'COEF':None },
    'MnII(2577)'  : { 'LRF':2576.877, 'COEF':None },
    'FeII(2383)'  : { 'LRF':2382.7641781, 'COEF':None },
    'FeII(2374)'  : { 'LRF':2374.4603294, 'COEF':None },
    'FeII(2344)'  : { 'LRF':2344.2129601, 'COEF':None },
    'AlIII(1863)' : { 'LRF':1862.79113, 'COEF':None },
    'AlIII(1855)' : { 'LRF':1854.71829, 'COEF':None },
    'AlII(1671)'  : { 'LRF':1670.7886, 'COEF':None },
    'FeII(1608)'  : { 'LRF':1608.4511, 'COEF':None },
    'CIV(1551)'   : { 'LRF':1550.77845, 'COEF':None },
    'CIV(eff)'    : { 'LRF':1549.06, 'COEF':None },
    'CIV(1548)'   : { 'LRF':1548.2049, 'COEF':None },
    'SiII(1527)'  : { 'LRF':1526.70698, 'COEF':None },
    'SiIV(1403)'  : { 'LRF':1402.77291, 'COEF':None },
    'SiIV(1394)'  : { 'LRF':1393.76018, 'COEF':None },
    'CII(1335)'   : { 'LRF':1334.5323, 'COEF':None },
    'SiII(1304)'  : { 'LRF':1304.3702, 'COEF':None },
    'OI(1302)'    : { 'LRF':1302.1685, 'COEF':None },
    'SiII(1260)'  : { 'LRF':1260.4221, 'COEF':0.002 },
    'NV(1243)'    : { 'LRF':1242.804, 'COEF':None },
    'NV(1239)'    : { 'LRF':1238.821, 'COEF':None },
    'LYA'         : { 'LRF':1215.67, 'COEF':1. },
    'SiIII(1207)' : { 'LRF':1206.500, 'COEF':0.005 },
    'NI(1200)'    : { 'LRF':1200., 'COEF':None },
    'SiII(1193)'  : { 'LRF':1193.2897, 'COEF':0.002 },
    'SiII(1190)'  : { 'LRF':1190.4158, 'COEF':0.002 },
    'OI(1039)'    : { 'LRF':1039.230, 'COEF':None },
    'OVI(1038)'   : { 'LRF':1037.613, 'COEF':None },
    'OVI(1032)'   : { 'LRF':1031.912, 'COEF':None },
    'LYB'         : { 'LRF':1025.72, 'COEF':None },
}

def read_lya_skewers(lyafile,indices=None,dla_=False) :
    '''
    Reads Lyman alpha transmission skewers (from CoLoRe, format v2.x.y)

    Args:
        lyafile: full path to input FITS filename

    Options:
        indices: indices of input file to sub-select

    Returns (wave[nwave], transmission[nlya, nwave], metadata[nlya])

    Input file must have WAVELENGTH, TRANSMISSION, and METADATA HDUs
    '''


    # this is the new format set up by Andreu
    log = get_logger()

    import fitsio
    h = fitsio.FITS(lyafile)
    if "WAVELENGTH" in h :
        wave  = h["WAVELENGTH"].read()
    else :
        log.warning("I assume WAVELENGTH is HDU 2")
        wave  = h[2].read()
    
    if "TRANSMISSION" in h :
        trans = h["TRANSMISSION"].read()
    else :
        log.warning("I assume TRANSMISSION is HDU 3")
        trans = h[3].read()

    if trans.shape[1] != wave.size :
        if trans.shape[0] == wave.size :
            trans = trans.T  # now shape is (nqso,nwave)
        else :
            log.error("shape of wavelength={} and transmission={} don't match".format(wave.shape,trans.shape))
            raise ValueError("shape of wavelength={} and transmission={} don't match".format(wave.shape,trans.shape))

    if "METADATA" in h :
        meta  = h["METADATA"].read()
    else :
        log.warning("I assume METADATA is HDU 1")
        meta = h[1].read()

    if indices is not None :
        trans = trans[indices]
        meta=meta[:][indices]

##ALMA
    if (dla_):
        if "DLA" in h:
           dla_=h["DLA"].read()
        else:
           log.warning("I assume TRANSMISSION is HDU 3")
           dla_=h[3].read(i)
        return wave,trans,meta,dla_
##ALMA
    return wave,trans,meta

def apply_lya_transmission(qso_wave,qso_flux,trans_wave,trans) :
    '''
    Apply transmission to input flux, interpolating if needed

    Args:
        qso_wave: 1D[nwave] array of QSO wavelengths
        qso_flux: 2D[nqso, nwave] array of fluxes
        trans_wave: 1D[ntranswave ] array of transmission wavelength samples
        trans: 2D[nqso, ntranswave] transmissions [0-1]

    Returns:
        output_flux[nqso, nwave]

    This routine simply apply the transmission
    the only thing besides multiplication is a wavelength interpolation
    of transmission to the QSO wavelength grid
    '''
    if qso_flux.shape[0] != trans.shape[0] :
        raise(ValueError("not same number of qso {} {}".format(qso_flux.shape[0],trans.shape[0])))
    
    output_flux = qso_flux.copy()
    for q in range(qso_flux.shape[0]) :
        output_flux[q, :] *= np.interp(qso_wave,trans_wave,trans[q, :],left=0,right=1)
    return output_flux
def apply_metals_transmission(qso_wave,qso_flux,trans_wave,trans,metals) :
    '''
    Apply metal transmission to input flux, interpolating if needed.
    The input transmission should be only due to lya, if not has no meaning

    Args:
        qso_wave: 1D[nwave] array of QSO wavelengths
        qso_flux: 2D[nqso, nwave] array of fluxes
        trans_wave: 1D[ntranswave ] array of lya transmission wavelength samples
        trans: 2D[nqso, ntranswave] transmissions [0-1]
        metals: list of metal names to use

    Returns:
        output_flux[nqso, nwave]

    '''
    if qso_flux.shape[0] != trans.shape[0] :
        raise(ValueError("not same number of qso {} {}".format(qso_flux.shape[0],trans.shape[0])))

    zPix = trans_wave*np.ones(qso_flux.shape[0])[:,None]/absorber_IGM['LYA']['LRF']-1.

    tau = np.zeros(zPix.shape)
    w = trans>1.e-100
    tau[w] = -np.log(trans[w])
    tau[~w] = -np.log(1.e-100)

    try:
        mtrans = { m:np.exp(-absorber_IGM[m]['COEF']*tau) for m in metals }
        mtrans_wave = { m:(zPix+1.)*absorber_IGM[m]['LRF'] for m in metals }
    except KeyError as e:
        lstMetals = ''
        nolstMetals = ''
        for m in absorber_IGM.keys():
            lstMetals += m+', '
        for m in np.array(metals)[~np.in1d(metals,[mm for mm in absorber_IGM.keys()])]:
            nolstMetals += m+', '
        raise Exception("Input metals {} are not in the list, available metals are {}".format(nolstMetals[:-2],lstMetals[:-2])) from e
    except TypeError as e:
        lstMetals = ''
        for m in [ m for m in metals if absorber_IGM[m]['COEF'] is None ]:
            lstMetals += m+', '
        raise Exception("Input metals {} have no values for COEF".format(lstMetals[:-2])) from e

    output_flux = qso_flux.copy()
    for q in range(qso_flux.shape[0]):
        for m in metals:
            output_flux[q,:] *= np.interp(qso_wave,mtrans_wave[m][q,:],mtrans[m][q,:],left=1.,right=1.)

    return output_flux

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

    Returns (flux, wave, meta, dla_meta) where:

        * flux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra
          (erg/s/cm2/A).
        * wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
        * meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum
          with columns defined in desisim.io.empty_metatable *plus* RA, DEC.
        * dla_meta (astropy.Table): Table of meta-data [ndla] for the DLAs injected
          into the spectra.  Only returned if add_dlas=True

    Note: `dla_meta` is only included if add_dlas=True.

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
        nqso = len(np.atleast_1d(templateid))

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
                                             nocolorcuts=nocolorcuts, lyaforest=False)
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
