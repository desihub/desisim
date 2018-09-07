"""
desisim.lya_spectra
===================

Function to simulate a QSO spectrum including Lyman-alpha absorption.

"""
from __future__ import division, print_function

import numpy as np
from desisim.dla import insert_dlas
from desiutil.log import get_logger

lambda_RF_LYA = 1215.67
absorber_IGM = {
    'MgI(2853)'   : { 'LRF':2852.96, 'COEF':1.e-4 },
    'MgII(2804)'  : { 'LRF':2803.5324, 'COEF':5.e-4 },
    'MgII(2796)'  : { 'LRF':2796.3511, 'COEF':9.e-4 },
    'FeII(2600)'  : { 'LRF':2600.1724835, 'COEF':1.e-4 },
    'FeII(2587)'  : { 'LRF':2586.6495659, 'COEF':1.e-4 },
    'MnII(2577)'  : { 'LRF':2576.877, 'COEF':1.e-4 },
    'FeII(2383)'  : { 'LRF':2382.7641781, 'COEF':1.e-4 },
    'FeII(2374)'  : { 'LRF':2374.4603294, 'COEF':1.e-4 },
    'FeII(2344)'  : { 'LRF':2344.2129601, 'COEF':1.e-4 },
    'AlIII(1863)' : { 'LRF':1862.79113, 'COEF':1.e-4 },
    'AlIII(1855)' : { 'LRF':1854.71829, 'COEF':1.e-4 },
    'AlII(1671)'  : { 'LRF':1670.7886, 'COEF':1.e-4 },
    'FeII(1608)'  : { 'LRF':1608.4511, 'COEF':1.e-4 },
    'CIV(1551)'   : { 'LRF':1550.77845, 'COEF':9.e-4 },
    'CIV(1548)'   : { 'LRF':1548.2049, 'COEF':1.e-3 },
    'SiII(1527)'  : { 'LRF':1526.70698, 'COEF':1.e-4 },
    'SiIV(1403)'  : { 'LRF':1402.77291, 'COEF':5.e-4 },
    'SiIV(1394)'  : { 'LRF':1393.76018, 'COEF':9.e-4 },
    'CII(1335)'   : { 'LRF':1334.5323, 'COEF':1.e-4 },
    'SiII(1304)'  : { 'LRF':1304.3702, 'COEF':1.e-4 },
    'OI(1302)'    : { 'LRF':1302.1685, 'COEF':1.e-4 },
    'SiII(1260)'  : { 'LRF':1260.4221, 'COEF':8.e-4 },
    'NV(1243)'    : { 'LRF':1242.804, 'COEF':5.e-4 },
    'NV(1239)'    : { 'LRF':1238.821, 'COEF':5.e-4 },
    'SiIII(1207)' : { 'LRF':1206.500, 'COEF':5.e-3 },
    'NI(1200)'    : { 'LRF':1200., 'COEF':1.e-3 },
    'SiII(1193)'  : { 'LRF':1193.2897, 'COEF':5.e-4 },
    'SiII(1190)'  : { 'LRF':1190.4158, 'COEF':5.e-4 },
    'OI(1039)'    : { 'LRF':1039.230, 'COEF':1.e-3 },
    'OVI(1038)'   : { 'LRF':1037.613, 'COEF':1.e-3 },
    'OVI(1032)'   : { 'LRF':1031.912, 'COEF':5.e-3 },
    'LYB'         : { 'LRF':1025.72, 'COEF':0.1901 },
    'CIII(977)'   : { 'LRF':977.020, 'COEF':5.e-3 },
    'OI(989)'     : { 'LRF':988.7, 'COEF':1.e-3 },
    'SiII(990)'   : { 'LRF':989.8731, 'COEF':1.e-3 },
}

def read_lya_skewers(lyafile,indices=None,read_dlas=False,add_metals=False) :
    '''
    Reads Lyman alpha transmission skewers (from CoLoRe, format v2.x.y)

    Args:
        lyafile: full path to input FITS filename

    Options:
        indices: indices of input file to sub-select
        read_dlas: try read DLA HDU from file
        add_metals: try to read metals HDU and multiply transmission

    Returns:
        wave[nwave]
        transmission[nlya, nwave]
        metadata[nlya]
        dlas[ndla] (if read_dlas=True, otherwise None)

    Input file must have WAVELENGTH, TRANSMISSION, and METADATA HDUs
    '''

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

    if (add_metals):
        if "METALS" in h :
            metals = h["METALS"].read()
            trans *= metals
        else :
            nom="No HDU with EXTNAME='METALS' in transmission file {}".format(lyafile)
            log.error(nom)
            raise KeyError(nom)
       
    if (read_dlas):
        if "DLA" in h:
            dlas=h["DLA"].read()
        else:
            mess="No HDU with EXTNAME='DLA' in transmission file {}".format(lyafile)
            log.error(mess)
            raise KeyError(mess)
    else: 
        dlas=None

    return wave,trans,meta,dlas

def apply_lya_transmission(qso_wave,qso_flux,trans_wave,trans) :
    '''
    Apply transmission to input flux, interpolating if needed. Note that the 
    transmission might include Lyman-beta and metal absorption, so we should 
    probably change the name of this function.

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

    if qso_wave.ndim == 2: # desisim.QSO(resample=True) returns a 2D wavelength array    
        for q in range(qso_flux.shape[0]) :
            output_flux[q, :] *= np.interp(qso_wave[q, :],trans_wave,trans[q, :],left=0,right=1)
    else:
        for q in range(qso_flux.shape[0]) :
            output_flux[q, :] *= np.interp(qso_wave,trans_wave,trans[q, :],left=0,right=1)
            
    return output_flux

def apply_metals_transmission(qso_wave,qso_flux,trans_wave,trans,metals) :
    '''
    Apply metal transmission to input flux, interpolating if needed.
    The input transmission should be only due to lya, if not has no meaning.
    This function should not be used in London mocks with version > 2.0, since 
    these have their own metal transmission already in the files, and even 
    the "TRANSMISSION" HDU includes already Lyman beta.

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

    if 'all' in metals:
        metals = [m for m in list(absorber_IGM.keys()) ]

    zPix = trans_wave*np.ones(qso_flux.shape[0])[:,None]/lambda_RF_LYA-1.

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
        * objmeta (astropy.Table): Table of additional object-specific meta-data
          [nmodel] for each output spectrum with columns defined in
          desisim.io.empty_metatable.
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

    # Hard-coded filtername!  Should match MAG_G!
    normfilt = load_filters(normfilter)

    if qso is None:
        qso = QSO(normfilter_south=normfilter, wave=wave)

    wave = qso.wave
    flux = np.zeros([nqso, len(wave)], dtype='f4')

    meta, objmeta = empty_metatable(objtype='QSO', nmodel=nqso)
    meta['TEMPLATEID'][:] = templateid
    meta['REDSHIFT'][:] = zqso
    meta['MAG'][:] = mag_g
    meta['MAGFILTER'][:] = normfilter
    meta['SEED'][:] = templateseed
    meta['RA'][:] = ra
    meta['DEC'][:] = dec

    # Lists for DLA meta data
    if add_dlas:
        dla_NHI, dla_z, dla_id = [], [], []

    # Loop on quasars
    for ii, indx in enumerate(templateid):
        flux1, _, meta1, objmeta1 = qso.make_templates(nmodel=1, redshift=np.atleast_1d(zqso[ii]), 
                                                mag=np.atleast_1d(mag_g[ii]), seed=templateseed[ii],
                                                nocolorcuts=nocolorcuts, lyaforest=False)
        flux1 *= 1e-17
        for col in meta1.colnames:
            meta[col][ii] = meta1[col][0]
        for col in objmeta1.colnames:
            objmeta[col][ii] = objmeta1[col][0]

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
            if np.min(wave/lambda_RF_LYA - 1) < zqso[ii]: # Any forest?
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
        return flux*1e17, wave, meta, objmeta, dla_meta
    else:
        return flux*1e17, wave, meta, objmeta
