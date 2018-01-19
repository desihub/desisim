"""
desisim.lya_spectra
===================

Function to simulate a QSO spectrum including Lyman-alpha absorption.

"""
from __future__ import division, print_function

import numpy as np
from desisim.dla import insert_dlas

def read_lya_skewers(lyafile,indices=None) :
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
    import fitsio
    h = fitsio.FITS(lyafile)
    wave  = h["WAVELENGTH"].read()
    trans = h["TRANSMISSION"].read().T # now shape is (nqso,nwave)
    meta  = h["METADATA"].read()
    if indices is not None :
        trans = trans[indices]
        meta=meta[:][indices]
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
