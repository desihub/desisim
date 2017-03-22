"""
desisim.lya_spectra
===================

Function to simulate a QSO spectrum including Lyman-alpha absorption.
"""

from __future__ import division, print_function

def get_spectra(lyafile, nqso=None, wave=None, templateid=None, normfilter='sdss2010-g',
                seed=None, rand=None, qso=None):
    '''Generate a QSO spectrum which includes Lyman-alpha absorption.

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

    Returns:
        flux (numpy.ndarray):
            Array [nmodel, npix] of observed-frame spectra (erg/s/cm2/A).

        wave (numpy.ndarray):
            Observed-frame [npix] wavelength array (Angstrom).

        meta (astropy.Table):
            Table of meta-data [nmodel] for each output spectrum
            with columns defined in desisim.io.empty_metatable *plus* RA, DEC.

    '''
    import numpy as np
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

    for ii, indx in enumerate(templateid):
        flux1, _, meta1 = qso.make_templates(nmodel=1, redshift=np.array([zqso[ii]]),
                                             mag=np.array([mag_g[ii]]), seed=templateseed[ii])

        # read lambda and forest transmission
        data = h[indx + 1].read()
        la = data['LAMBDA'][:]
        tr = data['FLUX'][:]

        if len(tr):
            # Interpolate the transmission at the spectral wavelengths, if
            # outside the forest, the transmission is 1.
            itr = interp1d(la, tr, bounds_error=False, fill_value=1.0)
            flux1 *= itr(wave)

        padflux, padwave = normfilt.pad_spectrum(flux1, wave, method='edge')
        normmaggies = np.array(normfilt.get_ab_maggies(padflux, padwave,
                                                       mask_invalid=True)[normfilter])
        flux1 *= 10**(-0.4 * mag_g[ii]) / normmaggies
        flux[ii, :] = flux1[:]

    h.close()

    return flux, wave, meta
