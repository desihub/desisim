from desisim.templates import QSO
from desisim.io import empty_metatable
import scipy as sp
from scipy import interpolate
from speclite import filters
import numpy as np

def get_spectra(infile, first=0, nqso=None, seed=None):
 
    '''
    read input lyman-alpha absorption files,
    return normalized spectra including forest absorption.

    Args:
        infile: name of the lyman-alpha spectra file

    Options:
        first: first spectrum to read
        nqso: number of spectra to read
        seed: random seed for template generation

    returns (flux, wave, meta)
    spectra with forest absorption normalized according to the magnitude
    
    Note:
        meta is metadata table from QSO continuum template generation
        plus RA and DEC columns
    '''

    import fitsio
    h = fitsio.FITS(infile)
    if nqso is None:
        nqso = len(h)-1

    if first<0:
        raise ValueError("first must be >= 0")

    heads = [head.read_header() for head in h[first+1:first+1+nqso]]

    zqso = [head["ZQSO"] for head in heads]
    ra = [head["RA"] for head in heads]
    dec = [head["DEC"] for head in heads]
    mag_g = [head["MAG_G"] for head in heads]

    assert(len(zqso) == nqso)

    rand = sp.random.RandomState(seed)
    seed = rand.randint(2**32, size=nqso)

    input_meta = empty_metatable(objtype='QSO', nmodel=1)

    filter_name = 'sdss2010-g'

    normfilt = filters.load_filters(filter_name)
    qso = QSO(normfilter=filter_name)

    flux = np.zeros([nqso, len(qso.wave)])
    meta = {}

    for i,head in enumerate(h[first+1:first+1+nqso]):
        input_meta["REDSHIFT"] = zqso[i]
        input_meta["MAG"] = mag_g[i]
        input_meta["SEED"] = seed[i]

        f, wave, meta_qso = qso.make_templates(input_meta=input_meta)
        ##append the meta information
        try:
            for k in meta_qso.keys():
                meta[k] = np.append(all_meta[k], meta_qso[k])
        except:
            for k in meta_qso.keys():
                meta[k] = meta_qso[k]

        ## read lambda and forest transmission
        la = head["LAMBDA"][:]
        tr = head["FLUX"][:]
        if len(tr)==0:
            continue

        ## will interpolate the transmission at the spectral wavelengths, 
        ## if outside the forest, the transmission is 1
        itr=interpolate.interp1d(la,tr,bounds_error=False,fill_value=1)
        f *= itr(wave)
        padflux, padwave = normfilt.pad_spectrum(f, wave, method='edge')
        normmaggies = sp.array(normfilt.get_ab_maggies(padflux, padwave, 
                               mask_invalid=True)[filter_name])
        f *= 10**(-0.4*input_meta['MAG'][0]) / normmaggies
        flux[i,:] = f.copy()

    h.close()

    # Add RA,DEC to output meta
    meta['RA'] = ra
    meta['DEC'] = dec
    
    return flux,wave,meta


