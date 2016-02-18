'''
Code for quickly generating an output zcatalog given fiber assignment tiles,
a truth catalog, and optionally a previous zcatalog.

The current redshift errors and ZWARN completeness are based upon Redmonster
performance on the zdc1 training samples, documented by Govinda Dhungana at
https://desi.lbl.gov/DocDB/cgi-bin/private/ShowDocument?docid=1657

TODO:
- Include magnitudes or [OII] flux as part of parameterizing results
'''

from __future__ import absolute_import, division, print_function

import os
from collections import Counter
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column

import astropy.constants
c = astropy.constants.c.to('km/s').value

#- redshift errors and zwarn fractions from DESI-1657
#- sigmav = c sigmaz / (1+z)
_sigma_v = {
    'ELG': 19.,
    'LRG': 40.,
    'QSO': 255.,
    'STAR': 18.,
    'SKY': 9999,      #- meaningless
    'UNKNOWN': 9999,  #- meaningless
}

_zwarn_fraction = {
    'ELG': 0.14,       # 1 - 4303/5000
    'LRG': 0.015,      # 1 - 4921/5000
    'QSO': 0.18,       # 1 - 4094/5000 
    'STAR': 0.238,     # 1 - 3811/5000
    'SKY': 1.0,
    'UNKNOWN': 1.0,
}

def get_observed_redshifts(truetype, truez):
    """
    Returns observed z, zerr, zwarn arrays given true object types and redshifts
    
    Args:
        truetype : array of ELG, LRG, QSO, STAR, SKY, or UNKNOWN
        truez: array of true redshifts
        
    Returns tuple of (zout, zerr, zwarn)
    """
    zout = truez.copy()
    zerr = np.zeros(len(truez), dtype=np.float32)
    zwarn = np.zeros(len(truez), dtype=np.int32)
    for objtype in _sigma_v.keys():
        ii = (truetype == objtype)
        n = np.count_nonzero(ii)
        zerr[ii] = _sigma_v[objtype] * (1+truez[ii]) / c
        zout[ii] += np.random.normal(scale=zerr[ii])
        #- randomly select some objects to set zwarn
        num_zwarn = int(_zwarn_fraction[objtype] * n)
        if num_zwarn > 0:
            jj = np.random.choice(np.where(ii)[0], size=num_zwarn, replace=False)
            zwarn[jj] = 4
        
    return zout, zerr, zwarn    

def quickcat(tilefiles, truth, zcat=None, perfect=False):
    """
    Generates quick output zcatalog
    
    Args:
        tilefiles : list of fiberassign tile files that were observed
        truth : astropy Table of input truth with columns TARGETID, Z, and TYPE
        
    Options:
        zcat : input zcatalog Table from previous observations
        perfect : if True, treat spectro pipeline as perfect with input=output,
            otherwise add noise and zwarn!=0 flags
        
    Returns:
        zcatalog astropy Table based upon input truth, plus ZERR, ZWARN,
        NUMOBS, and TYPE converted from integer to string
        
    Notes:
        Input truth 'TYPE' is hardcoded to match dark-time integer truth
        categories from fiber assignment.
        0 : 'QSO',      #- QSO-LyA
        1 : 'QSO',      #- QSO-Tracer
        2 : 'LRG',      #- LRG
        3 : 'ELG',      #- ELG
        4 : 'STAR',     #- QSO-Fake
        5 : 'UNKNOWN',  #- LRG-Fake
        6 : 'STAR',     #- StdStar
        7 : 'SKY',      #- Sky
    """
    #- Count how many times each target was observed for this set of tiles
    ### print('Reading {} tiles'.format(len(obstiles)))
    nobs = Counter()
    for infile in tilefiles:
        fibassign = fits.getdata(infile, 'FIBER_ASSIGNMENTS')
        ii = (fibassign['TARGETID'] != -1)  #- targets with assignments
        nobs.update(fibassign['TARGETID'][ii])

    #- Count how many times each target was observed in previous zcatalog
    #- NOTE: assumes that no tiles have been repeated
    if zcat is not None:
        ### print('Counting targets from previous zobs')
        for targetid, n in zip(zcat['TARGETID'], zcat['NUMOBS']):
            nobs[targetid] += 1

    #- Trim truth down to just ones that have already been observed
    ### print('Trimming truth to just observed targets')
    obs_targetids = np.array(nobs.keys())
    iiobs = np.in1d(truth['TARGETID'], obs_targetids)
    newzcat = truth[iiobs]

    #- Add numobs column
    ### print('Adding NUMOBS column')
    nz = len(newzcat)
    newzcat.add_column(Column(name='NUMOBS', length=nz, dtype=np.int32))
    for i in range(nz):
        newzcat['NUMOBS'][i] = nobs[newzcat['TARGETID'][i]]

    #- HACK: hardcoded mapping to replace truth integer TYPE
    #- with 'ELG', 'LRG', 'QSO', etc.
    _truetype2objtype = {
        0 : 'QSO',      #- QSO-LyA
        1 : 'QSO',      #- QSO-Tracer
        2 : 'LRG',      #- LRG
        3 : 'ELG',      #- ELG
        4 : 'STAR',     #- QSO-Fake
        5 : 'UNKNOWN',  #- LRG-Fake
        6 : 'STAR',     #- StdStar
        7 : 'SKY',      #- Sky
    }

    ### print('Converting integer TYPE -> string TYPE')
    truetype = newzcat['TYPE']
    for i in set(truetype):
        assert i in _truetype2objtype.keys(), 'TYPE {} missing'.format(i)

    objtype = np.empty(nz, dtype='S7')
    for i in _truetype2objtype:
        ii = (truetype == i)
        objtype[ii] = _truetype2objtype[i]

    #- Swap integer TYPE with string TYPE
    newzcat.remove_column('TYPE')
    newzcat.add_column(Column(data=objtype, name='TYPE'))

    #- Add ZERR and ZWARN
    ### print('Adding ZERR and ZWARN')
    if not perfect:
        z, zerr, zwarn = get_observed_redshifts(objtype, newzcat['Z'])
        newzcat['Z'] = z
    else:
        zerr = np.zeros(nz, dtype=np.float32)
        zwarn = np.zeros(nz, dtype=np.int32)
        
    zerr  = Column(name='ZERR', data=zerr)
    zwarn = Column(name='ZWARN', data=zwarn)
    newzcat.add_column(zerr)
    newzcat.add_column(zwarn)

    #- Metadata for header
    newzcat.meta['EXTNAME'] = 'ZCATALOG'

    return newzcat

