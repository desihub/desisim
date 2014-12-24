"""
Utility functions for working with simulated targets
"""

import yaml
import os
import numpy as np
import sys
import fitsio
import random
import desisim.cosmology
import desisim.interpolation
import astropy.units 
import math
import string

def sample_targets(nobj):
    """
    Return a random sampling of object types (ELG, LRG, QSO, STD, BAD_QSO)
    
    Args:
        nobj : number of objects to generate
        
    Returns:
        (true_objtype, target_objtype)
        
    where
        true_objtype   : array of what type the objects actually are
        target_objtype : array of type they were targeted as

    Notes:
    - Actual fiber assignment will result in higher relative fractions of
      LRGs and QSOs in early passes and more ELGs in later passes.
    """

    #- Load target densities
    #- TODO: what about nobs_boss (BOSS-like LRGs)?
    fx = open(os.getenv('DESIMODEL')+'/data/targets/targets.dat')
    tgt = yaml.load(fx)
    fx.close()
    ntgt = float(tgt['nobs_lrg'] + tgt['nobs_elg'] + \
                 tgt['nobs_qso'] + tgt['nobs_lya'] + tgt['ntarget_badqso'])
        
    #- Fraction of sky and standard star targets is guaranteed
    nsky = int(tgt['frac_sky'] * nobj)
    nstd = int(tgt['frac_std'] * nobj)
    
    #- Number of science fibers available
    nsci = nobj - (nsky+nstd)
    
    #- LRGs ELGs QSOs
    nlrg = np.random.poisson(nsci * tgt['nobs_lrg'] / ntgt)
    
    nqso = np.random.poisson(nsci * (tgt['nobs_qso'] + tgt['nobs_lya']) / ntgt)
    nqso_bad = np.random.poisson(nsci * (tgt['ntarget_badqso']) / ntgt)
    
    nelg = nobj - (nlrg+nqso+nqso_bad+nsky+nstd)
    
    true_objtype  = ['SKY']*nsky + ['STD']*nstd
    true_objtype += ['ELG']*nelg
    true_objtype += ['LRG']*nlrg
    true_objtype += ['QSO']*nqso + ['QSO_BAD']*nqso_bad
    assert(len(true_objtype) == nobj)
    np.random.shuffle(true_objtype)
    
    target_objtype = list()
    for x in true_objtype:
        if x == 'QSO_BAD':
            target_objtype.append('QSO')
        else:
            target_objtype.append(x)

    target_objtype = np.array(target_objtype)
    true_objtype = np.array(true_objtype)

    return true_objtype, target_objtype

#-------------------------------------------------------------------------
#- Currently unused, but keep around for now
def sample_nz(objtype, n):
    """
    Given `objtype` = 'LRG', 'ELG', 'QSO', 'STAR', 'STD'
    return array of `n` redshifts that properly sample n(z)
    from $DESIMODEL/data/targets/nz*.dat
    """
    #- TODO: should this be in desimodel instead?

    #- Stars are at redshift 0 for now.  Could consider a velocity dispersion.
    if objtype in ('STAR', 'STD'):
        return np.zeros(n, dtype=float)
        
    #- Determine which input n(z) file to use    
    targetdir = os.getenv('DESIMODEL')+'/data/targets/'
    objtype = objtype.upper()
    if objtype == 'LRG':
        infile = targetdir+'/nz_lrg.dat'
    elif objtype == 'ELG':
        infile = targetdir+'/nz_elg.dat'
    elif objtype == 'QSO':
        #- TODO: should use full dNdzdg distribution instead
        infile = targetdir+'/nz_qso.dat'
    else:
        raise ValueError("objtype {} not recognized (ELG LRG QSO STD STAR)".format(objtype))
            
    #- Read n(z) distribution
    zlo, zhi, ntarget = np.loadtxt(infile, unpack=True)[0:3]
    
    #- Construct normalized cumulative density function (cdf)
    cdf = np.cumsum(ntarget, dtype=float)
    cdf /= cdf[-1]

    #- Sample that distribution
    x = np.random.uniform(0.0, 1.0, size=n)
    return np.interp(x, cdf, zhi)

    
