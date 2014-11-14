"""
Utility functions for working with simulated targets
"""

import yaml
import os
import numpy as np

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

    
def sample_targets(nfiber):
    """
    Return tuple of arrays of true_objtype, target_objtype, z
    
    true_objtype   : array of what type the objects actually are
    target_objtype : array of type they were targeted as
    z : true object redshifts
    
    Notes:
    - Actual fiber assignment will result in higher relative fractions of
      LRGs and QSOs in early passes and more ELGs in later passes.
    - This could be expanded to include magnitude and color distributions
    """

    #- Load target densities
    #- TODO: what about nobs_boss (BOSS-like LRGs)?
    fx = open(os.getenv('DESIMODEL')+'/data/targets/targets.dat')
    tgt = yaml.load(fx)
    n = float(tgt['nobs_lrg'] + tgt['nobs_elg'] + \
              tgt['nobs_qso'] + tgt['nobs_lya'] + tgt['ntarget_badqso'])
        
    #- Fraction of sky and standard star targets is guaranteed
    nsky = int(tgt['frac_sky'] * nfiber)
    nstd = int(tgt['frac_std'] * nfiber)
    
    #- Number of science fibers available
    nsci = nfiber - (nsky+nstd)
    
    #- LRGs ELGs QSOs
    nlrg = np.random.poisson(nsci * tgt['nobs_lrg'] / n)
    
    nqso = np.random.poisson(nsci * (tgt['nobs_qso'] + tgt['nobs_lya']) / n)
    nqso_bad = np.random.poisson(nsci * (tgt['ntarget_badqso']) / n)
    
    nelg = nfiber - (nlrg+nqso+nqso_bad+nsky+nstd)
    
    true_objtype  = ['SKY']*nsky + ['STD']*nstd
    true_objtype += ['ELG']*nelg
    true_objtype += ['LRG']*nlrg
    true_objtype += ['QSO']*nqso + ['QSO_BAD']*nqso_bad
    assert(len(true_objtype) == nfiber)
    np.random.shuffle(true_objtype)
    
    target_objtype = list()
    for x in true_objtype:
        if x == 'QSO_BAD':
            target_objtype.append('QSO')
        else:
            target_objtype.append(x)

    target_objtype = np.array(target_objtype)
    true_objtype = np.array(true_objtype)

    #- Fill in z distributions; default 0 for STAR, STD, QSO_BAD
    z = np.zeros(nfiber)
    for objtype in ('ELG', 'LRG', 'QSO'):
        ii = np.where(true_objtype == objtype)[0]
        z[ii] = sample_nz(objtype, len(ii))
    
    return true_objtype, target_objtype, z
    
    
def get_templates(wave, objtype, redshift):
    """
    Return a set of template spectra
    
    Inputs:
    - wave : observer frame wavelength array in Angstroms
    - objtype : array of object types (LRG, ELG, QSO, STAR)
    - redshift : array of redshifts for these objects
    
    Returns 2D array of spectra[len(objtype), len(wave)]
    """
    
    nspec = len(redshift)

    #- Allow objtype to be a single string instead of an array of strings
    if is_instance(objtype, str):
        objtype = [objtype,] * nspec
    else:
        assert len(redshift) == len(objtype)
        
    #- Look for templates in
    #- $DESI_LRG_TEMPLATES, $DESI_ELG_TEMPLATES, etc.
    #- If those envars aren't set, default to most recent version number in
    #- $DESI_SPECTRO_TEMPLATES/{objtype}_templates/{version}/*.fits[.gz]
    
    #- Randomly subsample those templates to get nspec of them
        
    raise NotImplementedError
    