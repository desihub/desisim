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

def sample_objtype(nobj):
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

def get_targets(nspec, tileid=None):
    """
    Returns:
        fibermap
        truth table
        
    TODO: document this better
    """
    if tileid is None:
        tileid = get_next_tileid()

    tile_ra, tile_dec = get_tile_radec(tileid)
    
    #- Get distribution of target types
    true_objtype, target_objtype = sample_objtype(nspec)
    
    #- Get DESI wavelength coverage
    wavemin = io.load_throughput('b').wavemin
    wavemax = io.load_throughput('z').wavemax
    dw = 0.2
    wave = np.arange(round(wavemin, 1), wavemax, dw)
    nwave = len(wave)
    
    truth = dict()
    truth['FLUX'] = np.zeros( (nspec, len(wave)) )
    truth['REDSHIFT'] = np.zeros(nspec, dtype='f4')
    truth['TEMPLATEID'] = np.zeros(nspec, dtype='i4')
    truth['O2FLUX'] = np.zeros(nspec, dtype='f4')
    truth['OBJTYPE'] = np.zeros(nspec, dtype='S10')
    #- Note: unlike other elements, first index of WAVE isn't spectrum index
    truth['WAVE'] = wave
    
    fibermap = dict()
    fibermap['MAG'] = np.zeros(nspec, dtype='f4')
    fibermap['MAGSYS'] = np.zeros(nspec, dtype='S10')
    fibermap['OBJTYPE'] = np.zeros(nspec, dtype='S10')
    
    for objtype in set(true_objtype):
        ii = np.where(true_objtype == objtype)[0]
        fibermap['OBJTYPE'][ii] = target_objtype[ii]
        truth['OBJTYPE'][ii] = true_objtype[ii]

        if objtype == 'SKY':
            continue
                    
        try:
            simflux, meta = io.read_templates(wave, objtype, len(ii))
        except ValueError, e:
            print e
            continue
            
        truth['FLUX'][ii] = simflux
        
        #- STD don't have redshift Z; others do
        if 'Z' in meta.dtype.names:
            truth['REDSHIFT'][ii] = meta['Z']
        else:
            print "No redshifts for", objtype, len(ii)

        #- Only ELGs have [OII] flux
        if objtype == 'ELG':
            truth['O2FLUX'][ii] = meta['OII_3727']
        
        #- Everyone had a templateid and some sort of magnitude    
        #- TODO: make this better!
        truth['TEMPLATEID'][ii] = meta['TEMPLATEID']
        for x in ('SDSS_R', 'DECAM_R', 'DECAM_Z'):
            if x in meta.dtype.names:
                fibermap['MAG'][ii] = meta[x]
                fibermap['MAGSYS'][ii] = x
                break
    
    #- Load fiber -> positioner mapping and tile information
    #- NOTE: multiple file I/O here; seems clumsy
    fiberpos = fits.getdata(os.getenv('DESIMODEL')+'/data/focalplane/fiberpos.fits', upper=True)
                        
    #- Fill in the rest of the fibermap structure
    fibermap['FIBER'] = np.arange(nspec, dtype='i4')
    fibermap['POSITIONER'] = fiberpos['POSITIONER'][0:nspec]
    fibermap['SPECTROID'] = fiberpos['SPECTROGRAPH'][0:nspec]
    fibermap['TARGETID'] = np.random.randint(sys.maxint, size=nspec)
    fibermap['TARGETCAT'] = np.zeros(nspec, dtype='|S20')
    fibermap['LAMBDAREF'] = np.ones(nspec, dtype=np.float32)*5400
    fibermap['TARGET_MASK0'] = np.zeros(nspec, dtype='i8')
    fibermap['RA_TARGET'] = np.ones(nspec, dtype='f8') * tile_ra   #- TODO
    fibermap['DEC_TARGET'] = np.ones(nspec, dtype='f8') * tile_dec #- TODO
    fibermap['X_TARGET'] = fiberpos['X'][0:nspec]
    fibermap['Y_TARGET'] = fiberpos['Y'][0:nspec]
    fibermap['X_FVCOBS'] = fibermap['X_TARGET']
    fibermap['Y_FVCOBS'] = fibermap['Y_TARGET']
    fibermap['X_FVCERR'] = np.zeros(nspec, dtype=np.float32)
    fibermap['Y_FVCERR'] = np.zeros(nspec, dtype=np.float32)
    fibermap['RA_OBS'] = fibermap['RA_TARGET']
    fibermap['DEC_OBS'] = fibermap['DEC_TARGET']
    
    return fibermap, truth


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

    
