"""
Utility functions for working with simulated targets
"""

import os
import sys

import numpy as np
import yaml

from desimodel.focalplane import FocalPlane
import desimodel.io

from desispec import brick
from desispec.io.fibermap import empty_fibermap

from desisim import io

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
    - Also ensures at least 2 sky and 1 stdstar, even if nobj is small
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
    
    #- Assure at least 2 sky and 1 std
    if nobj >= 3:
        if nstd < 1:
            nstd = 1
        if nsky < 2:
            nsky = 2
    
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
        tile_ra, tile_dec = 0.0, 0.0
    else:
        tile_ra, tile_dec = io.get_tile_radec(tileid)
    
    #- Get distribution of target types
    true_objtype, target_objtype = sample_objtype(nspec)
    
    #- Get DESI wavelength coverage
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
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
    
    fibermap = empty_fibermap(nspec)
    
    for objtype in set(true_objtype):
        ii = np.where(true_objtype == objtype)[0]
        fibermap['OBJTYPE'][ii] = target_objtype[ii]
        truth['OBJTYPE'][ii] = true_objtype[ii]

        if objtype == 'SKY':
            continue
                    
        try:
            simflux, meta = io.read_templates(wave, objtype, len(ii))
        except ValueError, err:
            print err
            continue
            
        truth['FLUX'][ii] = simflux
        
        #- STD don't have redshift Z; others do
        #- In principle we should also have redshifts (radial velocities)
        #- for standards as well.
        if 'Z' in meta.dtype.names:
            truth['REDSHIFT'][ii] = meta['Z']
        elif objtype != 'STD':
            print "No redshifts for", objtype, len(ii)

        #- Only ELGs have [OII] flux
        if objtype == 'ELG':
            truth['O2FLUX'][ii] = meta['OII_3727']
        
        #- Everyone had a templateid
        truth['TEMPLATEID'][ii] = meta['TEMPLATEID']
        
        #- Extract magnitudes from colors
        #- TODO: make this more consistent at the input level
        
        #- Standard Stars have SDSS magnitudes
        if objtype == 'STD':
            magr = meta['SDSS_R']
            magi = magr - meta['SDSS_RI']
            magz = magi - meta['SDSS_IZ']
            magg = magr - meta['SDSS_GR']  #- R-G, not G-R ?
            magu = magg - meta['SDSS_UG']

            mags = np.vstack( [magu, magg, magr, magi, magz] ).T
            filters = ['SDSS_U', 'SDSS_G', 'SDSS_R', 'SDSS_I', 'SDSS_Z']

            fibermap['MAG'][ii] = mags
            fibermap['FILTER'][ii] = filters
        #- LRGs
        elif objtype == 'LRG':
            magz = meta['DECAM_Z']
            magr = magz - meta['DECAM_RZ']
            magw = magr - meta['DECAM_RW1']
            
            fibermap['MAG'][ii, 0:3] = np.vstack( [magr, magz, magw] ).T
            fibermap['FILTER'][ii, 0:3] = ['DECAM_R', 'DECAM_Z', 'WISE_W1']
            
        #- ELGs
        elif objtype == 'ELG':
            magr = meta['DECAM_R']
            magg = magr - meta['DECAM_GR']
            magz = magr - meta['DECAM_RZ']
            fibermap['MAG'][ii, 0:3] = np.vstack( [magg, magr, magz] ).T
            fibermap['FILTER'][ii, 0:3] = ['DECAM_G', 'DECAM_R', 'DECAM_Z']
        
        elif objtype == 'QSO':
            #- QSO templates don't have magnitudes yet
            pass
                
    #- Load fiber -> positioner mapping and tile information
    fiberpos = desimodel.io.load_fiberpos()

    #- Where are these targets?  Centered on positioners for now.
    x = fiberpos['X'][0:nspec]
    y = fiberpos['Y'][0:nspec]
    fp = FocalPlane(tile_ra, tile_dec)
    ra = np.zeros(nspec)
    dec = np.zeros(nspec)
    for i in range(nspec):
        ra[i], dec[i] = fp.xy2radec(x[i], y[i])
    
    #- Fill in the rest of the fibermap structure
    fibermap['FIBER'] = np.arange(nspec, dtype='i4')
    fibermap['POSITIONER'] = fiberpos['POSITIONER'][0:nspec]
    fibermap['SPECTROID'] = fiberpos['SPECTROGRAPH'][0:nspec]
    fibermap['TARGETID'] = np.random.randint(sys.maxint, size=nspec)
    fibermap['TARGETCAT'] = np.zeros(nspec, dtype='|S20')
    fibermap['LAMBDAREF'] = np.ones(nspec, dtype=np.float32)*5400
    fibermap['TARGET_MASK0'] = np.zeros(nspec, dtype='i8')
    fibermap['RA_TARGET'] = ra
    fibermap['DEC_TARGET'] = dec
    fibermap['X_TARGET'] = x
    fibermap['Y_TARGET'] = y
    fibermap['X_FVCOBS'] = fibermap['X_TARGET']
    fibermap['Y_FVCOBS'] = fibermap['Y_TARGET']
    fibermap['X_FVCERR'] = np.zeros(nspec, dtype=np.float32)
    fibermap['Y_FVCERR'] = np.zeros(nspec, dtype=np.float32)
    fibermap['RA_OBS'] = fibermap['RA_TARGET']
    fibermap['DEC_OBS'] = fibermap['DEC_TARGET']
    fibermap['BRICKNAME'] = brick.brickname(ra, dec)
    
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

    
