"""
Utility functions for working with simulated targets
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import string

import numpy as np
import yaml

from astropy.table import Table, Column, hstack

from desimodel.focalplane import FocalPlane
from desisim.io import empty_metatable
import desimodel.io
from desispec.log import get_logger
log = get_logger()

from desispec import brick
from desispec.io.fibermap import empty_fibermap
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

from desisim import io

def sample_objtype(nobj, flavor):
    """
    Return a random sampling of object types (dark, bright, MWS, BGS, ELG, LRG, QSO, STD, BAD_QSO)

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
    flavor = flavor.upper()

    #- Load target densities
    #- TODO: what about nobs_boss (BOSS-like LRGs)?
    #- TODO: This function should be using a desimodel.io function instead of opening desimodel directly.
    fx = open(os.environ['DESIMODEL']+'/data/targets/targets.dat')
    tgt = yaml.load(fx)
    fx.close()

    # initialize so we can ask for 0 of some kinds of survey targets later
    nlrg = nqso = nelg = nmws = nbgs = nbgs = nmws = 0
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
    true_objtype  = ['SKY']*nsky + ['STD']*nstd
        
    if (flavor == 'MWS'):
        true_objtype  +=  ['MWS_STAR']*nsci
    elif (flavor == 'QSO'):
        true_objtype  +=  ['QSO']*nsci
    elif (flavor == 'ELG'):
        true_objtype  +=  ['ELG']*nsci    
    elif (flavor == 'LRG'):
        true_objtype  +=  ['LRG']*nsci
    elif (flavor == 'STD'):
        true_objtype  +=  ['STD']*nsci
    elif (flavor == 'BGS'):
        true_objtype  +=  ['BGS']*nsci
    elif (flavor in ('GRAY', 'GREY')):
        true_objtype += ['ELG',] * nsci
    elif (flavor == 'BRIGHT'):
        #- BGS galaxies and MWS stars
        ntgt = float(tgt['nobs_BG'] + tgt['nobs_MWS'])
        prob_bgs = tgt['nobs_BG'] / ntgt
        prob_mws = 1 - prob_bgs
        
        p = [prob_bgs, prob_mws]
        nbgs, nmws = np.random.multinomial(nsci, p)
        
        true_objtype += ['BGS']*nbgs
        true_objtype += ['MWS_STAR']*nmws
    elif (flavor == 'DARK'):
        #- LRGs ELGs QSOs
        ntgt = float(tgt['nobs_lrg'] + tgt['nobs_elg'] + tgt['nobs_qso'] + tgt['nobs_lya'] + tgt['ntarget_badqso'])
        prob_lrg = tgt['nobs_lrg'] / ntgt
        prob_elg = tgt['nobs_elg'] / ntgt
        prob_qso = (tgt['nobs_qso'] + tgt['nobs_lya']) / ntgt
        prob_badqso = tgt['ntarget_badqso'] / ntgt
        
        p = [prob_lrg, prob_elg, prob_qso, prob_badqso]
        nlrg, nelg, nqso, nbadqso = np.random.multinomial(nsci, p)

        true_objtype += ['ELG']*nelg
        true_objtype += ['LRG']*nlrg
        true_objtype += ['QSO']*nqso + ['QSO_BAD']*nbadqso
    elif flavor == 'SKY':
        #- override everything else and just return sky objects
        nlrg = nqso = nelg = nmws = nbgs = nbgs = nmws = nstd = 0
        nsky = nobj
        true_objtype = ['SKY',] * nobj
    else:
        raise ValueError("Do not know the objtype mix for flavor "+ flavor)
        
    assert len(true_objtype) == nobj, \
        'len(true_objtype) mismatch for flavor {} : {} != {}'.format(\
        flavor, len(true_objtype), nobj)
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

#- multiprocessing needs one arg, not multiple args
def _wrap_get_targets(args):
    nspec, flavor, tileid, seed, specmin = args
    return get_targets(nspec, flavor, tileid, seed=seed, specmin=specmin)
    
def get_targets_parallel(nspec, flavor, tileid=None, nproc=None, seed=None):
    import multiprocessing as mp
    if nproc is None:
        nproc = mp.cpu_count() // 2

    #- don't bother with parallelism if there aren't many targets
    if nspec < 20:
        log.debug('Not Parallelizing get_targets for only {} targets'.format(nspec))
        return get_targets(nspec, flavor, tileid, seed=seed)
    else:
        nproc = min(nproc, nspec//10)        
        log.debug('Parallelizing get_targets using {} cores'.format(nproc))
        args = list()
        n = nspec // nproc
        #- Generate random seeds for each process to use as a random seed
        np.random.seed(seed)
        seeds = np.random.randint(2**32, size=nspec)
        for i in range(0, nspec, n):
            if i+n < nspec:
                args.append( (n, flavor, tileid, seeds[i], i) )
            else:
                args.append( (nspec-i, flavor, tileid, seeds[i], i) )

        pool = mp.Pool(nproc)
        results = pool.map(_wrap_get_targets, args)
        fibermaps, truthtables = zip(*results)
        fibermap = np.concatenate(fibermaps)

        truth = truthtables[0]
        for key in truth.keys():
            if key not in ('UNITS', 'WAVE'):
                truth[key] = np.concatenate([t[key] for t in truthtables])

        #- Fix FIBER and SPECTROID entries in fibermap
        fibermap['FIBER'] = np.arange(nspec)
        fibermap['SPECTROID'] = fibermap['FIBER'] // 500

        return fibermap, truth

#- Work in progress; don't use yet.
def _simspec_truth(truth, wave=None, seed=None):
    from astropy import table
    #- Get DESI wavelength coverage
    if wave is None:
        wavemin = desimodel.io.load_throughput('b').wavemin
        wavemax = desimodel.io.load_throughput('z').wavemax
        dw = 0.2
        wave = np.arange(round(wavemin, 1), wavemax, dw)

    truetype = truth['TRUETYPE']
    subtype = truth['TRUESUBTYPE']
    isGAL = (truetype == 'GALAXY')
    isQSO = (truetype == 'QSO')
    isSTAR = (truetype == 'STAR')

    isLRG = isGAL & (subtype == 'LRG')
    isELG = isGAL & (subtype == 'ELG')
    isBGS = isGAL & (subtype == 'BGS')
    isMWS = isSTAR & (subtype == 'MWS')
    isSTD = isSTAR & (subtype == 'FSTD')

    isFakeQSO = isSTAR & (subtype == 'FAKE_QSO')
    isFakeELG = isSTAR & (subtype == 'FAKE_ELG')
    isFakeLRG = isSTAR & (subtype == 'FAKE_LRG')

    #- did we cover all options?
    x = isLRG | isELG | isBGS | isQSO | isMWS | isSTD
    x |= isFakeQSO | isFakeELG | isFakeLRG
    if np.any(~x):
        unknown = set(zip(truetype[~x], subtype[~x]))
        message = 'Unknown objtype types {}'.format(unknown)
        log.fatal(message)
        raise ValueError(message)

    def make_templates(truth, template_class, wave, seed):
        nobj = len(truth)
        tx = template_class(wave=wave)
        print('generating {} {} targets'.format(nobj, tx.objtype))
        simflux, _x, meta = tx.make_templates(nmodel=nobj, seed=seed)
        meta.add_column(table.Column(name='TARGETID', data=truth['TARGETID']))
        return simflux, meta

    from desisim.templates import LRG, ELG, QSO, BGS, MWS_STAR, FSTD

    results = list()
    if np.any(isLRG):
        results.append( make_templates(truth[isLRG], LRG, wave, seed) )
    if np.any(isELG):
        results.append( make_templates(truth[isELG], ELG, wave, seed) )
    if np.any(isQSO):
        results.append( make_templates(truth[isQSO], QSO, wave, seed) )
    if np.any(isBGS):
        results.append( make_templates(truth[isBGS], BGS, wave, seed) )
    if np.any(isMWS):
        results.append( make_templates(truth[isMWS], MWS, wave, seed) )
    if np.any(isSTD):
        results.append( make_templates(truth[isSTD], FSTD, wave, seed) )
    if np.any(isFakeQSO):
        log.warn('not applying QSO color cuts to Fake QSOs yet')
        results.append( make_templates(truth[isFakeQSO], MWS_STAR, wave, seed) )
    if np.any(isFakeELG):
        log.warn('not applying ELG color cuts to Fake ELGs yet')
        results.append( make_templates(truth[isFakeELG], MWS_STAR, wave, seed) )
    if np.any(isFakeLRG):
        log.warn('not applying LRG color cuts to Fake LRGs yet')
        results.append( make_templates(truth[isFakeLRG], MWS_STAR, wave, seed) )

    simflux = np.vstack([x[0] for x in results])
    simmeta = table.vstack([x[1] for x in results])

    #- Sort to match order of input truth
    #- Is there a way to do this without 3 argsort calls?
    ii = np.argsort(np.asarray(truth['TARGETID']))
    jj = np.argsort(np.asarray(simmeta['TARGETID'])) 
    kk = np.argsort(ii)   
    simmeta = simmeta[jj[kk]]
    simflux = simflux[jj[kk]]

    return simflux, simmeta

def get_targets(nspec, flavor, tileid=None, seed=None, specmin=0):
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

    flavor = flavor.upper()
    log.debug('Using random seed {}'.format(seed))
    np.random.seed(seed)

    #- Get distribution of target types
    true_objtype, target_objtype = sample_objtype(nspec, flavor)
    
    #- Get DESI wavelength coverage
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    dw = 0.2
    wave = np.arange(round(wavemin, 1), wavemax, dw)
    nwave = len(wave)

    truth = dict()
    truth['FLUX'] = np.zeros( (nspec, len(wave)) )
    truth['OBJTYPE'] = np.zeros(nspec, dtype='S10')
    ##- Note: unlike other elements, first index of WAVE isn't spectrum index
    truth['WAVE'] = wave

    truth['META'] = empty_metatable(nmodel=nspec, objtype='SKY')

    fibermap = empty_fibermap(nspec)

    for objtype in set(true_objtype):
        ii = np.where(true_objtype == objtype)[0]
        nobj = len(ii)

        fibermap['OBJTYPE'][ii] = target_objtype[ii]
        truth['OBJTYPE'][ii] = true_objtype[ii]

        # Simulate spectra
        if objtype == 'SKY':
            fibermap['DESI_TARGET'][ii] = desi_mask.SKY
            continue

        elif objtype == 'ELG':
            from desisim.templates import ELG
            elg = ELG(wave=wave)
            simflux, wave1, meta = elg.make_templates(nmodel=nobj, seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.ELG

        elif objtype == 'LRG':
            from desisim.templates import LRG
            lrg = LRG(wave=wave)
            simflux, wave1, meta = lrg.make_templates(nmodel=nobj, seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.LRG

        elif objtype == 'BGS':
            from desisim.templates import BGS
            bgs = BGS(wave=wave)
            simflux, wave1, meta = bgs.make_templates(nmodel=nobj, seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.BGS_ANY
            fibermap['BGS_TARGET'][ii] = bgs_mask.BGS_BRIGHT

        elif objtype == 'QSO':
            from desisim.templates import QSO
            qso = QSO(wave=wave)
            simflux, wave1, meta = qso.make_templates(nmodel=nobj, seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.QSO

        # For a "bad" QSO simulate a normal star without color cuts, which isn't
        # right. We need to apply the QSO color-cuts to the normal stars to pull
        # out the correct population of contaminating stars.
        elif objtype == 'QSO_BAD':
            from desisim.templates import STAR
            star = STAR(wave=wave)
            simflux, wave1, meta = star.make_templates(nmodel=nobj, seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.QSO

        elif objtype == 'STD':
            from desisim.templates import FSTD
            fstd = FSTD(wave=wave)
            simflux, wave1, meta = fstd.make_templates(nmodel=nobj, seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.STD_FSTAR

        elif objtype == 'MWS_STAR':
            from desisim.templates import MWS_STAR
            mwsstar = MWS_STAR(wave=wave)
            # todo: mag ranges for different flavors of STAR targets should be in desimodel
            simflux, wave1, meta = mwsstar.make_templates(nmodel=nobj,rmagrange=(15.0,20.0), seed=seed)
            fibermap['DESI_TARGET'][ii] = desi_mask.MWS_ANY
            fibermap['MWS_TARGET'][ii] = mws_mask.MWS_PLX  #- ???

        else:
            raise ValueError('Unable to simulate OBJTYPE={}'.format(objtype))

        truth['FLUX'][ii] = 1e17 * simflux
        truth['UNITS'] = '1e-17 erg/s/cm2/A'
        truth['META'][ii] = meta
        
        # Pack in the photometry.  This needs updating!
        grz = 22.5-2.5*np.log10(meta['DECAM_FLUX'].data.flatten()[[1, 2, 4]])
        wise = 22.5-2.5*np.log10(meta['WISE_FLUX'].data.flatten()[[0, 1]])
        fibermap['MAG'][ii, :6] = np.vstack(np.hstack([grz, wise])).T
        fibermap['FILTER'][ii, :6] = ['DECAM_G', 'DECAM_R', 'DECAM_Z', 'WISE_W1', 'WISE_W2']

    # Only store the metadata table for non-sky spectra.
    notsky = np.where(true_objtype != 'SKY')[0]
    if len(notsky) > 0:
        truth['META'] = truth['META'][notsky]

    #- Load fiber -> positioner mapping and tile information
    fiberpos = desimodel.io.load_fiberpos()

    #- Where are these targets?  Centered on positioners for now.
    x = fiberpos['X'][specmin:specmin+nspec]
    y = fiberpos['Y'][specmin:specmin+nspec]
    fp = FocalPlane(tile_ra, tile_dec)
    ra = np.zeros(nspec)
    dec = np.zeros(nspec)
    for i in range(nspec):
        ra[i], dec[i] = fp.xy2radec(x[i], y[i])

    #- Fill in the rest of the fibermap structure
    fibermap['FIBER'] = np.arange(nspec, dtype='i4')
    fibermap['POSITIONER'] = fiberpos['POSITIONER'][specmin:specmin+nspec]
    fibermap['SPECTROID'] = fiberpos['SPECTROGRAPH'][specmin:specmin+nspec]
    fibermap['TARGETID'] = np.random.randint(sys.maxint, size=nspec)
    fibermap['TARGETCAT'] = np.zeros(nspec, dtype='|S20')
    fibermap['LAMBDAREF'] = np.ones(nspec, dtype=np.float32)*5400
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

