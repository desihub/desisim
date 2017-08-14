"""
desisim.targets
===============

Utility functions for working with simulated targets.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import string

import numpy as np
import yaml

from astropy.table import Table, Column, hstack

from desimodel.focalplane import FocalPlane
from desisim.io import empty_metatable, empty_star_properties
import desimodel.io
from desiutil.log import get_logger
log = get_logger()

from desiutil import brick
import desimodel.io
from desispec.io.fibermap import empty_fibermap
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

from desisim import io

def get_simtype(spectype, desi_target, bgs_target, mws_target):
    '''
    Derive the simulation type from the redshift spectype and target bits

    Args:
        spectype : array of 'GALAXY', 'QSO', 'STAR'
        *_target : target mask bits

    Returns array of simulation types: ELG, LRG, QSO, BGS, ...

    TODO: add subtypes of STAR
    '''
    simtype = np.zeros(len(spectype), dtype=(str,6))
    simtype[:] = '???'

    isGalaxy = (spectype == 'GALAXY')
    isQSO = (spectype == 'QSO')
    isStar = (spectype == 'STAR')
    isSky = (spectype == 'SKY')

    isLRG = isGalaxy & ((desi_target & desi_mask.LRG) != 0)
    isELG = isGalaxy & ((desi_target & desi_mask.ELG) != 0)
    isBGS = isGalaxy & ((bgs_target & bgs_mask.BGS_BRIGHT) != 0)
    isBGS |= isGalaxy & ((bgs_target & bgs_mask.BGS_FAINT) != 0)

    simtype[isLRG] = 'LRG'
    simtype[isELG] = 'ELG'
    simtype[isBGS] = 'BGS'
    simtype[isQSO] = 'QSO'
    simtype[isStar] = 'STAR'
    simtype[isSky] = 'SKY'

    assert np.all(simtype != '???')
    return simtype

def sample_objtype(nobj, program):
    """
    Return a random sampling of object types (dark, bright, MWS, BGS, ELG, LRG, QSO, STD, BAD_QSO)

    Args:
        nobj : number of objects to generate

    Returns:
        (true_objtype, target_objtype) where true_objtype is the array of
            what type the objects actually are and target_objtype is the
            array of type they were targeted as

    Notes:
        - Actual fiber assignment will result in higher relative fractions of
          LRGs and QSOs in early passes and more ELGs in later passes.
        - Also ensures at least 2 sky and 1 stdstar, even if nobj is small
    """
    program = program.upper()

    #- Load target densities
    tgt = desimodel.io.load_target_info()

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

    if (program == 'MWS'):
        true_objtype  +=  ['MWS_STAR']*nsci
    elif (program == 'QSO'):
        true_objtype  +=  ['QSO']*nsci
    elif (program == 'ELG'):
        true_objtype  +=  ['ELG']*nsci
    elif (program == 'LRG'):
        true_objtype  +=  ['LRG']*nsci
    elif (program == 'STD'):
        true_objtype  +=  ['STD']*nsci
    elif (program == 'BGS'):
        true_objtype  +=  ['BGS']*nsci
    elif (program in ('GRAY', 'GREY')):
        true_objtype += ['ELG',] * nsci
    elif (program == 'BRIGHT'):
        #- BGS galaxies and MWS stars
        #- TODO: split BGS bright vs. faint
        ntgt = float(tgt['nobs_bgs_faint'] + tgt['nobs_bgs_bright'] + tgt['nobs_mws'])
        prob_mws = tgt['nobs_mws'] / ntgt
        prob_bgs = 1 - prob_mws

        p = [prob_bgs, prob_mws]
        nbgs, nmws = np.random.multinomial(nsci, p)

        true_objtype += ['BGS']*nbgs
        true_objtype += ['MWS_STAR']*nmws
    elif (program == 'DARK'):
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
    elif program == 'SKY':
        #- override everything else and just return sky objects
        nlrg = nqso = nelg = nmws = nbgs = nbgs = nmws = nstd = 0
        nsky = nobj
        true_objtype = ['SKY',] * nobj
    else:
        raise ValueError("Do not know the objtype mix for program "+ program)

    assert len(true_objtype) == nobj, \
        'len(true_objtype) mismatch for program {} : {} != {}'.format(\
        program, len(true_objtype), nobj)
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
    nspec, program, tileid, seed, specify_targets, specmin = args
    return get_targets(nspec, program, tileid, seed=seed, specify_targets=specify_targets, specmin=specmin)

def get_targets_parallel(nspec, program, tileid=None, nproc=None, seed=None, specify_targets=dict()):
    '''
    Parallel wrapper for get_targets()

    nproc (int) is number of multiprocessing processes to use.
    '''
    import multiprocessing as mp
    if nproc is None:
        nproc = mp.cpu_count() // 2

    #- don't bother with parallelism if there aren't many targets
    if nspec < 20:
        log.debug('Not Parallelizing get_targets for only {} targets'.format(nspec))
        return get_targets(nspec, program, tileid, seed=seed, specify_targets=specify_targets)
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
                args.append( (n, program, tileid, seeds[i], specify_targets, i) )
            else:
                args.append( (nspec-i, program, tileid, seeds[i], specify_targets, i) )

        pool = mp.Pool(nproc)
        results = pool.map(_wrap_get_targets, args)
        fibermaps, targets = list(zip(*results))
        fibermap = np.concatenate(fibermaps)

        #- wave should be the same for all targets
        wave = targets[0][1]

        #- vstack for arrays, hstack for tables
        flux = np.vstack([tx[0] for tx in targets])
        meta = np.hstack([tx[2] for tx in targets])

        #- Fix FIBER and SPECTROID entries in fibermap
        fibermap['FIBER'] = np.arange(nspec)
        fibermap['SPECTROID'] = fibermap['FIBER'] // 500

        #- Check dimensionality
        nspec, nwave = flux.shape
        assert len(fibermap) == nspec
        assert len(meta) == nspec
        assert len(wave) == nwave

        return fibermap, (flux, wave, meta)

def get_targets(nspec, program, tileid=None, seed=None, specify_targets=dict(), specmin=0):
    """
    Generates a set of targets for the requested program

    Args:
        nspec: (int) number of targets to generate
        program: (str) program name DARK, BRIGHT, GRAY, MWS, BGS, LRG, ELG, ...

    Options:
      * tileid: (int) tileid, used for setting RA,dec
      * seed: (int) random number seed
      * specify_targets: (dict of dicts)  Define target properties like magnitude and redshift
                                 for each target class. Each objtype has its own key,value pair
                                 see simspec.templates.specify_galparams_dict() 
                                 or simsepc.templates.specify_starparams_dict()
      * specmin: (int) first spectrum number (0-indexed)

    Returns:
      * fibermap
      * targets as tuple of (flux, wave, meta)
    """
    if tileid is None:
        tile_ra, tile_dec = 0.0, 0.0
    else:
        tile_ra, tile_dec = io.get_tile_radec(tileid)

    program = program.upper()
    log.debug('Using random seed {}'.format(seed))
    np.random.seed(seed)

    #- Get distribution of target types
    true_objtype, target_objtype = sample_objtype(nspec, program)

    #- Get DESI wavelength coverage
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    dw = 0.2
    wave = np.arange(round(wavemin, 1), wavemax, dw)
    nwave = len(wave)

    flux = np.zeros( (nspec, len(wave)) )
    meta = empty_metatable(nmodel=nspec, objtype='SKY')
    fibermap = empty_fibermap(nspec)

    for objtype in set(true_objtype):
        ii = np.where(true_objtype == objtype)[0]
        nobj = len(ii)

        fibermap['OBJTYPE'][ii] = target_objtype[ii]

        if objtype in specify_targets.keys():
            obj_kwargs = specify_targets[objtype]
        else:
            obj_kwargs = dict()
                
        # Simulate spectra
        if objtype == 'SKY':
            fibermap['DESI_TARGET'][ii] = desi_mask.SKY
            continue

        elif objtype == 'ELG':
            from desisim.templates import ELG
            elg = ELG(wave=wave)
            simflux, wave1, meta1 = elg.make_templates(nmodel=nobj, seed=seed, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.ELG

        elif objtype == 'LRG':
            from desisim.templates import LRG
            lrg = LRG(wave=wave)
            simflux, wave1, meta1 = lrg.make_templates(nmodel=nobj, seed=seed, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.LRG

        elif objtype == 'BGS':
            from desisim.templates import BGS
            bgs = BGS(wave=wave)
            simflux, wave1, meta1 = bgs.make_templates(nmodel=nobj, seed=seed, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.BGS_ANY
            fibermap['BGS_TARGET'][ii] = bgs_mask.BGS_BRIGHT

        elif objtype == 'QSO':
            from desisim.templates import QSO
            qso = QSO(wave=wave)
            simflux, wave1, meta1 = qso.make_templates(nmodel=nobj, seed=seed, lyaforest=False, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.QSO

        # For a "bad" QSO simulate a normal star without color cuts, which isn't
        # right. We need to apply the QSO color-cuts to the normal stars to pull
        # out the correct population of contaminating stars.

        # Note by @moustakas: we can now do this using desisim/#150, but we are
        # going to need 'noisy' photometry (because the QSO color-cuts
        # explicitly avoid the stellar locus).
        elif objtype == 'QSO_BAD':
            from desisim.templates import STAR
            #from desitarget.cuts import isQSO
            #star = STAR(wave=wave, colorcuts_function=isQSO)
            star = STAR(wave=wave)
            simflux, wave1, meta1 = star.make_templates(nmodel=nobj, seed=seed, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.QSO

        elif objtype == 'STD':
            from desisim.templates import FSTD
            fstd = FSTD(wave=wave)
            simflux, wave1, meta1 = fstd.make_templates(nmodel=nobj, seed=seed, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.STD_FSTAR

        elif objtype == 'MWS_STAR':
            from desisim.templates import MWS_STAR
            mwsstar = MWS_STAR(wave=wave)
            # todo: mag ranges for different programs of STAR targets should be in desimodel
            if 'rmagrange' not in obj_kwargs.keys():
                obj_kwargs['rmagrange'] = (15.0,20.0)
            simflux, wave1, meta1 = mwsstar.make_templates(nmodel=nobj, seed=seed, **obj_kwargs)
            fibermap['DESI_TARGET'][ii] = desi_mask.MWS_ANY
            #- MWS bit names changed after desitarget 0.6.0 so use number
            #- instead of name for now (bit 0 = mask 1 = MWS_MAIN currently)
            fibermap['MWS_TARGET'][ii] = 1

        else:
            raise ValueError('Unable to simulate OBJTYPE={}'.format(objtype))

        flux[ii] = simflux
        meta[ii] = meta1

        fibermap['FILTER'][ii, :6] = ['DECAM_G', 'DECAM_R', 'DECAM_Z', 'WISE_W1', 'WISE_W2']
        fibermap['MAG'][ii, 0] = 22.5 - 2.5 * np.log10(meta['FLUX_G'][ii])
        fibermap['MAG'][ii, 1] = 22.5 - 2.5 * np.log10(meta['FLUX_R'][ii])
        fibermap['MAG'][ii, 2] = 22.5 - 2.5 * np.log10(meta['FLUX_Z'][ii])
        fibermap['MAG'][ii, 3] = 22.5 - 2.5 * np.log10(meta['FLUX_W1'][ii])
        fibermap['MAG'][ii, 4] = 22.5 - 2.5 * np.log10(meta['FLUX_W2'][ii])

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
    fibermap['TARGETID'] = np.random.randint(sys.maxsize, size=nspec)
    fibermap['TARGETCAT'] = np.zeros(nspec, dtype=(str, 20))
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

    return fibermap, (flux, wave, meta)

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


def _default_wave(wavemin=None, wavemax=None, dw=0.2):
    '''Construct and return the default wavelength vector.'''

    if wavemin is None:
        wavemin = desimodel.io.load_throughput('b').wavemin
    if wavemax is None:
        wavemax = desimodel.io.load_throughput('z').wavemax
    wave = np.arange(round(wavemin, 1), wavemax, dw)

    return wave
