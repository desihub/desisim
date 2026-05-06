'''
desisim.quickcat
================

Code for quickly generating an output zcatalog given fiber assignment tiles,
a truth catalog, and optionally a previous zcatalog.
'''

import os
import yaml
from collections import Counter
from importlib import resources
from time import asctime

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column, vstack
import sys
import scipy.special as sp
import desisim
from desisim.targets import get_simtype

import astropy.constants
c = astropy.constants.c.to('km/s').value

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

from desiutil.log import get_logger
log = get_logger()

from desisim.util import hadec2airmass

#- redshift errors, zwarn, cata fail rate fractions from
#- /project/projectdirs/desi/datachallenge/redwood/spectro/redux/redwood/
#- sigmav = c sigmaz / (1+z)
_sigma_v = {
#    'ELG': 38.03,
#    'LRG': 67.38,
    'BGS': 37.70,
#    'QSO': 182.16,
    'STAR': 51.51,
    'WD':54.35,
    'SKY': 9999,      #- meaningless
    'UNKNOWN': 9999,  #- meaningless
}

_zwarn_fraction = {
#    'ELG': 0.087,
#    'LRG': 0.007,
#    'QSO': 0.020,
    'BGS': 0.024,
    'STAR': 0.345,
    'WD':0.094,
    'SKY': 1.0,
    'UNKNOWN': 1.0,
}

_cata_fail_fraction = {
#    'ELG': 0.020,
#    'LRG': 0.002,
#    'QSO': 0.012,
    'BGS': 0.003,
    'STAR': 0.050,
    'WD':0.0,
    'SKY': 0.,
    'UNKNOWN': 0.,
}

def get_zeff_obs(simtype, obsconditions):
    '''
    '''
    if(simtype=='LRG'):
        p_v = [1.0, 0.15, -0.5]
        p_w = [1.0, 0.4, 0.0]
        p_x = [1.0, 0.06, 0.05]
        p_y = [1.0, 0.0, 0.08]
        p_z = [1.0, 0.0, 0.0]
        sigma_r = 0.02
    elif(simtype=='QSO'):
        p_v = [1.0, -0.2, 0.3]
        p_w = [1.0, -0.5, 0.6]
        p_x = [1.0, -0.1, -0.075]
        p_y = [1.0, -0.08, -0.04]
        p_z = [1.0, 0.0, 0.0]
        sigma_r = 0.05
    elif(simtype=='ELG'):
        p_v = [1.0, -0.1, -0.2]
        p_w = [1.0, 0.25, -0.75]
        p_x = [1.0, 0.0, 0.05]
        p_y = [1.0, 0.2, 0.1]
        p_z = [1.0, -10.0, 300.0]
        sigma_r = 0.075
    else:
        log.warning('No model for how observing conditions impact {} redshift efficiency'.format(simtype))
        return np.ones(len(obsconditions))

    ncond = len(np.atleast_1d(obsconditions['AIRMASS']))
    
    # airmass
    v = obsconditions['AIRMASS'] - np.mean(obsconditions['AIRMASS'])
    pv  = p_v[0] + p_v[1] * v + p_v[2] * (v**2. - np.mean(v**2))

    # ebmv
    # KeyError if dict or Table is missing EBMV
    # ValueError if ndarray is missing EBMV
    try:
        w = obsconditions['EBMV'] - np.mean(obsconditions['EBMV'])
        pw = p_w[0] + p_w[1] * w + p_w[2] * (w**2 - np.mean(w**2))
    except (KeyError, ValueError):
        pw = np.ones(ncond)
    
    # seeing
    x = obsconditions['SEEING'] - np.mean(obsconditions['SEEING'])
    px = p_x[0] + p_x[1]*x + p_x[2] * (x**2 - np.mean(x**2))

    # transparency
    try:
        y = obsconditions['LINTRANS'] - np.mean(obsconditions['LINTRANS'])
        py = p_y[0] + p_y[1]*y + p_y[2] * (y**2 - np.mean(y**2))
    except (KeyError, ValueError):
        py = np.ones(ncond)
    
    # moon illumination fraction
    z = obsconditions['MOONFRAC'] - np.mean(obsconditions['MOONFRAC'])
    pz = p_z[0] + p_z[1]*z + p_z[2] * (z**2 - np.mean(z**2))

    #- if moon is down phase doesn't matter
    pz = np.ones(ncond)
    pz[obsconditions['MOONALT'] < 0] = 1.0

    pr = 1.0 + np.random.normal(size=ncond, scale=sigma_r)

    #- this correction factor can be greater than 1, but not less than 0
    pobs = (pv * pw * px * py * pz * pr).clip(min=0.0)

    return pobs


def get_redshift_efficiency(simtype, targets, truth, targets_in_tile, obsconditions, params, ignore_obscondition=False):
    """
    Simple model to get the redshift effiency from the observational conditions or observed magnitudes+redshuft

    Args:
        simtype: ELG, LRG, QSO, MWS, BGS
        targets: target catalog table; currently used only for TARGETID
        truth: truth table with OIIFLUX, TRUEZ
        targets_in_tile: dictionary. Keys correspond to tileids, its values are the
            arrays of targetids observed in that tile.
        obsconditions: table observing conditions with columns
           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
        parameter_filename: yaml file with quickcat parameters
        ignore_obscondition: if True, no variation of efficiency with obs. conditions (adjustment of exposure time should correct for mean change of S/N)
    Returns:
        tuple of arrays (observed, p) both with same length as targets

        observed: boolean array of whether the target was observed in these tiles

        p: probability to get this redshift right
    """
    targetid = targets['TARGETID']
    n = len(targetid)

    try:
        if 'DECAM_FLUX' in targets.dtype.names :
            true_gflux = targets['DECAM_FLUX'][:, 1]
            true_rflux = targets['DECAM_FLUX'][:, 2]
        else:
            true_gflux = targets['FLUX_G']
            true_rflux = targets['FLUX_R']
    except:
        raise Exception('Missing photometry needed to estimate redshift efficiency!')

    a_small_flux=1e-40
    true_gflux[true_gflux<a_small_flux]=a_small_flux
    true_rflux[true_rflux<a_small_flux]=a_small_flux
    

    
    if (obsconditions is None) or ('OIIFLUX' not in truth.dtype.names):
        raise Exception('Missing obsconditions and flux information to estimate redshift efficiency')

    
    
    if (simtype == 'ELG'):
        # Read the model OII flux threshold (FDR fig 7.12 modified to fit redmonster efficiency on OAK)
        # filename = str(resources.files('desisim').joinpath('data', 'quickcat_elg_oii_flux_threshold.txt'))
        
        # Read the model OII flux threshold (FDR fig 7.12)
        filename = str(resources.files('desisim').joinpath('data', 'elg_oii_flux_threshold_fdr.txt'))
        fdr_z, modified_fdr_oii_flux_threshold = np.loadtxt(filename, unpack=True)
        
        # Compute OII flux thresholds for truez
        oii_flux_limit = np.interp(truth['TRUEZ'],fdr_z,modified_fdr_oii_flux_threshold)
        oii_flux_limit[oii_flux_limit<1e-20]=1e-20
        
        # efficiency is modeled as a function of flux_OII/f_OII_threshold(z) and an arbitrary sigma_fudge
        
        snr_in_lines       = params["ELG"]["EFFICIENCY"]["SNR_LINES_SCALE"]*7*truth['OIIFLUX']/oii_flux_limit
        snr_in_continuum   = params["ELG"]["EFFICIENCY"]["SNR_CONTINUUM_SCALE"]*true_rflux
        snr_tot            = np.sqrt(snr_in_lines**2+snr_in_continuum**2)
        sigma_fudge        = params["ELG"]["EFFICIENCY"]["SIGMA_FUDGE"]
        nsigma             = 3.
        simulated_eff = eff_model(snr_tot,nsigma,sigma_fudge)

    elif(simtype == 'LRG'):
        
       
        r_mag = 22.5 - 2.5*np.log10(true_rflux)
        
        sigmoid_cutoff = params["LRG"]["EFFICIENCY"]["SIGMOID_CUTOFF"]
        sigmoid_fudge  = params["LRG"]["EFFICIENCY"]["SIGMOID_FUDGE"]
        simulated_eff = 1./(1.+np.exp((r_mag-sigmoid_cutoff)/sigmoid_fudge))

        log.info("{} eff = sigmoid with cutoff = {:4.3f} fudge = {:4.3f}".format(simtype,sigmoid_cutoff,sigmoid_fudge))
    
    elif(simtype == 'QSO'):
        
        zsplit = params['QSO_ZSPLIT']
        r_mag = 22.5 - 2.5*np.log10(true_rflux) 
        simulated_eff = np.ones(r_mag.shape)

        # lowz tracer qsos
        sigmoid_cutoff = params["LOWZ_QSO"]["EFFICIENCY"]["SIGMOID_CUTOFF"]
        sigmoid_fudge  = params["LOWZ_QSO"]["EFFICIENCY"]["SIGMOID_FUDGE"]
        ii=(truth['TRUEZ']<=zsplit)
        simulated_eff[ii] = 1./(1.+np.exp((r_mag[ii]-sigmoid_cutoff)/sigmoid_fudge))
        log.info("{} eff = sigmoid with cutoff = {:4.3f} fudge = {:4.3f}".format("LOWZ QSO",sigmoid_cutoff,sigmoid_fudge))
                
        # highz lya qsos
        sigmoid_cutoff = params["LYA_QSO"]["EFFICIENCY"]["SIGMOID_CUTOFF"]
        sigmoid_fudge  = params["LYA_QSO"]["EFFICIENCY"]["SIGMOID_FUDGE"]
        ii=(truth['TRUEZ']>zsplit)
        simulated_eff[ii] = 1./(1.+np.exp((r_mag[ii]-sigmoid_cutoff)/sigmoid_fudge))

        log.info("{} eff = sigmoid with cutoff = {:4.3f} fudge = {:4.3f}".format("LYA QSO",sigmoid_cutoff,sigmoid_fudge))
        
    elif simtype == 'BGS':
        simulated_eff = 0.98 * np.ones(n)

    elif simtype == 'MWS':
        simulated_eff = 0.98 * np.ones(n)

    else:
        default_zeff = 0.98
        log.warning('using default redshift efficiency of {} for {}'.format(default_zeff, simtype))
        simulated_eff = default_zeff * np.ones(n)

    #- Get the corrections for observing conditions per tile, then
    #- correct targets on those tiles.  Parameterize in terms of failure
    #- rate instead of success rate to handle bookkeeping of targets that
    #- are observed on more than one tile.
    #- NOTE: this still isn't quite right since multiple observations will
    #- be simultaneously fit instead of just taking whichever individual one
    #- succeeds.

    if ignore_obscondition :
        ncond = len(np.atleast_1d(obsconditions['AIRMASS']))
        zeff_obs = np.ones(ncond)
    else :
        zeff_obs = get_zeff_obs(simtype, obsconditions)
    pfail = np.ones(n)
    observed = np.zeros(n, dtype=bool)

    # More efficient alternative for large numbers of tiles + large target
    # list, but requires pre-computing the sort order of targetids.
    # Assume targets['TARGETID'] is unique, so not checking this.
    sort_targetid = np.argsort(targetid)

    # Extract the targets-per-tile lists into one huge list.
    concat_targets_in_tile  = np.concatenate([targets_in_tile[tileid] for tileid in obsconditions['TILEID']])
    ntargets_per_tile       = np.array([len(targets_in_tile[tileid])  for tileid in obsconditions['TILEID']])

    # Match entries in each tile list against sorted target list.
    target_idx    = targetid[sort_targetid].searchsorted(concat_targets_in_tile,side='left')
    target_idx_r  = targetid[sort_targetid].searchsorted(concat_targets_in_tile,side='right')
    del(concat_targets_in_tile)

    # Flag targets in tiles that do not appear in the target list (sky,
    # standards).
    not_matched = target_idx_r - target_idx == 0
    target_idx[not_matched] = -1
    del(target_idx_r,not_matched)

    # Not every tile has 5000 targets, so use individual counts to
    # construct offset of each tile in target_idx.
    offset  = np.concatenate([[0],np.cumsum(ntargets_per_tile[:-1])])

    # For each tile, process targets.
    for i, tileid in enumerate(obsconditions['TILEID']):
        if ntargets_per_tile[i] > 0:
            # Quickly get all the matched targets on this tile.
            targets_this_tile  = target_idx[offset[i]:offset[i]+ntargets_per_tile[i]]
            targets_this_tile  = targets_this_tile[targets_this_tile > 0]
            # List of indices into sorted target list for each observed
            # source.
            ii  = sort_targetid[targets_this_tile]
            tmp = (simulated_eff[ii]*zeff_obs[i]).clip(0, 1)
            pfail[ii] *= (1-tmp)
            observed[ii] = True

    simulated_eff = (1-pfail)

    return observed, simulated_eff

# Efficiency model
def eff_model(x, nsigma, sigma, max_efficiency=1):
    return 0.5*max_efficiency*(1.+sp.erf((x-nsigma)/(np.sqrt(2.)*sigma)))

def reverse_dictionary(a):
    """Inverts a dictionary mapping.

    Args:
        a: input dictionary.

    Returns:
        b: output reversed dictionary.
    """
    b = {}
    for i in a.items():
        try:
            for k in i[1]:
                if k not in b.keys():
                    b[k] = [i[0]]
                else:
                    b[k].append(i[0])
        except:
            k = i[1]
            if k not in b.keys():
                b[k] = [i[0]]
            else:
                b[k].append(i[0])
    return b

def get_observed_redshifts(targets, truth, targets_in_tile, obsconditions, parameter_filename=None, ignore_obscondition=False):
    """
    Returns observed z, zerr, zwarn arrays given true object types and redshifts

    Args:
        targets: target catalog table; currently used only for target mask bits
        truth: truth table with OIIFLUX, TRUEZ
        targets_in_tile: dictionary. Keys correspond to tileids, its values are the
            arrays of targetids observed in that tile.
        obsconditions: table observing conditions with columns
           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
        parameter_filename: yaml file with quickcat parameters
        ignore_obscondition: if True, no variation of efficiency with obs. conditions (adjustment of exposure time should correct for mean change of S/N)
    Returns:
        tuple of (zout, zerr, zwarn)
    """
    
    if parameter_filename is None :
        # Load efficiency parameters yaml file
        parameter_filename = str(resources.files('desisim').joinpath('data', 'quickcat.yaml'))
    
    params=None
    with open(parameter_filename,"r") as file :
        params = yaml.safe_load(file)
        
    
    simtype = get_simtype(np.char.strip(truth['TRUESPECTYPE']), targets['DESI_TARGET'], targets['BGS_TARGET'], targets['MWS_TARGET'])
    #simtype = get_simtype(np.char.strip(truth['TEMPLATETYPE']), targets['DESI_TARGET'], targets['BGS_TARGET'], targets['MWS_TARGET'])
    truez = truth['TRUEZ']
    targetid = truth['TARGETID']

    try:
        if 'DECAM_FLUX' in targets.dtype.names :
            true_gflux = targets['DECAM_FLUX'][:, 1]
            true_rflux = targets['DECAM_FLUX'][:, 2]
        else:
            true_gflux = targets['FLUX_G']
            true_rflux = targets['FLUX_R']
    except:
        raise Exception('Missing photometry needed to estimate redshift efficiency!')

    a_small_flux=1e-40
    true_gflux[true_gflux<a_small_flux]=a_small_flux
    true_rflux[true_rflux<a_small_flux]=a_small_flux
    
    zout = truez.copy()
    zerr = np.zeros(len(truez), dtype=np.float32)
    zwarn = np.zeros(len(truez), dtype=np.int32)

    objtypes = list(set(simtype))
    n_tiles = len(np.unique(obsconditions['TILEID']))

    if(n_tiles!=len(targets_in_tile)):
        raise ValueError('Number of obsconditions {} != len(targets_in_tile) {}'.format(n_tiles, len(targets_in_tile)))

    for objtype in objtypes:

        ii=(simtype==objtype)

        ###################################
        # redshift errors
        ###################################
        if objtype =='ELG' :

            sigma         = params["ELG"]["UNCERTAINTY"]["SIGMA_17"]
            powerlawindex = params["ELG"]["UNCERTAINTY"]["POWER_LAW_INDEX"]
            oiiflux       = truth['OIIFLUX'][ii]*1e17
            zerr[ii]      = sigma/(1.e-9+oiiflux**powerlawindex)*(1.+truez[ii])
            zout[ii]     += np.random.normal(scale=zerr[ii])
            
            log.info("ELG sigma={:6.5f} index={:3.2f} median zerr={:6.5f}".format(sigma,powerlawindex,np.median(zerr[ii])))
                
        elif objtype == 'LRG' :

            sigma         = params["LRG"]["UNCERTAINTY"]["SIGMA_17"]
            powerlawindex = params["LRG"]["UNCERTAINTY"]["POWER_LAW_INDEX"]
            
            zerr[ii]  = sigma/(1.e-9+true_rflux[ii]**powerlawindex)*(1.+truez[ii])
            zout[ii] += np.random.normal(scale=zerr[ii])
                
            log.info("LRG sigma={:6.5f} index={:3.2f} median zerr={:6.5f}".format(sigma,powerlawindex,np.median(zerr[ii])))

        elif objtype == 'QSO' :

            zsplit = params['QSO_ZSPLIT']
            sigma         = params["LOWZ_QSO"]["UNCERTAINTY"]["SIGMA_17"]
            powerlawindex = params["LOWZ_QSO"]["UNCERTAINTY"]["POWER_LAW_INDEX"]                               
            jj=ii&(truth['TRUEZ']<=zsplit)
            zerr[jj]  = sigma/(1.e-9+(true_rflux[jj])**powerlawindex)*(1.+truez[jj])

            log.info("LOWZ QSO sigma={:6.5f} index={:3.2f} median zerr={:6.5f}".format(sigma,powerlawindex,np.median(zerr[jj])))
            
            sigma         = params["LYA_QSO"]["UNCERTAINTY"]["SIGMA_17"]
            powerlawindex = params["LYA_QSO"]["UNCERTAINTY"]["POWER_LAW_INDEX"]
            jj=ii&(truth['TRUEZ']>zsplit)
            zerr[jj]  = sigma/(1.e-9+(true_rflux[jj])**powerlawindex)*(1.+truez[jj])
            
            if np.count_nonzero(jj) > 0:
                log.info("LYA QSO sigma={:6.5f} index={:3.2f} median zerr={:6.5f}".format(
                    sigma,powerlawindex,np.median(zerr[jj])))
            else:
                log.warning("No LyA QSO generated")

            
            zout[ii] += np.random.normal(scale=zerr[ii])
        elif objtype in _sigma_v.keys() :

            log.info("{} use constant sigmav = {} km/s".format(objtype,_sigma_v[objtype]))
            ii = (simtype == objtype)
            zerr[ii] = _sigma_v[objtype] * (1+truez[ii]) / c
            zout[ii] += np.random.normal(scale=zerr[ii])
        else :
            log.info("{} no redshift error model, will use truth")
                
        ###################################
        # redshift efficiencies
        ###################################
        # Set ZWARN flags for some targets
        # the redshift efficiency only sets warning, but does not impact
        # the redshift value and its error.
        was_observed, goodz_prob = get_redshift_efficiency(
            objtype, targets[ii], truth[ii], targets_in_tile,
            obsconditions=obsconditions,params=params,
            ignore_obscondition=ignore_obscondition)

        n=np.sum(ii)
        assert len(was_observed) == n
        assert len(goodz_prob) == n
        r = np.random.random(len(was_observed))
        zwarn[ii] = 4 * (r > goodz_prob) * was_observed

        ###################################
        # catastrophic failures
        ###################################

        
        zlim=[0.,3.5]
        cata_fail_fraction = np.zeros(n)
        if objtype == "ELG" :
            cata_fail_fraction[:] = params["ELG"]["FAILURE_RATE"]
            zlim=[0.6,1.7]
        elif objtype == "LRG" :
            cata_fail_fraction[:] = params["LRG"]["FAILURE_RATE"]
            zlim=[0.5,1.1]
        elif objtype == "QSO" :
            zsplit = params["QSO_ZSPLIT"]
            cata_fail_fraction[truth['TRUEZ'][ii]<=zsplit] = params["LOWZ_QSO"]["FAILURE_RATE"]
            cata_fail_fraction[truth['TRUEZ'][ii]>zsplit] = params["LYA_QSO"]["FAILURE_RATE"]
            zlim=[0.5,3.5]
        elif objtype in _cata_fail_fraction :
            cata_fail_fraction[:] = _cata_fail_fraction[objtype]
        
        failed = (np.random.uniform(size=n)<cata_fail_fraction)&(zwarn[ii]==0)
        failed_indices = np.where(ii)[0][failed]
        log.info("{} n_failed/n_tot={}/{}={:4.3f}".format(objtype,failed_indices.size,n,failed_indices.size/float(n)))
        zout[failed_indices] = np.random.uniform(zlim[0],zlim[1],failed_indices.size)
        
    return zout, zerr, zwarn

def get_median_obsconditions(tileids):
    """Gets the observational conditions for a set of tiles.

    Args:
       tileids : list of tileids that were observed

    Returns:
        Table with the observational conditions for every tile.

        It inclues at least the following columns::

           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
    """
    #- Load standard DESI tiles and trim to this list of tileids
    import desimodel.io
    tiles = desimodel.io.load_tiles()
    tileids = np.asarray(tileids)
    ii = np.isin(tiles['TILEID'], tileids)

    tiles = tiles[ii]
    assert len(tiles) == len(tileids)

    #- Sort tiles to match order of tileids
    i = np.argsort(tileids)
    j = np.argsort(tiles['TILEID'])
    k = np.argsort(i)

    tiles = tiles[j[k]]
    assert np.all(tiles['TILEID'] == tileids)

    n = len(tileids)

    #- ensure UPPERCASE just in case
    program = np.char.upper(tiles['PROGRAM'])

    obsconditions = Table()
    obsconditions['TILEID'] = tileids
    obsconditions['AIRMASS'] = hadec2airmass(tiles['DESIGNHA'], tiles['DEC'])
    obsconditions['EBMV'] = tiles['EBV_MED']
    obsconditions['LINTRANS'] = np.ones(n)
    obsconditions['SEEING'] = np.ones(n) * 1.1

    #- Add lunar conditions, defaulting to dark time
    obsconditions['MOONFRAC'] = np.zeros(n)
    obsconditions['MOONALT'] = -20.0 * np.ones(n)
    obsconditions['MOONSEP'] = 180.0 * np.ones(n)

    #- bright, bright1b programs
    ii = np.char.startswith(program, 'BRIGHT')
    obsconditions['MOONFRAC'][ii] = 0.7
    obsconditions['MOONALT'][ii] = 60.0
    obsconditions['MOONSEP'][ii] = 50.0

    #- backup program
    ii = np.char.startswith(program, 'BACKUP')
    obsconditions['MOONFRAC'][ii] = 1.0
    obsconditions['MOONALT'][ii] = 60.0
    obsconditions['MOONSEP'][ii] = 50.0

    return obsconditions

def quickcat(tilefiles, targets, truth, fassignhdu='FIBERASSIGN', zcat=None, obsconditions=None, perfect=False):
    """
    Generates quick output zcatalog

    Args:
        tilefiles : list of fiberassign tile files that were observed
        targets : astropy Table of targets
        truth : astropy Table of input truth with columns TARGETID, TRUEZ, and TRUETYPE
        zcat (optional): input zcatalog Table from previous observations
        obsconditions (optional): Table or ndarray with observing conditions from surveysim
        perfect (optional): if True, treat spectro pipeline as perfect with input=output,
            otherwise add noise and zwarn!=0 flags

    Returns:
        zcatalog astropy Table based upon input truth, plus ZERR, ZWARN,
        NUMOBS, and TYPE columns
    """
    #- convert to Table for easier manipulation
    if not isinstance(truth, Table):
        truth = Table(truth)

    #- Count how many times each target was observed for this set of tiles
    log.info('{} QC Reading {} tiles'.format(asctime(), len(tilefiles)))
    nobs = Counter()
    targets_in_tile = {}
    tileids = list()
    for infile in tilefiles:
        
        fibassign, header = fits.getdata(infile, fassignhdu, header=True)
 
        # hack needed here rnc 7/26/18
        if 'TILEID' in header:
            tileidnew = header['TILEID']
        else:
            fnew=infile.split('/')[-1]
            tileidnew=fnew.split("_")[-1]
            tileidnew=int(tileidnew[:-5])
            log.error('TILEID missing from {} header'.format(fnew))
            log.error('{} -> TILEID {}'.format(infile, tileidnew))

        tileids.append(tileidnew)

        ii = (fibassign['TARGETID'] != -1)  #- targets with assignments
        nobs.update(fibassign['TARGETID'][ii])
        targets_in_tile[tileidnew] = fibassign['TARGETID'][ii]

    #- Trim obsconditions to just the tiles that were observed
    if obsconditions is not None:
        ii = np.isin(obsconditions['TILEID'], tileids)
        if np.any(ii == False):
            obsconditions = obsconditions[ii]
        assert len(obsconditions) > 0

    #- Sort obsconditions to match order of tiles
    #- This might not be needed, but is fast for O(20k) tiles and may
    #- prevent future surprises if code expects them to be row aligned
    tileids = np.array(tileids)
    if (obsconditions is not None) and \
       (np.any(tileids != obsconditions['TILEID'])):
        i = np.argsort(tileids)
        j = np.argsort(obsconditions['TILEID'])
        k = np.argsort(i)
        obsconditions = obsconditions[j[k]]
        assert np.all(tileids == obsconditions['TILEID'])

    #- Trim truth down to just ones that have already been observed
    log.info('{} QC Trimming truth to just observed targets'.format(asctime()))
    obs_targetids = np.array(list(nobs.keys()))
    iiobs = np.isin(truth['TARGETID'], obs_targetids)
    truth = truth[iiobs]
    targets = targets[iiobs]

    #- Construct initial new z catalog
    log.info('{} QC Constructing new redshift catalog'.format(asctime()))
    newzcat = Table()
    newzcat['TARGETID'] = truth['TARGETID']
    if 'BRICKNAME' in truth.dtype.names:
        newzcat['BRICKNAME'] = truth['BRICKNAME']
    else:
        newzcat['BRICKNAME'] = np.zeros(len(truth), dtype=(str, 8))

    #- Copy TRUETYPE -> SPECTYPE so that we can change without altering original
    newzcat['SPECTYPE'] = truth['TRUESPECTYPE'].copy()

    #- Add ZERR and ZWARN
    log.info('{} QC Adding ZERR and ZWARN'.format(asctime()))
    nz = len(newzcat)
    if perfect:
        newzcat['Z'] = truth['TRUEZ'].copy()
        newzcat['ZERR'] = np.zeros(nz, dtype=np.float32)
        newzcat['ZWARN'] = np.zeros(nz, dtype=np.int32)
    else:
        # get the observational conditions for the current tilefiles
        if obsconditions is None:
            obsconditions = get_median_obsconditions(tileids)

        # get the redshifts
        z, zerr, zwarn = get_observed_redshifts(targets, truth, targets_in_tile, obsconditions)
        newzcat['Z'] = z  #- update with noisy redshift
        newzcat['ZERR'] = zerr
        newzcat['ZWARN'] = zwarn

    #- Add numobs column
    log.info('{} QC Adding NUMOBS column'.format(asctime()))
    newzcat.add_column(Column(name='NUMOBS', length=nz, dtype=np.int32))
    for i in range(nz):
        newzcat['NUMOBS'][i] = nobs[newzcat['TARGETID'][i]]

    #- Merge previous zcat with newzcat
    log.info('{} QC Merging previous zcat'.format(asctime()))
    if zcat is not None:
        #- don't modify original
        #- Note: this uses copy on write for the columns to be memory
        #- efficient while still letting us modify a column if needed
        zcat = zcat.copy()

        # needed to have the same ordering both in zcat and newzcat
        # to ensure consistent use of masks from np.isin()
        zcat.sort(keys='TARGETID')
        newzcat.sort(keys='TARGETID') 
        
        #- targets that are in both zcat and newzcat
        repeats = np.isin(zcat['TARGETID'], newzcat['TARGETID'])

        #- update numobs in both zcat and newzcat
        ii = np.isin(newzcat['TARGETID'], zcat['TARGETID'][repeats])
        orig_numobs = zcat['NUMOBS'][repeats].copy()
        new_numobs = newzcat['NUMOBS'][ii].copy()
        zcat['NUMOBS'][repeats] += new_numobs
        newzcat['NUMOBS'][ii] += orig_numobs

        #- replace only repeats that had ZWARN flags in original zcat
        #- replace in new
        replace = repeats & (zcat['ZWARN'] != 0)
        jj = np.isin(newzcat['TARGETID'], zcat['TARGETID'][replace])
        zcat[replace] = newzcat[jj]

        #- trim newzcat to ones that shouldn't override original zcat
        discard = np.isin(newzcat['TARGETID'], zcat['TARGETID'])
        newzcat = newzcat[~discard]

        #- Should be non-overlapping now
        assert np.all(np.isin(zcat['TARGETID'], newzcat['TARGETID']) == False)

        #- merge them
        newzcat = vstack([zcat, newzcat])

    #- check for duplicates
    targetids = newzcat['TARGETID']
    assert len(np.unique(targetids)) == len(targetids)

    #- Metadata for header
    newzcat.meta['EXTNAME'] = 'ZCATALOG'
    
    #newzcat.sort(keys='TARGETID')

    log.info('{} QC done'.format(asctime()))
    return newzcat
