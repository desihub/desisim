'''
desisim.quickcat
================

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
from pkg_resources import resource_filename
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

from desitarget.targets import desi_mask
from desitarget.targets import bgs_mask
from desitarget.targets import mws_mask

from desiutil.log import get_logger
log = get_logger()

#- redshift errors and zwarn fractions from DESI-1657
#- sigmav = c sigmaz / (1+z)
_sigma_v = {
    'ELG': 19.,
    'LRG': 40.,
    'BGS': 13.,
    'QSO': 423.,
    'STAR': 18.,
    'SKY': 9999,      #- meaningless
    'UNKNOWN': 9999,  #- meaningless
}

_zwarn_fraction = {
    'ELG': 0.14,       # 1 - 4303/5000
    'LRG': 0.015,      # 1 - 4921/5000
    'QSO': 0.18,       # 1 - 4094/5000
    'BGS': 0.01,
    'STAR': 0.05,
    'SKY': 1.0,
    'UNKNOWN': 1.0,
}


_cata_fail_fraction = {
   # Catastrophic error fractions from redmonster on oak (ELG, LRG, QSO)
    'ELG': 0.08,
    'LRG': 0.013,
    'QSO': 0.20,
    'BGS': 0.,
    'STAR': 0.,
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

    # airmass
    v = obsconditions['AIRMASS'] - np.mean(obsconditions['AIRMASS'])
    pv  = p_v[0] + p_v[1] * v + p_v[2] * (v**2. - np.mean(v**2))

    # ebmv
    w = obsconditions['EBMV'] - np.mean(obsconditions['EBMV'])
    pw = p_w[0] + p_w[1] * w + p_w[2] * (w**2 - np.mean(w**2))

    # seeing
    x = obsconditions['SEEING'] - np.mean(obsconditions['SEEING'])
    px = p_x[0] + p_x[1]*x + p_x[2] * (x**2 - np.mean(x**2))

    # transparency
    y = obsconditions['LINTRANS'] - np.mean(obsconditions['LINTRANS'])
    py = p_y[0] + p_y[1]*y + p_y[2] * (y**2 - np.mean(y**2))

    # moon illumination fraction
    z = obsconditions['MOONFRAC'] - np.mean(obsconditions['MOONFRAC'])
    pz = p_z[0] + p_z[1]*z + p_z[2] * (z**2 - np.mean(z**2))

    #- if moon is down phase doesn't matter
    pz[obsconditions['MOONALT'] < 0] = 1.0

    pr = 1.0 + np.random.normal(size=len(z), scale=sigma_r)

    #- this correction factor can be greater than 1, but not less than 0
    pobs = (pv * pw * px * py * pz * pr).clip(min=0.0)

    return pobs


def get_redshift_efficiency(simtype, targets, truth, targets_in_tile, obsconditions=None):
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

    Returns:
        tuple of arrays (observed, p) both with same length as targets

        observed: boolean array of whether the target was observed in these tiles

        p: probability to get this redshift right
    """
    targetid = targets['TARGETID']
    n = len(targetid)

    if (simtype == 'ELG'):
        # Read the model OII flux threshold (FDR fig 7.12 modified to fit redmonster efficiency on OAK)
        filename = resource_filename('desisim', 'data/quickcat_elg_oii_flux_threshold.txt')
        fdr_z, modified_fdr_oii_flux_threshold = np.loadtxt(filename, unpack=True)

        # Get OIIflux from truth
        try:
            true_oii_flux = truth['OIIFLUX']
        except:
            raise Exception('Missing OII flux information to estimate redshift efficiency for ELGs')

        # Compute OII flux thresholds for truez
        oii_flux_threshold = np.interp(truth['TRUEZ'],fdr_z,modified_fdr_oii_flux_threshold)
        assert (oii_flux_threshold.size == true_oii_flux.size),"oii_flux_threshold and true_oii_flux should have the same size"

        # efficiency is modeled as a function of flux_OII/f_OII_threshold(z) and an arbitrary sigma_fudge
        sigma_fudge = 1.0
        max_efficiency = 1.0
        simulated_eff = eff_model(true_oii_flux/oii_flux_threshold,sigma_fudge,max_efficiency)

    elif(simtype == 'LRG'):
        # Read the model rmag efficiency
        filename = resource_filename('desisim', 'data/quickcat_lrg_rmag_eff.txt')
        magr, magr_eff = np.loadtxt(filename, unpack=True)

        # Get Rflux from truth
        try:
            true_rflux = targets['DECAM_FLUX'][:,2]
        except:
            raise Exception('Missing Rmag information to estimate redshift efficiency for LRGs')

        r_mag = 22.5 - 2.5*np.log10(true_rflux)

        mean_eff_mag=np.interp(r_mag,magr,magr_eff)
        fudge=0.002
        max_efficiency = 0.98
        simulated_eff = max_efficiency*mean_eff_mag*(1.+fudge*np.random.normal(size=mean_eff_mag.size))
        simulated_eff[np.where(simulated_eff>max_efficiency)]=max_efficiency

    elif(simtype == 'QSO'):
        # Read the model gmag threshold
        filename = resource_filename('desisim', 'data/quickcat_qso_gmag_threshold.txt')
        zc, qso_gmag_threshold_vs_z = np.loadtxt(filename, unpack=True)

        # Get Gflux from truth
        try:
            true_gmag = targets['DECAM_FLUX'][:,1]
        except:
            raise Exception('Missing Gmag information to estimate redshift efficiency for QSOs')

        # Computes QSO mag thresholds for truez
        qso_gmag_threshold=np.interp(truth['TRUEZ'],zc,qso_gmag_threshold_vs_z)
        assert (qso_gmag_threshold.size == true_gmag.size),"qso_gmag_threshold and true_gmag should have the same size"

        # Computes G flux
        qso_true_normed_flux = 10**(-0.4*(true_gmag-qso_gmag_threshold))

        #model effificieny for QSO:
        sigma_fudge = 0.5
        max_efficiency = 0.95
        simulated_eff = eff_model(qso_true_normed_flux,sigma_fudge,max_efficiency)

    elif simtype == 'BGS':
        simulated_eff = 0.98 * np.ones(n)

    elif simtype == 'MWS':
        simulated_eff = 0.98 * np.ones(n)

    else:
        default_zeff = 0.98
        log.warning('using default redshift efficiency of {} for {}'.format(default_zeff, simtype))
        simulated_eff = default_zeff * np.ones(n)

    if (obsconditions is None) and (truth['OIIFLUX'] is None) and (targets['DECAM_FLUX'] is None):
        raise Exception('Missing obsconditions and flux information to estimate redshift efficiency')

    #- Get the corrections for observing conditions per tile, then
    #- correct targets on those tiles.  Parameterize in terms of failure
    #- rate instead of success rate to handle bookkeeping of targets that
    #- are observed on more than one tile.
    #- NOTE: this still isn't quite right since multiple observations will
    #- be simultaneously fit instead of just taking whichever individual one
    #- succeeds.

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
def eff_model(x, sigma, max_efficiency):
    return 0.5*max_efficiency*(1.+sp.erf((x-1)/(np.sqrt(2.)*sigma)))

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

def get_observed_redshifts(targets, truth, targets_in_tile, obsconditions):
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

    Returns:
        tuple of (zout, zerr, zwarn)
    """

    simtype = get_simtype(np.char.strip(truth['TRUESPECTYPE']), targets['DESI_TARGET'], targets['BGS_TARGET'], targets['MWS_TARGET'])
    truez = truth['TRUEZ']
    targetid = truth['TARGETID']

    zout = truez.copy()
    zerr = np.zeros(len(truez), dtype=np.float32)
    zwarn = np.zeros(len(truez), dtype=np.int32)

    objtypes = list(set(simtype))
    n_tiles = len(np.unique(obsconditions['TILEID']))

    if(n_tiles!=len(targets_in_tile)):
        raise ValueError('Number of obsconditions {} != len(targets_in_tile) {}'.format(n_tiles, len(targets_in_tile)))

    for objtype in objtypes:
        if objtype in _sigma_v.keys():
            ii = (simtype == objtype)
            n = np.count_nonzero(ii)

            # Error model for ELGs
            if (objtype =='ELG'):
                filename = resource_filename('desisim', 'data/quickcat_elg_oii_errz.txt')
                oii, errz_oii = np.loadtxt(filename, unpack=True)
                try:
                    true_oii_flux = truth['OIIFLUX'][ii]
                except:
                    raise Exception('Missing OII flux information to estimate redshift error for ELGs')

                mean_err_oii = np.interp(true_oii_flux,oii,errz_oii)
                zerr[ii] = mean_err_oii*(1.+truez[ii])
                zout[ii] += np.random.normal(scale=zerr[ii])

            # Error model for LRGs
            elif (objtype == 'LRG'):
                redbins=np.linspace(0.55,1.1,5)
                mag = np.linspace(20.5,23.,50)
                try:
                    true_magr=targets['DECAM_FLUX'][ii,2]
                except:
                    raise Exception('Missing DECAM r flux information to estimate redshift error for LRGs')

                coefs = [[9.46882282e-05,-1.87022383e-03],[6.14601021e-05,-1.17406643e-03],\
                             [8.85342362e-05,-1.76079966e-03],[6.96202042e-05,-1.38632104e-03]]

                '''
                Coefs of linear fit of LRG zdc1 redmonster error as a function of Rmag in 4 bins of redshift: (z_low,coef0,coef1)
                0.55 9.4688228224e-05 -0.00187022382514
                0.6875 6.14601021052e-05 -0.00117406643273
                0.825 8.85342361594e-05 -0.00176079966322
                0.9625 6.96202042482e-05 -0.00138632103551
                '''

                # for each redshift bin, select the corresponding coefs
                zerr_tmp = np.zeros(len(truez[ii]))
                for i in range(redbins.size-1):
                    index0, = np.where((truez[ii]>=redbins[i]) & (truez[ii]<redbins[i+1]))
                    if (i==0):
                        index1, = np.where(truez[ii]<redbins[0])
                        index = np.concatenate((index0,index1))
                    elif (i==(redbins.size-2)):
                        index1, = np.where(truez[ii]>=redbins[-1])
                        index = np.concatenate((index0,index1))
                    else:
                        index=index0

                    # Find mean error at true mag
                    pol = np.poly1d(coefs[i])
                    mean_err_mag=np.interp(true_magr[index],mag,pol(mag))

                    # Computes output error and redshift
                    zerr_tmp[index] = mean_err_mag
                zerr[ii]=zerr_tmp*(1.+truez[ii])
                zout[ii] += np.random.normal(scale=zerr[ii])


            # Error model for QSOs
            elif (objtype == 'QSO'):
                redbins = np.linspace(0.5,3.5,7)
                mag = np.linspace(21.,23.,50)

                try:
                    true_magg=targets['DECAM_FLUX'][ii,1]
                except:
                    raise Exception('Missing DECAM g flux information to estimate redshift error for QSOs')

                coefs = [[0.000156950059747,-0.00320719603886],[0.000461779391179,-0.00924485142818],\
                             [0.000458672517009,-0.0091254038977],[0.000461427968475,-0.00923812594293],\
                             [0.000312919487343,-0.00618137905849],[0.000219438845624,-0.00423782927109]]
                '''
                Coefs of linear fit of QSO zdc1 redmonster error as a function of Gmag in 6 bins of redshift: (z_low,coef0,coef1)
                0.5 0.000156950059747 -0.00320719603886
                1.0 0.000461779391179 -0.00924485142818
                1.5 0.000458672517009 -0.0091254038977
                2.0 0.000461427968475 -0.00923812594293
                2.5 0.000312919487343 -0.00618137905849
                3.0 0.000219438845624 -0.00423782927109
                '''

                # for each redshift bin, select the corresponding coefs
                zerr_tmp = np.zeros(len(truez[ii]))
                for i in range(redbins.size-1):
                    index0, = np.where((truez[ii]>=redbins[i]) & (truez[ii]<redbins[i+1]))
                    if (i==0):
                        index1, = np.where(truez[ii]<redbins[0])
                        index = np.concatenate((index0,index1))
                    elif (i==(redbins.size-2)):
                        index1, = np.where(truez[ii]>=redbins[-1])
                        index = np.concatenate((index0,index1))
                    else:
                        index=index0
                    # Find mean error at true mag
                    pol = np.poly1d(coefs[i])
                    mean_err_mag=np.interp(true_magg[index],mag,pol(mag))

                    # Computes output error and redshift
                    zerr_tmp[index] = mean_err_mag
                zerr[ii]=zerr_tmp*(1.+truez[ii])
                zout[ii] += np.random.normal(scale=zerr[ii])

            else:
                zerr[ii] = _sigma_v[objtype] * (1+truez[ii]) / c
                zout[ii] += np.random.normal(scale=zerr[ii])

            # Set ZWARN flags for some targets
            # the redshift efficiency only sets warning, but does not impact
            # the redshift value and its error.
            was_observed, goodz_prob = get_redshift_efficiency(
                objtype, targets[ii], truth[ii], targets_in_tile,
                obsconditions=obsconditions)

            assert len(was_observed) == n
            assert len(goodz_prob) == n
            r = np.random.random(len(was_observed))
            zwarn[ii] = 4 * (r > goodz_prob) * was_observed

            # Add fraction of catastrophic failures (zwarn=0 but wrong z)
            nzwarnzero = np.count_nonzero(zwarn[ii][was_observed] == 0)
            num_cata = np.random.poisson(_cata_fail_fraction[objtype] * nzwarnzero)
            if (objtype == 'ELG'): zlim=[0.6,1.7]
            elif (objtype == 'LRG'): zlim=[0.5,1.1]
            elif (objtype == 'QSO'): zlim=[0.5,3.5]
            if num_cata > 0:
                #- tmp = boolean array for all targets, flagging only those
                #- that are of this simtype and were observed this epoch
                tmp = np.zeros(len(ii), dtype=bool)
                tmp[ii] = was_observed
                kk, = np.where((zwarn==0) & tmp)
                index = np.random.choice(kk, size=num_cata, replace=False)
                assert np.all(np.in1d(index, np.where(ii)[0]))
                assert np.all(zwarn[index] == 0)

                zout[index] = np.random.uniform(zlim[0],zlim[1],len(index))

        else:
            msg = 'No redshift efficiency model for {}; using true z\n'.format(objtype) + \
                  'Known types are {}'.format(list(_sigma_v.keys()))
            log.warning(msg)

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
    ii = np.in1d(tiles['TILEID'], tileids)

    tiles = tiles[ii]
    assert len(tiles) == len(tileids)

    #- Sort tiles to match order of tileids
    i = np.argsort(tileids)
    j = np.argsort(tiles['TILEID'])
    k = np.argsort(i)

    tiles = tiles[j[k]]
    assert np.all(tiles['TILEID'] == tileids)

    #- fix type bug after reading desi-tiles.fits
    if tiles['OBSCONDITIONS'].dtype == np.float64:
        tiles = Table(tiles)
        tiles.replace_column('OBSCONDITIONS', tiles['OBSCONDITIONS'].astype(int))

    n = len(tileids)

    obsconditions = Table()
    obsconditions['TILEID'] = tileids
    obsconditions['AIRMASS'] = tiles['AIRMASS']
    obsconditions['EBMV'] = tiles['EBV_MED']
    obsconditions['LINTRANS'] = np.ones(n)
    obsconditions['SEEING'] = np.ones(n) * 1.1

    #- Add lunar conditions, defaulting to dark time
    from desitarget import obsconditions as obsbits
    obsconditions['MOONFRAC'] = np.zeros(n)
    obsconditions['MOONALT'] = -20.0 * np.ones(n)
    obsconditions['MOONDIST'] = 180.0 * np.ones(n)

    ii = (tiles['OBSCONDITIONS'] & obsbits.GRAY) != 0
    obsconditions['MOONFRAC'][ii] = 0.1
    obsconditions['MOONALT'][ii] = 10.0
    obsconditions['MOONDIST'][ii] = 60.0

    ii = (tiles['OBSCONDITIONS'] & obsbits.BRIGHT) != 0
    obsconditions['MOONFRAC'][ii] = 0.7
    obsconditions['MOONALT'][ii] = 60.0
    obsconditions['MOONDIST'][ii] = 50.0

    return obsconditions

def quickcat(tilefiles, targets, truth, zcat=None, obsconditions=None, perfect=False):
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
    print('{} QC Reading {} tiles'.format(asctime(), len(tilefiles)))
    nobs = Counter()
    targets_in_tile = {}
    tileids = list()
    for infile in tilefiles:
        fibassign, header = fits.getdata(infile, 'FIBER_ASSIGNMENTS', header=True)
        tile_id = header['TILEID']
        tileids.append(tile_id)

        ii = (fibassign['TARGETID'] != -1)  #- targets with assignments
        nobs.update(fibassign['TARGETID'][ii])
        targets_in_tile[tile_id] = fibassign['TARGETID'][ii]

    #- Trim obsconditions to just the tiles that were observed
    if obsconditions is not None:
        ii = np.in1d(obsconditions['TILEID'], tileids)
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
    print('{} QC Trimming truth to just observed targets'.format(asctime()))
    obs_targetids = np.array(list(nobs.keys()))
    iiobs = np.in1d(truth['TARGETID'], obs_targetids)
    truth = truth[iiobs]
    targets = targets[iiobs]

    #- Construct initial new z catalog
    print('{} QC Constructing new redshift catalog'.format(asctime()))
    newzcat = Table()
    newzcat['TARGETID'] = truth['TARGETID']
    if 'BRICKNAME' in truth.dtype.names:
        newzcat['BRICKNAME'] = truth['BRICKNAME']
    else:
        newzcat['BRICKNAME'] = np.zeros(len(truth), dtype=(str, 8))

    #- Copy TRUETYPE -> SPECTYPE so that we can change without altering original
    newzcat['SPECTYPE'] = truth['TRUESPECTYPE'].copy()

    #- Add ZERR and ZWARN
    print('{} QC Adding ZERR and ZWARN'.format(asctime()))
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
    print('{} QC Adding NUMOBS column'.format(asctime()))
    newzcat.add_column(Column(name='NUMOBS', length=nz, dtype=np.int32))
    for i in range(nz):
        newzcat['NUMOBS'][i] = nobs[newzcat['TARGETID'][i]]

    #- Merge previous zcat with newzcat
    print('{} QC Merging previous zcat'.format(asctime()))
    if zcat is not None:
        #- don't modify original
        #- Note: this uses copy on write for the columns to be memory
        #- efficient while still letting us modify a column if needed
        zcat = zcat.copy()

        #- targets that are in both zcat and newzcat
        repeats = np.in1d(zcat['TARGETID'], newzcat['TARGETID'])

        #- update numobs in both zcat and newzcat
        ii = np.in1d(newzcat['TARGETID'], zcat['TARGETID'][repeats])
        orig_numobs = zcat['NUMOBS'][repeats].copy()
        new_numobs = newzcat['NUMOBS'][ii].copy()
        zcat['NUMOBS'][repeats] += new_numobs
        newzcat['NUMOBS'][ii] += orig_numobs

        #- replace only repeats that had ZWARN flags in original zcat
        #- replace in new
        replace = repeats & (zcat['ZWARN'] != 0)
        jj = np.in1d(newzcat['TARGETID'], zcat['TARGETID'][replace])
        zcat[replace] = newzcat[jj]

        #- trim newzcat to ones that shouldn't override original zcat
        discard = np.in1d(newzcat['TARGETID'], zcat['TARGETID'])
        newzcat = newzcat[~discard]

        #- Should be non-overlapping now
        assert np.all(np.in1d(zcat['TARGETID'], newzcat['TARGETID']) == False)

        #- merge them
        newzcat = vstack([zcat, newzcat])

    #- check for duplicates
    targetids = newzcat['TARGETID']
    assert len(np.unique(targetids)) == len(targetids)

    #- Metadata for header
    newzcat.meta['EXTNAME'] = 'ZCATALOG'

    print('{} QC done'.format(asctime()))
    return newzcat
