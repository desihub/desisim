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
import sys

import astropy.constants
c = astropy.constants.c.to('km/s').value

from desitarget.targets import desi_mask
from desitarget.targets import bgs_mask
from desitarget.targets import mws_mask

#- redshift errors and zwarn fractions from DESI-1657
#- sigmav = c sigmaz / (1+z)
_sigma_v = {
    'ELG': 19.,
    'LRG': 40.,
    'QSO': 423.,
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

def get_redshift_efficiency(truetype, truez, targetid, targets_in_tile, obsconditions=None, flux=None):
    """
    Simple model to get the redshift effiency from the observational conditions or observed magnitudes+redshuft
    Args:
        truetype: string, could be 'ELG', 'LRG', 'QSO', 'STAR', 'SKY', or 'UNKNOWN'
        truez: array of true redshift values.
        targets_in_tile: dictionary. Keys correspond to tileids, its values are the 
            arrays of targetids observed in that tile.

        obsconditions: Table or ndarray with the observational conditions for every tile,
        including the following columns
            TILEID: array of tile IDs
            AIRMASS: array of airmass values on a tile
            EBMV: array of E(B-V) values on a tile
            LINTRANS: array of atmospheric transparency during spectro obs; floats [0-1]
            MOONFRAC: array of moonfraction values on a tile.
            SEEING: array of FWHM seeing during spectroscopic observation on a tile.
        flux: Dictionary including the relevant flux values (TBD) to diagnose redshift efficiency.
           
    Outputs:
    
        p: array marking the probability to get this redshift right. This must have the size of targetid.
    """

    n = len(targetid)
    p = np.ones(n)

    # reverse the dictionary targets in tile
    tiles_for_target = reverse_dictionary(targets_in_tile)
    for i in range(n):
        t_id = targetid[i]
        if t_id not in tiles_for_target.keys():
            p[i] = 0.0

    if obsconditions is not None: # use this if we have obsconditions  information
        we_have_a_model = True    
        
        p_v = [1.0, 0.0, 0.0]
        p_w = [1.0, 0.0, 0.0]
        p_x = [1.0, 0.0, 0.0]
        p_y = [1.0, 0.0, 0.0]
        p_z = [1.0, 0.0, 0.0]
        sigma_r = 0.0
        p_total = 1.0
        if(truetype=='LRG'):
            p_v = [1.0, 0.15, 0.5]
            p_w = [1.0, 0.4, 0.0]
            p_x = [1.0, 0.06, 0.05]
            p_y = [1.0, 0.0, 0.08]
            p_z = [1.0, 0.0, 0.0]
            sigma_r = 0.02
            p_total = 0.95
        elif(truetype=='QSO'):
            p_v = [1.0, -0.2, 0.3]
            p_w = [1.0, -0.5, 0.6]
            p_x = [1.0, -0.1, -0.075]
            p_y = [1.0, -0.08, -0.04]
            p_z = [1.0, 0.0, 0.0]
            sigma_r = 0.05
            p_total = 0.90
        elif(truetype=='ELG'):
            p_v = [1.0, -0.1, -0.2]
            p_w = [1.0, 0.25, -0.75]
            p_x = [1.0, 0.0, 0.05]
            p_y = [1.0, 0.2, 0.1]
            p_z = [1.0, -10.0, 300.0]
            sigma_r = 0.02
            p_total = 0.95
        else:
            print('WARNING in desisim.quickcat.get_redshift_efficiency()')
            print('\t We are not modelling yet the redshift efficiency for type: {}. Set to 1.0'.format(truetype))
            we_have_a_model = False


        if we_have_a_model:
            normal_values = np.random.normal(size=len(obsconditions['TILEID'])) # set of normal values across tiles
            for i in range(n):
                t_id = targetid[i]
                if t_id in tiles_for_target.keys():
                    tiles = tiles_for_target[t_id]
                    p_final = 0.0 # just in case a target is found in multiple tiles
                    for tileid in tiles:            
                        ii = (obsconditions['TILEID'] == tileid)
                        
                        v = obsconditions['AIRMASS'][ii] - np.mean(obsconditions['AIRMASS'])
                        pv  = p_v[0] + p_v[1] * v + p_v[2] * (v**2 - np.mean(obsconditions['AIRMASS']**2))
                        
                        w = obsconditions['EBMV'][ii] - np.mean(obsconditions['EBMV'])
                        pw = p_w[0] + p_w[1] * w + p_w[2] * (w**2 - np.mean(obsconditions['EBMV']**2))
                        
                        x = obsconditions['SEEING'][ii] - np.mean(obsconditions['SEEING'])
                        px = p_x[0] + p_x[1]*x + p_x[2] * (x**2 - np.mean(obsconditions['SEEING']**2))
                        
                        y = obsconditions['LINTRANS'][ii] - np.mean(obsconditions['LINTRANS'])
                        py = p_y[0] + p_y[1]*y + p_y[2] * (y**2 - np.mean(obsconditions['LINTRANS']**2))
                        
                        z = obsconditions['MOONFRAC'][ii] - np.mean(obsconditions['MOONFRAC'])
                        pz = p_z[0] + p_z[1]*z + p_z[2] * (z**2 - np.mean(obsconditions['MOONFRAC']**2))
                        
                        pr = 1.0 + sigma_r * normal_values[ii]
                        
                        p[i] = p_total * pv * pw * px * py * pz * pr 
                        
                        if(p_final > p[i]): #select the best condition of all tiles
                            p[i] = p_final
                            p_final = p[i]

        
    if flux is not None : # use this if we have flux information
        # TODO. Christophe: this is the place to include your model, the goal is computing p_from_fluxes.
        # 'flux' is a dictionary. Add there all the things you need and we will figure it out upstream how
        # to get the information down here.
        for i in range(n):
            t_id = targetid[i]
            if t_id in tiles_for_target.keys():
                tiles = tiles_for_target[t_id]
                p_from_fluxes = 1.0
                p[i] *= p_from_fluxes
                    
    if (obsconditions is None) and (flux is None): 
        raise Exception('Missing obsconditions and flux information to estimate redshift efficiency')

    return p

def reverse_dictionary(a):
    """
    Inverts a dictionary mapping.
    Input.
        a: input dictionary.
    Output:
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

def get_observed_redshifts(truetype, truez, targetid, targets_in_tile, obsconditions=None, flux=None):
    """
    Returns observed z, zerr, zwarn arrays given true object types and redshifts

    Args:       
        truetype : array of ELG, LRG, QSO, STAR, SKY, or UNKNOWN
        truez: array of true redshifts
        targets_in_tile: Dictionary having as keys the tileid and the corresponding values the
            array of targetids observed in that tile.
        obsconditions: Dictionary with the observational conditions for every tile.
            It inclues at least the following keys>
           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
       flux: Dictionary including the relevant flux values (TBD) to diagnose redshift efficiency.
           
    Returns tuple of (zout, zerr, zwarn)

    """
    zout = truez.copy()
    zerr = np.zeros(len(truez), dtype=np.float32)
    zwarn = np.zeros(len(truez), dtype=np.int32)


    objtypes = list(set(truetype))
    n_tiles = len(obsconditions['TILEID'])
    
    if(n_tiles!=len(targets_in_tile.keys())):
        print('ERROR desisim.quickcat.get_observed_redshifts()')
        print('\t Number of obsconditions different from targets_in_tile.')
        sys.exit()


    for objtype in objtypes:
        if objtype in _sigma_v.keys():
            ii = (truetype == objtype)
            n = np.count_nonzero(ii)        
            zerr[ii] = _sigma_v[objtype] * (1+truez[ii]) / c
            zout[ii] += np.random.normal(scale=zerr[ii])

            # the redshift efficiency only sets warning, but does not impact the redshift value and its error.
            if (obsconditions is not None) or (flux is not None):
                p_obs = get_redshift_efficiency(objtype, truez[ii], targetid[ii], targets_in_tile, obsconditions=obsconditions, flux=flux)            
                z_eff = p_obs.copy()            
                r = np.random.random(n)
                jj = r > z_eff
                zwarn_type = np.zeros(n, dtype=np.int32)
                zwarn_type[jj] = 4
                zwarn[ii] = zwarn_type

            elif(obsconditions is None) and (flux is None):
                #- randomly select some objects to set zwarn
                num_zwarn = int(_zwarn_fraction[objtype] * n)
                if num_zwarn > 0:
                    jj = np.random.choice(np.where(ii)[0], size=num_zwarn, replace=False)
                    zwarn[jj] = 4
        else:
            print('WARNING desisim.quickcat.get_observed_redshifts()')
            print('\t We dont have a model for objtype {}. Simply assigning a redshift from the truth table.'.format(objtype))
            
            
    return zout, zerr, zwarn

def get_median_obsconditions(tileids):
    """
    Gets the observational conditions for a set of tiles.
    Args:
       tileids : list of tileids that were observed
    
    Outputs:
        obsconditions: Table with the observational conditions for every tile.
            It inclues at least the following columns
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

    #- fix type bug in desi-tiles.fits
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
    
    Optional:
        zcat: input zcatalog Table from previous observations
        obsconditions: Table or ndarray with observing conditions from surveysim
        perfect: if True, treat spectro pipeline as perfect with input=output,
            otherwise add noise and zwarn!=0 flags
        
    Returns:
        zcatalog astropy Table based upon input truth, plus ZERR, ZWARN,
        NUMOBS, and TYPE columns

    TODO: Add BGS, MWS support
    """
    #- convert to Table for easier manipulation
    truth = Table(truth)

    #- Count how many times each target was observed for this set of tiles
    ### print('Reading {} tiles'.format(len(obstiles)))
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

    #- Count how many times each target was observed in previous zcatalog
    #- NOTE: assumes that no tiles have been repeated
    if zcat is not None:
        ### print('Counting targets from previous zobs')
        for targetid, n in zip(zcat['TARGETID'], zcat['NUMOBS']):
            nobs[targetid] += n

    #- Trim truth down to just ones that have already been observed
    ### print('Trimming truth to just observed targets')
    obs_targetids = np.array(list(nobs.keys()))
    iiobs = np.in1d(truth['TARGETID'], obs_targetids)
    truth = truth[iiobs]
    targets = targets[iiobs]

    # TODO: add some logic to not reassign a redshift to something that already has one

    #- Construct initial new z catalog
    newzcat = Table()
    newzcat['TARGETID'] = truth['TARGETID']
    if 'BRICKNAME' in truth.dtype.names:
        newzcat['BRICKNAME'] = truth['BRICKNAME']
    else:
        newzcat['BRICKNAME'] = np.zeros(len(truth), dtype=(str, 8))

    #- Copy TRUEZ -> Z so that we can add errors without altering original
    newzcat['Z'] = truth['TRUEZ'].copy()
    newzcat['SPECTYPE'] = truth['TRUETYPE'].copy()

    #- Add numobs column
    ### print('Adding NUMOBS column')
    nz = len(newzcat)
    newzcat.add_column(Column(name='NUMOBS', length=nz, dtype=np.int32))
    for i in range(nz):
        newzcat['NUMOBS'][i] = nobs[newzcat['TARGETID'][i]]

    #- Add ZERR and ZWARN
    ### print('Adding ZERR and ZWARN')
    if not perfect:
        #- GALAXY -> ELG, LRG, BGS
        objtype = newzcat['SPECTYPE'].copy()
        isLRG = (objtype == 'GALAXY') & ((targets['DESI_TARGET'] & desi_mask.LRG) != 0)
        isELG = (objtype == 'GALAXY') & ((targets['DESI_TARGET'] & desi_mask.ELG) != 0)
        isBGS = (objtype == 'GALAXY') & ((targets['DESI_TARGET'] & bgs_mask.BGS_BRIGHT) != 0)
        isBGS |= (objtype == 'GALAXY') & ((targets['DESI_TARGET'] & bgs_mask.BGS_FAINT) != 0)
        objtype[isLRG] = 'LRG'
        objtype[isELG] = 'ELG'
        objtype[isBGS] = 'BGS'

        if obsconditions is None:
            obsconditions = get_median_obsconditions(tileids)

        #get the redshifts
        z, zerr, zwarn = get_observed_redshifts(objtype, newzcat['Z'], newzcat['TARGETID'], targets_in_tile, obsconditions)

        newzcat['Z'] = z  #- update with noisy redshift
    else:
        zerr = np.zeros(nz, dtype=np.float32)
        zwarn = np.zeros(nz, dtype=np.int32)

    newzcat['ZERR'] = zerr
    newzcat['ZWARN'] = zwarn

    #- Metadata for header
    newzcat.meta['EXTNAME'] = 'ZCATALOG'

    # TODO: add some logic to check that we don't have duplicates in newzcat

    return newzcat
