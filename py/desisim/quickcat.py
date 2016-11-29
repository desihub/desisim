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
from pkg_resources import resource_filename

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
import sys
import scipy.special as sp
import desisim
from desisim.targets import get_simtype

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


def get_redshift_efficiency(simtype, targets, truth, targets_in_tile, obsconditions=None):
    """
    Simple model to get the redshift effiency from the observational conditions or observed magnitudes+redshuft
    Args:
        simtype: ELG, LRG, QSO, STAR, BGS
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
           
    Outputs:
        p: array marking the probability to get this redshift right. This must have the size of targetid.
    """
    targetid = targets['TARGETID']
    n = len(targetid)
    p = np.ones(n)

    # reverse the dictionary targets in tile
    tiles_for_target = reverse_dictionary(targets_in_tile)
    for i in range(n):
        t_id = targetid[i]
        if t_id not in tiles_for_target.keys():
            p[i] = 0.0

    if obsconditions is not None: # use this if we have obsconditions information
        we_have_a_model = True    
        
        p_v = [1.0, 0.0, 0.0]
        p_w = [1.0, 0.0, 0.0]
        p_x = [1.0, 0.0, 0.0]
        p_y = [1.0, 0.0, 0.0]
        p_z = [1.0, 0.0, 0.0]
        p_v_1 = [1.0, 0.0, 0.0]
        p_w_1 = [1.0, 0.0, 0.0]
        p_x_1 = [1.0, 0.0, 0.0]
        p_y_1 = [1.0, 0.0, 0.0]
        p_z_1 = [1.0, 0.0, 0.0]
        sigma_r = 0.0
        p_total = 0.98
        sigma_r_1 = 0.0
        p_total_1 = 1.0
        if(simtype=='LRG'):
            p_v = [1.0, 0.15, -0.5]
            p_w = [1.0, 0.4, 0.0]
            p_x = [1.0, 0.06, 0.05]
            p_y = [1.0, 0.0, 0.08]
            p_z = [1.0, 0.0, 0.0]
            sigma_r = 0.02
            p_total = 0.95
        elif(simtype=='QSO'):
            p_v = [1.0, -0.1, 0.2]
            p_w = [1.0, -0.5, 0.6]
            p_x = [1.0, -0.1, -0.075]
            p_y = [1.0, -0.08, -0.04]
            p_z = [1.0, 0.0, 0.0]
            sigma_r = 0.05
            p_total = 0.90
        elif(simtype=='ELG'):
            p_v = [1.0, -0.1, -0.2]
            p_w = [1.0, 0.25, -0.75]
            p_x = [1.0, 0.0, 0.05]
            p_y = [1.0, 0.2, 0.1]
            p_z = [1.0, -10.0, 300.0]
            sigma_r = 0.075
            p_total = 0.8

            p_v_1 = [1.0, 0.45, 0.45]
            p_w_1 = [1.0, 1., -1.5]
            p_x_1 = [1.0, 0.1, 0.3]
            p_y_1 = [1.0, -0.2, -0.1]
            p_z_1 = [1.0, 40.0, -1200.0]
            sigma_r_1 = 0.1
            p_total_1 = 1.0


        else:
            print('WARNING in desisim.quickcat.get_redshift_efficiency()')
            print('\t We are not modelling yet the redshift efficiency for type: {}. Set to {}'.format(simtype, p_total))
            we_have_a_model = False


        if we_have_a_model:
            normal_values = np.random.normal(size=len(obsconditions['TILEID'])) # set of normal values across tiles
            print ("normal_values",normal_values)

            for key in obsconditions.keys():
                print (key,obsconditions[key])

            # Computes means of obsconditions
            airmass_mean = np.mean(obsconditions['AIRMASS'])
            airmass_mean2 = np.mean(obsconditions['AIRMASS']**2)
            print ("A, A2",airmass_mean,airmass_mean2)
            ebmv_mean = np.mean(obsconditions['EBMV'])
            ebmv_mean2 = np.mean(obsconditions['EBMV']**2)
            seeing_mean = np.mean(obsconditions['SEEING'])
            seeing_mean2 = np.mean(obsconditions['SEEING']**2)
            lintrans_mean = np.mean(obsconditions['LINTRANS'])
            lintrans_mean2 = np.mean(obsconditions['LINTRANS']**2)
            moonfrac_mean = np.mean(obsconditions['MOONFRAC'])
            moonfrac_mean2 = np.mean(obsconditions['MOONFRAC']**2)

            for i in range(n):
                t_id = targetid[i]
                if t_id in tiles_for_target.keys():
                    tiles = tiles_for_target[t_id]
                    p_final = 0.0 # just in case a target is found in multiple tiles

                    for tileid in tiles:            
                        ii = (obsconditions['TILEID'] == tileid)
                        '''
                        v = obsconditions['AIRMASS'][ii] - airmass_mean
                        pv  = p_v[0] + p_v[1] * v + p_v[2] * (v**2 - airmass_mean2) 
                        
                        w = obsconditions['EBMV'][ii] - ebmv_mean
                        pw = p_w[0] + p_w[1] * w + p_w[2] * (w**2 - ebmv_mean2)
                        
                        x = obsconditions['SEEING'][ii] - seeing_mean
                        px = p_x[0] + p_x[1]*x + p_x[2] * (x**2 - seeing_mean2)
                        
                        y = obsconditions['LINTRANS'][ii] - lintrans_mean
                        py = p_y[0] + p_y[1]*y + p_y[2] * (y**2 - lintrans_mean2)
                        
                        z = obsconditions['MOONFRAC'][ii] - moonfrac_mean
                        pz = p_z[0] + p_z[1]*z + p_z[2] * (z**2 - moonfrac_mean2)
                        
                        pr = 1.0 + sigma_r * normal_values[ii]
                        
                        p[i] = p_total * pv * pw * px * py * pz * pr 
                        '''

                        # airmass pivot chi = 1.1
                        airm_piv = 1.1
                        v = (obsconditions['AIRMASS'][ii] - airm_piv)
                        pv  = p_v_1[0] + p_v_1[1] * v + p_v_1[2] * (v**2.) 
                        

                        # ebmv pivot E(B-V) = 0.05 
                        ebmv_piv = 0.05
                        w = obsconditions['EBMV'][ii] - ebmv_piv                                                                                                                                          
                        pw = p_w_1[0] + p_w_1[1] * w + p_w_1[2] * (w**2)

                        # seeing pivot s = 1.1
                        s_piv = 1.1
                        x = obsconditions['SEEING'][ii] - s_piv
                        px = p_x_1[0] + p_x_1[1]*x + p_x_1[2] * (x**2)

                        # lintrans pivot = 0.
                        lintrans_piv = 0.0
                        y = obsconditions['LINTRANS'][ii] - lintrans_piv
                        py = p_y_1[0] + p_y_1[1]*y + p_y_1[2] * (y**2)

                        # moon frac piv = 0.
                        moonfrac_piv = 0.0
                        z = obsconditions['MOONFRAC'][ii] - moonfrac_piv
                        pz = p_z_1[0] + p_z_1[1]*z + p_z_1[2] * (z**2)

                        pr = 1.0 + sigma_r_1 * normal_values[ii]

                        p[i] = p_total_1 * pv * pw * px * py * pz * pr
                        
                        if(p_final > p[i]): #select the best condition of all tiles
                            p[i] = p_final
                            p_final = p[i]

    print ("p",p)

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
        
        # Take obsconditions into account                                                                                                                                                                 
        oii_flux_threshold*=p

        # efficiency is modeled as a function of flux_OII/f_OII_threshold(z) and an arbitrary sigma_fudge
        sigma_fudge = 1.0
        max_efficiency = 1.0
        simulated_eff = eff_model(true_oii_flux/oii_flux_threshold,sigma_fudge,max_efficiency)
        print ("simulated_eff",simulated_eff)

    if(simtype == 'LRG'):
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

    if(simtype == 'QSO'):
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

    #- this could be quite slow    
    for i in range(n):
        t_id = targetid[i]
        if t_id in tiles_for_target.keys():
            tiles = tiles_for_target[t_id]
            p_from_fluxes = simulated_eff[i]
            p[i] = p_from_fluxes

    if (obsconditions is None) and (truth['OIIFLUX'] is None) and (targets['DECAM_FLUX'] is None): 
        raise Exception('Missing obsconditions and flux information to estimate redshift efficiency')

    return p

# Efficiency model
def eff_model(x, sigma, max_efficiency):
    return 0.5*max_efficiency*(1.+sp.erf((x-1)/(np.sqrt(2.)*sigma)))

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

def get_observed_redshifts(targets, truth, targets_in_tile, obsconditions=None):
    """
    Returns observed z, zerr, zwarn arrays given true object types and redshifts

    Args:       
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
           
    Returns tuple of (zout, zerr, zwarn)

    """
    
    simtype = get_simtype(truth['TRUETYPE'], targets['DESI_TARGET'], targets['BGS_TARGET'], targets['MWS_TARGET'])
    truez = truth['TRUEZ']
    targetid = truth['TARGETID']

    zout = truez.copy()
    zerr = np.zeros(len(truez), dtype=np.float32)
    zwarn = np.zeros(len(truez), dtype=np.int32)

    objtypes = list(set(simtype))
    n_tiles = len(obsconditions['TILEID'])
    
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
                true_oii_flux = truth['OIIFLUX'][ii]
                mean_err_oii = np.interp(true_oii_flux,oii,errz_oii)

                sign = np.random.choice([-1,1],size=np.count_nonzero(ii))

                fudge=0.05
                zerr[ii] = mean_err_oii*(1.+fudge*np.random.normal(size=mean_err_oii.size))*(1.+truez[ii])
                zout[ii] += sign*zerr[ii]

            # Error model for LRGs
            elif (objtype == 'LRG'):
                redbins=np.linspace(0.55,1.1,5)
                mag = np.linspace(20.5,23.,50)
                true_magr=targets['DECAM_FLUX'][ii,2]
                coefs = [[9.46882282e-05,-1.87022383e-03],[6.14601021e-05,-1.17406643e-03],\
                             [8.85342362e-05,-1.76079966e-03],[6.96202042e-05,-1.38632104e-03]]

                '''
                Coefs of linear fit of LRG zdc1 redmonster error as a function of Rmag in 4 bins of redshift: (z_low,coef0,coef1)
                0.55 9.4688228224e-05 -0.00187022382514
                0.6875 6.14601021052e-05 -0.00117406643273
                0.825 8.85342361594e-05 -0.00176079966322
                0.9625 6.96202042482e-05 -0.00138632103551
                '''
                
                fudge=0.02 
                # for each redshift bin, select the corresponding coefs
                for i in range(redbins.size-1):
                    index0, = np.where((truez[ii]>redbins[i]) & (truez[ii]<redbins[i+1]))
                    if (i==0):
                        index1, = np.where(truez[ii]<redbins[0])
                        index = np.concatenate((index0,index1))
                    elif (i==(redbins.size-2)):
                        index1, = np.where(truez[ii]>redbins[-1])
                        index = np.concatenate((index0,index1))
                    else:
                        index=index0

                    # Find mean error at true mag 
                    pol = np.poly1d(coefs[i])
                    mean_err_mag=np.interp(true_magr[index],mag,pol(mag))
                    
                    # Computes output error and redshift
                    sign = np.random.choice([-1,1],size=len(truez[ii]))
                    zerr_tmp = np.zeros(len(truez[ii]))
                    zerr_tmp[index] = mean_err_mag*(1.+fudge*np.random.normal(size=mean_err_mag.size))*(1.+truez[ii][index])

                    zerr[ii]=zerr_tmp
                    zout[ii] += sign*zerr_tmp


            # Error model for QSOs
            elif (objtype == 'QSO'):
                redbins = np.linspace(0.5,3.5,7)
                mag = np.linspace(21.,23.,50)
                true_magg=targets['DECAM_FLUX'][ii,1]
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
                
                fudge=0.02
                # for each redshift bin, select the corresponding coefs
                for i in range(redbins.size-1):
                    index0, = np.where((truez[ii]>redbins[i]) & (truez[ii]<redbins[i+1]))
                    if (i==0): 
                        index1, = np.where(truez[ii]<redbins[0])
                        index = np.concatenate((index0,index1))
                    elif (i==(redbins.size-2)):
                        index1, = np.where(truez[ii]>redbins[-1])
                        index = np.concatenate((index0,index1))
                    else:
                        index=index0
                    # Find mean error at true mag    
                    pol = np.poly1d(coefs[i])
                    mean_err_mag=np.interp(true_magg[index],mag,pol(mag))
                    
                    # Computes output error and redshift 
                    sign = np.random.choice([-1,1],size=len(truez[ii]))
                    zerr_tmp = np.zeros(len(truez[ii]))
                    zerr_tmp[index] = mean_err_mag*(1.+fudge*np.random.normal(size=mean_err_mag.size))*(1.+truez[ii][index])
                    zerr[ii]=zerr_tmp
                    zout[ii] += sign*zerr_tmp



            
            else:
                zerr[ii] = _sigma_v[objtype] * (1+truez[ii]) / c
                zout[ii] += np.random.normal(scale=zerr[ii])


            # the redshift efficiency only sets warning, but does not impact the redshift value and its error.
            if (obsconditions is not None):
#                p_obs = get_redshift_efficiency(objtype, targets[ii], truth[ii], targets_in_tile, obsconditions=obsconditions)
                p_obs = get_redshift_efficiency(objtype, targets[ii], truth[ii], targets_in_tile, obsconditions=obsconditions)
                z_eff = p_obs.copy()
                r = np.random.random(n)
                jj = r > z_eff
                zwarn_type = np.zeros(n, dtype=np.int32)
                zwarn_type[jj] = 4
                zwarn[ii] = zwarn_type

                # Add fraction of catastrophic failures
                num_cata = int(_cata_fail_fraction[objtype] * n)
                if (objtype == 'ELG'): zlim=[0.6,1.7]
                elif (objtype == 'LRG'): zlim=[0.5,1.1]
                elif (objtype == 'QSO'): zlim=[0.5,3.5]
                if num_cata > 0:
                    jj, = np.where(zwarn[ii]==0)
                    index = np.random.choice(jj, size=num_cata, replace=False)
                    zout[index] = np.random.uniform(zlim[0],zlim[1],len(index)) 
                

            elif (obsconditions is None):
                #- randomly select some objects to set zwarn
                num_zwarn = int(_zwarn_fraction[objtype] * n)
                if num_zwarn > 0:
                    jj = np.random.choice(np.where(ii)[0], size=num_zwarn, replace=False)
                    zwarn[jj] = 4
        else:
            print('WARNING desisim.quickcat.get_observed_redshifts()')
            print('\tWe dont have a model for objtype {}. Simply assigning a redshift from the truth table.'.format(objtype))
            print('\tKnown types {}'.format(list(_sigma_v.keys())))

    return zout, zerr, zwarn


def get_obsconditions(tilefiles):
    """
    Gets the observational conditions for a set of tiles.
    Args:
       tilefiles : list of fiberassign tile files that were observed.
    Outputs:
        obsconditions: Dictionary with the observational conditions for every tile.
            It inclues at least the following keys
           'TILEID': array of tile IDs
           'AIRMASS': array of airmass values on a tile
           'EBMV': array of E(B-V) values on a tile
           'LINTRANS': array of atmospheric transparency during spectro obs; floats [0-1]
           'MOONFRAC': array of moonfraction values on a tile.
           'SEEING': array of FWHM seeing during spectroscopic observation on a tile.
     """
    n = len(tilefiles)

    obsconditions = {}
    
    obsconditions['TILEID'] =np.zeros(n, dtype='int')
    obsconditions['AIRMASS'] = np.ones(n)
    obsconditions['EBMV'] = np.ones(n)
    obsconditions['LINTRANS'] = np.ones(n)
    obsconditions['MOONFRAC'] = np.zeros(n)
    obsconditions['SEEING'] = np.ones(n)
        
    for i in range(n):
        infile = tilefiles[i]
        data = fits.open(infile)
        tile_id = data[1].header['TILEID']
        data.close()
        obsconditions['TILEID'][i] = tile_id
    return obsconditions


def quickcat(tilefiles, targets, truth, zcat=None, perfect=False):
    """
    Generates quick output zcatalog

    Args:
        tilefiles : list of fiberassign tile files that were observed
        targets : astropy Table of targets
        truth : astropy Table of input truth with columns TARGETID, TRUEZ, and TRUETYPE
        zcat (Optional): input zcatalog Table from previous observations
        perfect (Optional): if True, treat spectro pipeline as perfect with input=output,
            otherwise add noise and zwarn!=0 flags
        
    Returns:
        zcatalog astropy Table based upon input truth, plus ZERR, ZWARN,
        NUMOBS, and TYPE columns
    """
    #- convert to Table for easier manipulation
    truth = Table(truth)

    #- Count how many times each target was observed for this set of tiles
    ### print('Reading {} tiles'.format(len(obstiles)))
    nobs = Counter()
    targets_in_tile = {}
    for infile in tilefiles:
        fibassign = fits.getdata(infile, 'FIBER_ASSIGNMENTS')

        data = fits.open(infile)
        tile_id = data[1].header['TILEID']
        data.close()

        ii = (fibassign['TARGETID'] != -1)  #- targets with assignments
        nobs.update(fibassign['TARGETID'][ii])
        targets_in_tile[tile_id] = fibassign['TARGETID'][ii]
        
#        print (targets_in_tile)

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
    newzcat['SPECTYPE'] = truth['TRUETYPE'].copy()

    #- Add numobs column
    ### print('Adding NUMOBS column')
    nz = len(newzcat)
    newzcat.add_column(Column(name='NUMOBS', length=nz, dtype=np.int32))
    for i in range(nz):
        newzcat['NUMOBS'][i] = nobs[newzcat['TARGETID'][i]]

    #- Add ZERR and ZWARN
    ### print('Adding ZERR and ZWARN')
    if perfect:
        newzcat['Z'] = truth['TRUEZ'].copy()
        newzcat['ZERR'] = np.zeros(nz, dtype=np.float32)
        newzcat['ZWARN'] = np.zeros(nz, dtype=np.int32)
    else:
        # get the observational conditions for the current tilefiles
        obsconditions = get_obsconditions(tilefiles)

        # get the redshifts
        z, zerr, zwarn = get_observed_redshifts(targets, truth, targets_in_tile, obsconditions)
        newzcat['Z'] = z  #- update with noisy redshift
        newzcat['ZERR'] = zerr
        newzcat['ZWARN'] = zwarn

    #- Metadata for header
    newzcat.meta['EXTNAME'] = 'ZCATALOG'

    # TODO: add some logic to check that we don't have duplicates in newzcat

    return newzcat
