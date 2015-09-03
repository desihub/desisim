"""
desisim.spec_qa.high_level
============

Module to run high_level QA on a given DESI run
 Written by JXP on 3 Sep 2015
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import sys, os, pdb, glob

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy.table import Table, vstack, hstack, MaskedColumn

from desispec import util
from desispec.io import read_fiberflat
from desispec import interpolation as desi_interp

from desispec.io import frame as desi_io_frame
from desispec.io import fibermap as desi_io_fmap
from desispec.io import read_sky
from desispec.io import meta as dio_meta
from desispec import sky as dspec_sky

from desispec.fiberflat import apply_fiberflat

def calc_stats(simz_tab, objtype):
    '''Calculate redshift statistics for a given objtype

	Parameters:
	-----------
	objtype: str
	  Object type, e.g. 'ELG', 'LRG'
    '''
   	# Cut on targets that were analyzed by RedMonster
    cut_tab = simz_tab[np.where(simz_tab['REDM_Z'].mask == False)[0]]

    # Init
    stat_dict = dict(OBJTYPE=objtype)

    # N targets 
    stat_dict['NTARG'] = len(np.where(simz_tab['OBJTYPE_1']==objtype)[0])

    # Number of objects with RedMonster
    gdo = np.where(cut_tab['OBJTYPE_1']==objtype)[0]
    stat_dict['N_Z'] = len(gdo) 

    # Redshift measured (includes catastrophics)
    idxz = np.where( (cut_tab['OBJTYPE_1']==objtype) & (cut_tab['REDM_ZWARN']==0))[0]
    mskz = np.array([True]*len(idxz))

    # Catastrophic failures
    dv = catastrophic_dv(objtype) # km/s
    cat = np.where(np.abs(cut_tab['REDM_Z'][idxz]-cut_tab['REDSHIFT'][idxz])*
    	3e5/(1+cut_tab['REDSHIFT'][idxz]) > dv)[0]
    stat_dict['NCAT'] = len(cat)
    stat_dict['CAT_RATE'] = len(cat)

    # Good redshifts
    mskz[cat] = False
    gdz = idxz[mskz]

    # Efficiency
    stat_dict['EFF'] = float(len(gdz))/float(stat_dict['N_Z'])

    # delta z
    dz = ((cut_tab['REDM_Z'][gdz]-cut_tab['REDSHIFT'][gdz])/ 
        (1+cut_tab['REDSHIFT'][gdz]))
    stat_dict['MEAN_DZ'] = np.mean(dz)
    stat_dict['MED_DZ'] = np.median(dz)
    stat_dict['RMS_DZ'] = np.std(dz)

    # Return
    return stat_dict

def catastrophic_dv(objtype):
	'''Pass back catastrophic velocity limit for given objtype
	From DESI document 318 (August 2015) in docdb

	Parameters:
	-----------
	objtype: str
	  Object type, e.g. 'ELG', 'LRG'
	'''
	cat_dict = dict(ELG=1000., LRG=1000.)
	#
	return cat_dict[objtype]

def load_z(fibermap_files, zbest_files, outfil=None):
    '''Load input and output redshift values for a set of exposures

    Parameters:
    -----------
    fibermap_files: list
      List of fibermap files
    zbest_files: list
      List of zbest output files from Redmonster
    outfil: str, optional
      Output file for the table

    Returns:
    -----------
    simz_tab: Table  
      Table of target info including Redmonster redshifts
    '''
    # imports

    # Init

    # Load up fibermap and simspec tables
    fbm_tabs = []
    sps_tabs = []
    for fibermap_file in fibermap_files:
        fbm_hdu = fits.open(fibermap_file)
        print('Reading: {:s}'.format(fibermap_file))
        # Load simspec
        simspec_fil = fibermap_file.replace('fibermap','simspec')
        sps_hdu = fits.open(simspec_fil)
        # Make Tables
        assert fbm_hdu[1].name == 'FIBERMAP'
        fbm_tabs.append(Table(fbm_hdu[1].data))
        assert sps_hdu[2].name == 'METADATA'
        sps_tabs.append(Table(sps_hdu[2].data))
    # Stack
    fbm_tab = vstack(fbm_tabs)
    sps_tab = vstack(sps_tabs)
    del fbm_tabs, sps_tabs

    # Drop to unique
    univ, uni_idx = np.unique(np.array(fbm_tab['TARGETID']),return_index=True)
    fbm_tab = fbm_tab[uni_idx]
    sps_tab = sps_tab[uni_idx]

    # Combine + Sort
    simz_tab = hstack([fbm_tab,sps_tab],join_type='exact')
    simz_tab.sort('TARGETID')
    nsim = len(simz_tab)

    # Load up zbest files
    zb_tabs = []
    for zbest_file in zbest_files:
        zb_hdu = fits.open(zbest_file)
        zb_tabs.append(Table(zb_hdu[1].data))
    # Stack
    zb_tab = vstack(zb_tabs)
    univ, uni_idx = np.unique(np.array(zb_tab['TARGETID']),return_index=True)
    zb_tab = zb_tab[uni_idx]

    # Match up
    sim_id = np.array(simz_tab['TARGETID'])
    z_id = np.array(zb_tab['TARGETID'])
    inz = np.in1d(z_id,sim_id,assume_unique=True)
    ins = np.in1d(sim_id,z_id,assume_unique=True)

    z_idx = np.arange(z_id.shape[0])[inz]
    sim_idx = np.arange(sim_id.shape[0])[ins]
    assert np.array_equal(sim_id[sim_idx],z_id[z_idx])

    # Fill up
    ztags = ['Z','ZERR','ZWARN','TYPE']
    new_tags = ['REDM_Z','REDM_ZERR','REDM_ZWARN','REDM_TYPE']
    new_clms = []
    mask = np.array([True]*nsim)
    mask[sim_idx] = False
    for kk,ztag in enumerate(ztags):
        # Generate a MaskedColumn
        new_clm  = MaskedColumn([zb_tab[ztag][z_idx[0]]]*nsim,
            name=new_tags[kk], mask=mask)
        # Fill
        new_clm[sim_idx] = zb_tab[ztag][z_idx]
        # Append
        new_clms.append(new_clm)
    # Add columns
    simz_tab.add_columns(new_clms)

    # Write?
    if outfil is not None:
    	simz_tab.write(outfil,overwrite=True)
    # Return
    return simz_tab # Masked Table

def obj_requirements(zstats, objtype):
	'''Assess where a given objtype passes the requirements
	Requirements from Doc 318 (August 2015)

	Parameters:
	-----------
	objtype: str
	  Object type, e.g. 'ELG', 'LRG'
	'''
	# 
	all_dict=dict(ELG={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.95})
	#
	req_dict = all_dict[objtype]

	tst_fail = ''
	passf = 'PASS'
	for key in req_dict.keys():
		if key in ['EFF']: # Greater than requirement
			if zstats[key] < req_dict[key]:
				passf = 'FAIL'
				tst_fail = tst_fail+key+'-'
		else:
			if zstats[key] > req_dict[key]:
				passf = 'FAIL'
				tst_fail = tst_fail+key+'-'
	if passf == 'FAIL':
		tst_fail = tst_fail[:-1]
		print('OBJ={:s} failed tests {:s}'.format(objtype,tst_fail))
	#
	return passf, tst_fail


def summ_stats(simz_tab, outfil=None):
    '''Generate summary stats 

    Parameters:
    -----------
    simz_tab: Table
      Table summarizing redshifts

    Returns:
    ---------
    summ_stats: Table
      Table of summary stats
  	'''
    otypes = ['ELG']  # WILL HAVE TO DEAL WITH QSO_TRACER vs QSO_LYA

    rows = []
    for otype in otypes:
    	# Calculate stats
    	stat_dict = calc_stats(simz_tab, otype)
    	# Check requirements
    	stat_dict['REQ'], tst_fail = obj_requirements(stat_dict,otype)
    	# Append
    	rows.append(stat_dict)

	# Generate Table
    stat_tab = Table(rows=rows)

    # Reorder
    stat_tab=stat_tab['OBJTYPE', 'NTARG', 'N_Z', 'MED_DZ', 'NCAT', 'REQ']
    # Return
    return stat_tab


