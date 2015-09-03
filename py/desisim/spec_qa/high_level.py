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

from desispec.io import meta as dio_meta
from desispec.io import util as dio_util

from desisim.spec_qa import redshifts as dsqa_z

def get_meta():
    '''Get META data on production
    '''
    # Dummy for now
    meta = dict(SIMSPECV='9.999')
    return meta

def main():
    '''Runs the process
    '''
    # Check environmental variables are set
    assert 'DESI_SPECTRO_DATA' in os.environ, 'Missing $DESI_SPECTRO_DATA environment variable'
    assert 'PRODNAME' in os.environ, 'Missing $PRODNAME environment variable'
    assert 'DESI_SPECTRO_REDUX' in os.environ, 'Missing $DESI_SPECTRO_REDUX environment variable'

    # Grab nights in data redux
    nights = glob.glob(dio_meta.specprod_root()+'/exposures/*')
    if len(nights) == 0:
        raise ValueError('No nights in exposures!')

    # Build up list of fibermap files
    fibermap_files = []
    for night in nights:
        onight = night[night.rfind('/'):]
        files = glob.glob(dio_meta.data_root()+'/'+onight+'/fibermap*')
        # 
        fibermap_files += files

    # Get list of zbest files
    zbest_files = glob.glob(dio_meta.specprod_root()+'/bricks/*/zbest*')
    if len(zbest_files) == 0:
        raise ValueError('No redshifts?!')

    # Meta
    meta = get_meta()

    # Load+write table
    simtab_fil = dio_meta.specprod_root()+'/QA/sim_z_table.fits'
    dio_util.makepath(simtab_fil)
    simz_tab = dsqa_z.load_z(fibermap_files, zbest_files)
    # Write
    simz_tab.meta = meta
    simz_tab.write(simtab_fil,overwrite=True)

    # Summary stats
    summ_file = dio_meta.specprod_root()+'/QA/sim_z_summ.ascii'
    dio_util.makepath(summ_file)
    summ_tab = dsqa_z.summ_stats(simz_tab)
    summ_tab.meta = meta
    summ_tab.write(summ_file,format='ascii.ecsv', 
        formats=dict(MED_DZ='%8.6f'))#,clobber=True)

    # QA Figures


    # Write 
    #pdb.set_trace()

