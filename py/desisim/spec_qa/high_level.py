"""
desisim.spec_qa.high_level
==========================

Module to run high_level QA on a given DESI run
 Written by JXP on 3 Sep 2015
"""
from __future__ import print_function, absolute_import, division

import numpy as np
import sys, os, pdb, glob
import yaml

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from astropy.io import fits
from astropy.table import Table, vstack, hstack, MaskedColumn

from desispec.io import meta as dio_meta
from desispec.io import util as dio_util

from desisim.spec_qa import redshifts as dsqa_z

def get_meta():
    '''Get META data on production
    '''
    # Dummy for now
    meta = dict(SIMSPECV='9.999', SPECPROD=os.getenv('SPECPROD'))
    return meta

def main():
    '''Runs the process
    '''
    # Check environmental variables are set
    assert 'DESI_SPECTRO_DATA' in os.environ, 'Missing $DESI_SPECTRO_DATA environment variable'
    assert 'SPECPROD' in os.environ, 'Missing $SPECPROD environment variable'
    assert 'DESI_SPECTRO_REDUX' in os.environ, 'Missing $DESI_SPECTRO_REDUX environment variable'

    # Grab nights in data redux
    nights = glob.glob(dio_meta.specprod_root()+'/exposures/*')
    if len(nights) == 0:
        raise ValueError('No nights in exposures!')

    # Build up list of fibermap files
    fibermap_files = []
    for night in nights:
        onight = night[night.rfind('/'):]
        files = glob.glob(dio_meta.rawdata_root()+'/'+onight+'/fibermap*')
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
    simz_tab.meta = meta
    simz_tab.write(simtab_fil,overwrite=True)

    # Summary stats
    summ_file = dio_meta.specprod_root()+'/QA/sim_z_summ.yaml'
    dio_util.makepath(summ_file)
    summ_dict = dsqa_z.summ_stats(simz_tab)
    # Write
    with open(summ_file, 'w') as outfile:
        outfile.write( yaml.dump(meta))#, default_flow_style=True) )
        outfile.write( yaml.dump(summ_dict, default_flow_style=False) )
    #import pdb
    #pdb.set_trace()


    '''
    # Reorder + cut
    summ_tab=full_summ_tab['OBJTYPE', 'NTARG', 'N_SURVEY', 'EFF', 'MED_DZ', 'CAT_RATE', 'REQ_FINAL']
    # Write
    summ_tab.meta = meta
    summ_tab.write(summ_file,format='ascii.ecsv',
        formats=dict(MED_DZ='%8.6f',EFF='%5.3f',CAT_RATE='%6.4f'))#,clobber=True)
    '''

    # QA Figures
    fig_file = dio_meta.specprod_root()+'/QA/sim_z.pdf'
    dio_util.makepath(fig_file)
    pp = PdfPages(fig_file)
    # Summ
    dsqa_z.summ_fig(simz_tab, summ_dict, meta, pp=pp)
    for objtype in ['ELG','LRG', 'QSO_T', 'QSO_L']:
        dsqa_z.obj_fig(simz_tab, objtype, summ_dict, pp=pp)
    # All done
    pp.close()


    # Write
    #pdb.set_trace()
