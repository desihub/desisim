'''
Code for quickly simulating the survey results given a mock catalog and
a list of tile epochs to observe.

Direclty Depends on the following desiproducts
     * desitarget.mtl 
     * desisim.quickcat
     * fiberassign
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import os
import shutil
import glob
import subprocess
from astropy.table import Table, Column
import os.path
from collections import Counter
from time import time, asctime

import desitarget.mtl
from desisim.quickcat import quickcat
from astropy.table import join
from desitarget.targets import desi_mask

class SimSetup(object):
    def __init__(self, **kwargs):
        if 'output_path' in kwargs.keys() :
            self.output_path = kwargs['output_path']        
        else:
            raise NameError('output_path was not set')

        if 'targets_path' in kwargs.keys() :
            self.targets_path = kwargs['targets_path']        
        else:
            raise NameError('targets_path was not set')

        if 'epochs_path' in kwargs.keys() :
            self.epochs_path = kwargs['epochs_path']        
        else:
            raise NameError('epochs_path was not set')

        if 'fiberassign_exec' in kwargs.keys() :
            self.fiberassign_exec = kwargs['fiberassign_exec']        
        else:
            raise NameError('fiberassign was not set')

        if 'template_fiberassign' in kwargs.keys() :
            self.template_fiberassign = kwargs['template_fiberassign']        
        else:
            raise NameError('template_fiberassign was not set')

        if 'n_epochs' in kwargs.keys() :
            self.n_epochs = kwargs['n_epochs']
        else:
            raise NameError('n_epochs was not set')
                
        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.truthfile  = os.path.join(self.targets_path,'truth.fits')
        self.targetsfile = os.path.join(self.targets_path,'targets.fits')
        self.zcat_file = None
        self.mtl_file = None

        self.tile_ids = []
        self.tilefiles = []
        self.mtl_epochs = []
        self.fiber_epochs = []
        self.epochs_list = range(self.n_epochs)

def reset_lists(_setup):
    _setup.tile_ids = []
    _setup.tilefiles = []
    _setup.mtl_epochs = []
    _setup.fiber_epochs = []    

def set_mtl_epochs(_setup, epochs_list = [0]):
    if hasattr(epochs_list, '__iter__'):
        _setup.mtl_epochs = list(epochs_list)
    else:
        _setup.mtl_epochs = list([epochs_list])

def set_fiber_epochs(_setup, epochs_list = [0]):
    if hasattr(epochs_list, '__iter__'):
        _setup.fiber_epochs = list(epochs_list)
    else:
        _setup.fiber_epochs = list([epochs_list])


def create_directories(_setup):
    if not os.path.exists(_setup.output_path):
        os.makedirs(_setup.output_path)
        
    if not os.path.exists(_setup.tmp_output_path):
        os.makedirs(_setup.tmp_output_path)

    if not os.path.exists(_setup.tmp_fiber_path):
        os.makedirs(_setup.tmp_fiber_path)

def cleanup_directories(_setup):
    if os.path.exists(_setup.tmp_output_path):
        shutil.rmtree(_setup.tmp_output_path)

def backup_epoch_data(_setup, epoch_id=0):
    backup_path = os.path.join(_setup.output_path, '{}'.format(epoch_id))

    # keep a copy of zcat.fits
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)        

    # keep a copy of mtl.fits 
    shutil.copy(_setup.mtl_file, backup_path)
    shutil.copy(_setup.zcat_file, backup_path)

    # keep a copy of all the fiberassign files 
    fiber_backup_path = os.path.join(backup_path, 'fiberassign')
    if not os.path.exists(fiber_backup_path):
        os.makedirs(fiber_backup_path)

    for tilefile in _setup.tilefiles:
        shutil.copy(tilefile, fiber_backup_path)


def create_surveyfile(_setup):
    # create survey list from mtl_epochs IDS
    surveyfile = os.path.join(_setup.tmp_output_path, "survey_list.txt")
    _setup.tile_ids = []
    for i in _setup.mtl_epochs:
        epochfile = os.path.join(_setup.epochs_path, "epoch{}.txt".format(i))        
        if os.path.exists(epochfile):
            _setup.tile_ids = np.append(_setup.tile_ids, np.loadtxt(epochfile))
        else:
            raise NameError('epochfile {} was not found'.format(epochfile))

    _setup.tile_ids = np.int_(_setup.tile_ids)
    np.savetxt(surveyfile, _setup.tile_ids, fmt='%d')
    print("{} tiles to be included in fiberassign".format(len(_setup.tile_ids)))

def create_fiberassign_input(_setup):
    params = ''.join(open(_setup.template_fiberassign).readlines())
    fx = open(os.path.join(_setup.tmp_output_path, 'fa_features.txt'), 'w')
    fx.write(params.format(inputdir = _setup.tmp_output_path, targetdir = _setup.targets_path))
    fx.close()


def simulate_epoch(_setup, perfect=False, epoch_id=0):
    #load truth / targets / zcat
    truth = Table.read(os.path.join(_setup.targets_path,'truth.fits'))
    targets = Table.read(os.path.join(_setup.targets_path,'targets.fits'))

    _setup.mtl_file = os.path.join(_setup.tmp_output_path, 'mtl.fits')    
    print("{} Starting MTL".format(asctime()))
    if _setup.zcat_file is None:
        mtl = desitarget.mtl.make_mtl(targets)
        mtl.write(_setup.mtl_file, overwrite=True)
    else:
        zcat = Table.read(_setup.zcat_file, format='fits')
        mtl = desitarget.mtl.make_mtl(targets, zcat)
        mtl.write(_setup.mtl_file, overwrite=True)
    print("{} Finished MTL".format(asctime()))

    # clean all fibermap fits files before running fiberassing
    tilefiles = sorted(glob.glob(_setup.tmp_fiber_path+'/tile*.fits'))
    if tilefiles:
        for tilefile in tilefiles:
            os.remove(tilefile)
            
    # launch fiberassign
    print("{} Launching fiberassign".format(asctime()))
    p = subprocess.call([_setup.fiberassign_exec, os.path.join(_setup.tmp_output_path, 'fa_features.txt')], stdout=subprocess.PIPE)
    print("{} Finished fiberassign".format(asctime()))


    #create a list of fibermap tiles to read and update zcat
    _setup.ids = []
    for i in _setup.fiber_epochs:
        epochfile = os.path.join(_setup.epochs_path, "epoch{}.txt".format(i))        
        _setup.ids = np.append(_setup.ids, np.loadtxt(epochfile))
    _setup.ids = np.int_(_setup.ids)


    _setup.tilefiles = []
    for i in _setup.ids:
        tilename = os.path.join(_setup.tmp_fiber_path, 'tile_%05d.fits'%(i))
        if os.path.isfile(tilename):
            _setup.tilefiles.append(tilename)
        else:
            print('Suggested but does not exist {}'.format(tilename))
    print("{} {} tiles to gather in zcat".format(asctime(), len(_setup.tilefiles)))
    
    # write the zcat
    if _setup.zcat_file is None:
        _setup.zcat_file = os.path.join(_setup.tmp_output_path, 'zcat.fits')
        zcat = quickcat(_setup.tilefiles, targets, truth, zcat=None, perfect=perfect)
        zcat.write(_setup.zcat_file, overwrite=True)
    else:
        _setup.zcat_file = os.path.join(_setup.tmp_output_path, 'zcat.fits')
        zcat = Table.read(_setup.zcat_file, format='fits')
        newzcat = quickcat(_setup.tilefiles, targets, truth, zcat=zcat, perfect=perfect)
        newzcat.write(_setup.zcat_file, format='fits', overwrite=True)
    print("{} Finished zcat".format(asctime()))

    # backup data into separate directories

def print_efficiency_stats(truth, mtl, zcat):
    print('Overall efficiency')
    tmp_init = join(mtl, truth, keys='TARGETID')    
    total = join(zcat, tmp_init, keys='TARGETID')

    true_types = ['LRG', 'ELG', 'QSO']
    zcat_types = ['GALAXY', 'GALAXY', 'QSO']
    
    for true_type, zcat_type in zip(true_types, zcat_types):
        i_initial = ((tmp_init['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (tmp_init['TRUETYPE'] == zcat_type)
        i_final = ((total['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (total['TYPE'] == zcat_type)             
        n_t = 1.0*len(total['TARGETID'][i_final])
        n_i = 1.0*len(tmp_init['TARGETID'][i_initial])
        print("\t {} fraction : {}".format(true_type, n_t/n_i))
    #print("\t TRUE:ZCAT\n\t {}\n".format(Counter(zip(total['DESI_TARGET'], total['TYPE']))))
    return

def print_numobs_stats(truth, targets, zcat):
    print('Target distributions')
    #- truth and targets are row-matched, so directly add columns instead of join
    for colname in targets.colnames:
        if colname not in truth.colnames:
            truth[colname] = targets[colname]

    xcat = join(zcat, truth, keys='TARGETID')

    for times_observed in range(1,5):
        print('\t Fraction (number) with exactly {} observations'.format(times_observed))
        ii = (xcat['NUMOBS']==times_observed)
        c = Counter(xcat['DESI_TARGET'][ii])

        total = np.sum(c.values())
        for k in c.keys():
            print("\t\t {}: {} ({} total)".format(desi_mask.names(k), c[k]/total, c[k]))
    return

def efficiency_numobs_stats(_setup, epoch_id=0):
    backup_path = os.path.join(_setup.output_path, '{}'.format(epoch_id))
    backup_path_0 = os.path.join(_setup.output_path, '{}'.format(0))

    truth = Table.read(os.path.join(_setup.targets_path,'truth.fits'))
    targets = Table.read(os.path.join(_setup.targets_path,'targets.fits'))
    mtl0 = Table.read(os.path.join(backup_path_0,'mtl.fits'))
    zcat = Table.read(os.path.join(backup_path, 'zcat.fits'))

    print_efficiency_stats(truth, mtl0, zcat)
    print_numobs_stats(truth, targets, zcat)
    return


def simulate_setup(_setup):
    create_directories(_setup)

    for epoch in _setup.epochs_list:
        print('Epoch {}'.format(epoch))
        set_mtl_epochs(_setup, epochs_list = _setup.epochs_list[epoch:])
        set_fiber_epochs(_setup, epochs_list = _setup.epochs_list[epoch])
        create_surveyfile(_setup)
        create_fiberassign_input(_setup)
        simulate_epoch(_setup, perfect=True, epoch_id = _setup.epochs_list[epoch])
        backup_epoch_data(_setup, epoch_id = _setup.epochs_list[epoch])
        reset_lists(_setup)

    cleanup_directories(_setup)


def summary_setup(_setup):    
    for epoch in _setup.epochs_list:
        print('Summary for Epoch {}'.format(epoch))
        efficiency_numobs_stats(_setup, epoch_id = epoch)
