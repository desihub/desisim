'''
Code for quickly simulating the survey results given a mock catalog and
a list of tile epochs to observe.

Direclty Depends on the following desiproducts
     * targets.mtl 
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

import desitarget.mtl
from desisim.quickcat import quickcat

class _sim_setup(object):
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


        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.truthfile  = os.path.join(self.targets_path,'truth.fits')
        self.targetsfile = os.path.join(self.targets_path,'targets.fits')
        self.zcatfile = None
        
        self.tile_ids = []
        self.tilefiles = []
        self.mtl_epochs = []

def set_mtl_epochs(_setup, epochs_list = [0]):
    _setup.mtl_epochs = list(epochs_list)

def create_directories(_setup):
    if not os.path.exists(_setup.output_path):
        os.makedirs(_setup.output_path)
        
    if not os.path.exists(_setup.tmp_output_path):
        os.makedirs(_setup.tmp_output_path)

    if not os.path.exists(_setup.tmp_fiber_path):
        os.makedirs(_setup.tmp_fiber_path)

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
