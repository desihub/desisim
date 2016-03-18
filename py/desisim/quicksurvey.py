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


        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.truthfile  = os.path.join(self.targets_path,'truth.fits')
        self.targetsfile = os.path.join(self.targets_path,'targets.fits')
        self.zcatfile = None

