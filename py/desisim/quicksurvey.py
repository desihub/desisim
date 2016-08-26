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
    """Setup to simulate the DESI survey
    
    Attributes:
        output_path (str): Path to write the outputs.x
        targets_path (str): Path where the files targets.fits can be found
        epochs_path (str): Path where the epoch files can be found.
        fiberassign_exec (str): Name of the fiberassign executable
        template_fiberassign (str): Filename of the template input for fiberassign
        n_epochs (int): number of epochs to be simulated.
                
    """
    def __init__(self, **kwargs):
        """Initializes all the paths, filenames and numbers describing DESI survey.
       
        Note:
           All parameters are required

        Args: 
            output_path (str): Path to write the outputs.x
            targets_path (str): Path where the files targets.fits can be found
            epochs_path (str): Path where the epoch files can be found.
            fiberassign_exec (str): Name of the fiberassign executable
            template_fiberassign (str): Filename of the template input for fiberassign
            n_epochs (int): number of epochs to be simulated.

        """
        if 'output_path' in kwargs:
            self.output_path = kwargs['output_path']        
        else:
            raise NameError('output_path was not set')

        if 'targets_path' in kwargs:
            self.targets_path = kwargs['targets_path']        
        else:
            raise NameError('targets_path was not set')

        if 'epochs_path' in kwargs:
            self.epochs_path = kwargs['epochs_path']        
        else:
            raise NameError('epochs_path was not set')

        if 'fiberassign_exec' in kwargs:
            self.fiberassign_exec = kwargs['fiberassign_exec']        
        else:
            raise NameError('fiberassign was not set')

        if 'template_fiberassign' in kwargs:
            self.template_fiberassign = kwargs['template_fiberassign']        
        else:
            raise NameError('template_fiberassign was not set')

        if 'n_epochs' in kwargs:
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
        self.epochs_list = list(range(self.n_epochs))

    def reset_lists(self):
        """Resets counters

        """
        self.tile_ids = []
        self.tilefiles = []
        self.mtl_epochs = []
        self.fiber_epochs = []    

    def set_mtl_epochs(self, epochs_list = [0]):
        """Sets the mtl_epochs list 

        Args:
            epochs_list (int, int list): optional variable listing integers IDs for the epochs.

        """
        if hasattr(epochs_list, '__iter__'):
            self.mtl_epochs = list(epochs_list)
        else:
            self.mtl_epochs = list([epochs_list])

    def set_fiber_epochs(self, epochs_list = [0]):
        """Sets the fiber_epochs list 

        Args:            
            epochs_list (int, int list): optional variable listing integers IDs for the epochs.

        """
        if hasattr(epochs_list, '__iter__'):
            self.fiber_epochs = list(epochs_list)
        else:
            self.fiber_epochs = list([epochs_list])


    def create_directories(self):
        """Creates output directories to store simulation results.

        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        if not os.path.exists(self.tmp_output_path):
            os.makedirs(self.tmp_output_path)

        if not os.path.exists(self.tmp_fiber_path):
            os.makedirs(self.tmp_fiber_path)
                
    def cleanup_directories(self):
        """Deletes files in the temporary output directory

        """
        if os.path.exists(self.tmp_output_path):
            shutil.rmtree(self.tmp_output_path)
            
    def backup_epoch_data(self, epoch_id=0):
        """Deletes files in the temporary output directory

        Args:
            epoch_id (int): Epoch's ID to backup/copy from the output directory.
        
        """
        backup_path = os.path.join(self.output_path, '{}'.format(epoch_id))

        # keep a copy of zcat.fits
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)        

        # keep a copy of mtl.fits 
        shutil.copy(self.mtl_file, backup_path)
        shutil.copy(self.zcat_file, backup_path)

        # keep a copy of all the fiberassign files 
        fiber_backup_path = os.path.join(backup_path, 'fiberassign')
        if not os.path.exists(fiber_backup_path):
            os.makedirs(fiber_backup_path)
        
        for tilefile in self.tilefiles:
            shutil.copy(tilefile, fiber_backup_path)


    def create_surveyfile(self):
        """Creates text file survey_list.txt to be used by fiberassign

        Notes: 
            The file is written to the temporary directory in self.tmp_output_path
        """

        # create survey list from mtl_epochs IDS
        surveyfile = os.path.join(self.tmp_output_path, "survey_list.txt")
        self.tile_ids = []
        for i in self.mtl_epochs:
            epochfile = os.path.join(self.epochs_path, "epoch{}.txt".format(i))        
            if os.path.exists(epochfile):
                self.tile_ids = np.append(self.tile_ids, np.loadtxt(epochfile))
            else:
                raise NameError('epochfile {} was not found'.format(epochfile))

        self.tile_ids = np.int_(self.tile_ids)
        np.savetxt(surveyfile, self.tile_ids, fmt='%d')
        print("{} tiles to be included in fiberassign".format(len(self.tile_ids)))

    def create_fiberassign_input(self):
        """Creates input files for fiberassign from the provided template 
        
        Notes:
            The template filename is in self.template_fiberassign    
        """
        params = ''.join(open(self.template_fiberassign).readlines())
        fx = open(os.path.join(self.tmp_output_path, 'fa_features.txt'), 'w')
        fx.write(params.format(inputdir = self.tmp_output_path, targetdir = self.targets_path))
        fx.close()
        
    def simulate_epoch(self, perfect=False, epoch_id=0, truth=None, targets=None, mtl=None, zcat=None):
        """Core routine simulating a DESI epoch, 

        Args:
            epoch_id (int): Integer ID of the epoch to be simulated
            perfect (boolean): Default: False. Selects whether how redshifts are taken from the truth file.
                True: redshifts are taken without any error from the truth file.
                False: redshifts include uncertainties.
            truth (Table): Truth data
            targets (Table): Targets data
            mtl (Table): Merged Targets List data
            zcat (Table): Redshift Catalog Data
        Notes:
            This routine simulates three steps:
            * Merged target list creation
            * Fiber allocation 
            * Redshift catalogue construction
        """
        #load truth / targets / zcat
        if truth is None:
            truth = Table.read(os.path.join(self.targets_path,'truth.fits'))
        if targets is None:
            targets = Table.read(os.path.join(self.targets_path,'targets.fits'))
            
        self.mtl_file = os.path.join(self.tmp_output_path, 'mtl.fits')    
        print("{} Starting MTL".format(asctime()))
        if self.zcat_file is None:            
            mtl = desitarget.mtl.make_mtl(targets)
            mtl.write(self.mtl_file, overwrite=True)
        else:
            #zcat = Table.read(self.zcat_file, format='fits')
            mtl = desitarget.mtl.make_mtl(targets, zcat)
            mtl.write(self.mtl_file, overwrite=True)
        print("{} Finished MTL".format(asctime()))

        # clean all fibermap fits files before running fiberassing
        tilefiles = sorted(glob.glob(self.tmp_fiber_path+'/tile*.fits'))
        if tilefiles:
            for tilefile in tilefiles:
                os.remove(tilefile)
            
        # launch fiberassign
        print("{} Launching fiberassign".format(asctime()))
        f = open('fiberassign.log','a')
        p = subprocess.call([self.fiberassign_exec, os.path.join(self.tmp_output_path, 'fa_features.txt')], stdout=f)# stdout=subprocess.PIPE)
        print("{} Finished fiberassign".format(asctime()))
        f.close()

        #create a list of fibermap tiles to read and update zcat
        # find first the set of tiles corresponding to this epoch
        self.tile_ids = []
        for i in self.fiber_epochs:
            epochfile = os.path.join(self.epochs_path, "epoch{}.txt".format(i))        
            self.tile_ids = np.append(self.tile_ids, np.loadtxt(epochfile))

        self.tile_ids = np.int_(self.tile_ids)

        # finally add the corresponding tiles to the list of fibermap files to read
        self.tilefiles = []
        for i in self.tile_ids:
            tilename = os.path.join(self.tmp_fiber_path, 'tile_%05d.fits'%(i))
            if os.path.isfile(tilename):
                self.tilefiles.append(tilename)
            else:
                print('Suggested but does not exist {}'.format(tilename))
        print("{} {} tiles to gather in zcat".format(asctime(), len(self.tilefiles)))
    
        # write the zcat, it uses the tilesfiles constructed in the last step
        if self.zcat_file is None:
            self.zcat_file = os.path.join(self.tmp_output_path, 'zcat.fits')
            newzcat = quickcat(self.tilefiles, targets, truth, zcat=None, perfect=perfect)
            newzcat.write(self.zcat_file, overwrite=True)
        else:
            self.zcat_file = os.path.join(self.tmp_output_path, 'zcat.fits')
            #zcat = Table.read(self.zcat_file, format='fits')
            newzcat = quickcat(self.tilefiles, targets, truth, zcat=zcat, perfect=perfect)
            newzcat.write(self.zcat_file, format='fits', overwrite=True)
        print("{} Finished zcat".format(asctime()))
        return truth, targets, mtl, newzcat


    def simulate(self):
        """Simulate the DESI setup described by a SimSteup object.
        """
        self.create_directories()

        truth=targets=mtl=zcat=None
        for epoch in self.epochs_list:
            print('Epoch {}'.format(epoch))

            self.set_mtl_epochs(epochs_list = self.epochs_list[epoch:])

            self.set_fiber_epochs(epochs_list = self.epochs_list[epoch])

            self.create_surveyfile()

            self.create_fiberassign_input()

            truth, targets, mtl, zcat = self.simulate_epoch(perfect=True, epoch_id = self.epochs_list[epoch], 
                                                            truth=truth, targets=targets, mtl=mtl, zcat=zcat)

            self.backup_epoch_data(epoch_id = self.epochs_list[epoch])

            self.reset_lists()

        self.cleanup_directories()




def print_efficiency_stats(truth, mtl_initial, zcat):
    print('Overall efficiency')
    tmp_init = join(mtl_initial, truth, keys='TARGETID')    
    total = join(zcat, tmp_init, keys='TARGETID')

    true_types = ['LRG', 'ELG', 'QSO']
    zcat_types = ['GALAXY', 'GALAXY', 'QSO']
    
    for true_type, zcat_type in zip(true_types, zcat_types):
        i_initial = ((tmp_init['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (tmp_init['TRUETYPE'] == zcat_type)
        i_final = ((total['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (total['SPECTYPE'] == zcat_type)             
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

        total = np.sum(list(c.values()))
        for k in c:
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


def summary_setup(_setup):    
    for epoch in _setup.epochs_list:
        print('Summary for Epoch {}'.format(epoch))
        efficiency_numobs_stats(_setup, epoch_id = epoch)
