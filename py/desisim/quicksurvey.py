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
    def __init__(self, output_path, targets_path,
        fiberassign_exec, template_fiberassign,
        start_epoch=0, n_epochs=None, epochs_path=None,
        obsconditions=None, fiberassign_dates=None):
        """Initializes all the paths, filenames and numbers describing DESI survey.
       
        Args: 
            output_path (str): Path to write the outputs.x
            targets_path (str): Path where the files targets.fits can be found
            fiberassign_exec (str): Name of the fiberassign executable
            template_fiberassign (str): Filename of the template input for fiberassign
            n_epochs (int): number of epochs to be simulated.

        Options:
            epochs_path (str): Path where the epoch files can be found.
                Required if obsconditions and fiberassign_dates is not given.
            obsconditions: file with observation conditions list from surveysim,
                (obslist_all.fits) or Table read from that file
            fiberassign_dates: file with list of dates to run fiberassign;
                one YEARMMDD or YEAR-MM-DD per line
        
        Notes:
            must provide epochs_path or (obsconditions and fiberassign_dates)
            obsconditions and epochs_path is also allowed, but there are no
                checks that the order of the tiles in epochs_path/epochs*.txt
                make any sense given the DATE-OBS in the obsconditions
        """
        self.output_path = output_path
        self.targets_path = targets_path
        self.fiberassign_exec = fiberassign_exec
        self.template_fiberassign = template_fiberassign

        if obsconditions is not None:
            if isinstance(obsconditions, (Table, np.ndarray)):
                self.obsconditions = Table(obsconditions)
            else:
                self.obsconditions = Table.read(obsconditions)
        else:
            self.obsconditions = obsconditions

        #- Add dates when fiberassign should be run; use YEAR-MM-DD strings
        #- to be able to compare to DATE-OBS YEAR-MM-DDThh:mm:ss.sss .
        if fiberassign_dates is not None:
            if obsconditions is None:
                raise ArgumentError('fiberassign_dates requires obsconditions')
                
            if epochs_path is not None:
                raise ArgumentError('epochs_path and fiberassign_dates are mutually exclusive')

            dates = list()
            with open(fiberassign_dates) as fx:
                for line in fx:
                    line = line.strip()
                    if line.startswith('#') or len(line) < 2:
                        continue
                    yearmmdd = line.replace('-', '')
                    year_mm_dd = yearmmdd[0:4]+'-'+yearmmdd[4:6]+'-'+yearmmdd[6:8]
                    dates.append(year_mm_dd)

            #- add pre- and post- dates for date range bookkeeping
            if dates[0] > min(self.obsconditions['DATE-OBS']):
                dates.insert(0, self.obsconditions['DATE-OBS'][0][0:10])

            dates.append('9999-99-99')
            
            if n_epochs is None:
                n_epochs = len(dates) - 1
          
            self.epoch_tiles = []
            dateobs = self.obsconditions['DATE-OBS']            
            for i in range(len(dates)-1):
                ii = (dates[i] < dateobs) & (dateobs < dates[i+1])
                self.epoch_tiles.append(self.obsconditions['TILEID'][ii])

        self.start_epoch = start_epoch
        if n_epochs is not None:
            self.n_epochs = n_epochs
        else:
            raise NameError('n_epochs was not set')

        if epochs_path is not None:
            if fiberassign_dates is not None:
                raise ArgumentError('epochs_path and fiberassign_dates are mutually exclusive')

            self.epochs_path = epochs_path
            # load tile list per epoch
            self.epoch_tiles = []
            for i in range(self.n_epochs):
                epochfile = os.path.join(self.epochs_path, "epoch{}.txt".format(i))        
                self.epoch_tiles.append(np.loadtxt(epochfile, dtype=int))

        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.truthfile  = os.path.join(self.targets_path,'truth.fits')
        self.targetsfile = os.path.join(self.targets_path,'targets.fits')
        self.zcat_file = None
        self.mtl_file = None

        self.tilefiles = []
        self.epochs_list = list(range(self.n_epochs))
        

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


    def create_surveyfile(self, epoch):
        """Creates text file of tiles survey_list.txt to be used by fiberassign

        Args:
            epoch (int) : epoch of tiles to write

        Notes: 
            The file is written to the temporary directory in self.tmp_output_path
        """

        # create survey list from mtl_epochs IDS
        surveyfile = os.path.join(self.tmp_output_path, "survey_list.txt")
        tiles = np.concatenate(self.epoch_tiles[epoch:])
        np.savetxt(surveyfile, tiles, fmt='%d')
        print("{} tiles to be included in fiberassign".format(len(tiles)))

    def create_fiberassign_input(self):
        """Creates input files for fiberassign from the provided template 
        
        Notes:
            The template filename is in self.template_fiberassign    
        """
        params = ''.join(open(self.template_fiberassign).readlines())
        fx = open(os.path.join(self.tmp_output_path, 'fa_features.txt'), 'w')
        fx.write(params.format(inputdir = self.tmp_output_path, targetdir = self.targets_path))
        fx.close()
        
    def simulate_epoch(self, epoch, perfect=False, truth=None, targets=None, mtl=None, zcat=None):
        """Core routine simulating a DESI epoch, 

        Args:
            epoch (int): epoch to simulate
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
        # load truth / targets / zcat
        if truth is None:
            truth = Table.read(os.path.join(self.targets_path,'truth.fits'))
        if targets is None:
            targets = Table.read(os.path.join(self.targets_path,'targets.fits'))
            
        print("{} Starting MTL".format(asctime()))
        self.mtl_file = os.path.join(self.tmp_output_path, 'mtl.fits')    
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

        # create a list of fiberassign tiles to read and update zcat
        self.tilefiles = []
        for i in self.epoch_tiles[epoch]:
            tilename = os.path.join(self.tmp_fiber_path, 'tile_%05d.fits'%(i))
            if os.path.isfile(tilename):
                self.tilefiles.append(tilename)
            else:
                print('Suggested but does not exist {}'.format(tilename))
        print("{} {} tiles to gather in zcat".format(asctime(), len(self.tilefiles)))

        # If applicable, select observations for these tiles
        if self.obsconditions is not None:
            ii = np.in1d(self.obsconditions['TILEID'], self.epoch_tiles[epoch])
            obsconditions = self.obsconditions[ii]
        else:
            obsconditions = None

        # write the zcat, it uses the tilesfiles constructed in the last step
        self.zcat_file = os.path.join(self.tmp_output_path, 'zcat.fits')
        print("{} starting quickcat".format(asctime()))
        newzcat = quickcat(self.tilefiles, targets, truth, zcat=zcat, 
                           obsconditions=obsconditions, perfect=perfect)
        print("{} writing zcat".format(asctime()))
        newzcat.write(self.zcat_file, format='fits', overwrite=True)
        print("{} Finished zcat".format(asctime()))
        return truth, targets, mtl, newzcat


    def simulate(self):
        """Simulate the DESI setup described by a SimSetup object.
        """
        self.create_directories()

        truth=targets=mtl=zcat=None
        for epoch in self.epochs_list[self.start_epoch:]:
            print('--- Epoch {} ---'.format(epoch))

            self.create_surveyfile(epoch)

            self.create_fiberassign_input()

            truth, targets, mtl, zcat = self.simulate_epoch(epoch, perfect=False,
                                                            truth=truth, targets=targets, mtl=mtl, zcat=zcat)

            self.backup_epoch_data(epoch_id=epoch)

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
