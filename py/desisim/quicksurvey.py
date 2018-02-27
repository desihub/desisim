'''
desisim.quicksurvey
===================

Code for quickly simulating the survey results given a mock catalog and
a list of tile epochs to observe.

Directly depends on the following DESI products:

* desitarget.mtl
* :mod:`desisim.quickcat`
* `fiberassign <https://github.com/desihub/fiberassign>`_
'''

from __future__ import absolute_import, division, print_function
import gc
import numpy as np
import os
import shutil
import glob
import subprocess
from astropy.table import Table, Column
import os.path
from collections import Counter
from time import time, asctime
import fitsio
import desitarget.mtl
from desisim.quickcat import quickcat
from astropy.table import join
from desitarget.targetmask import desi_mask

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
    def __init__(self, output_path, targets_path, fiberassign_exec, template_fiberassign,
                 exposures, fiberassign_dates):
        """Initializes all the paths, filenames and numbers describing DESI survey.

        Args:
            output_path (str): Path to write the outputs.x
            targets_path (str): Path where the files targets.fits can be found
            fiberassign_exec (str): Name of the fiberassign executable
            template_fiberassign (str): Filename of the template input for fiberassign
            exposures (stri): exposures.fits file summarazing surveysim results
            fiberassign_dates (str): ascii file with the dates to run fiberassign.
        """
        self.output_path = output_path
        self.targets_path = targets_path
        self.fiberassign_exec = fiberassign_exec
        self.template_fiberassign = template_fiberassign

        self.exposures = fitsio.read(exposures, upper=True)

        self.tmp_output_path = os.path.join(self.output_path, 'tmp/')
        self.tmp_fiber_path = os.path.join(self.tmp_output_path, 'fiberassign/')
        self.surveyfile = os.path.join(self.tmp_output_path, 'survey_list.txt')
        self.truthfile  = os.path.join(self.targets_path,'truth.fits')
        self.targetsfile = os.path.join(self.targets_path,'targets.fits')
        self.zcat_file = None
        self.mtl_file = None

        self.epoch_tiles = list()
        self.tilefiles = list()
        self.plan_tiles = list()
        self.observed_tiles = list()
        self.epochs_list = list()
        self.n_epochs = 0
        self.start_epoch = 0
 
        dateobs = np.core.defchararray.decode(self.exposures['NIGHT'])
        dates = list()
        with open(fiberassign_dates) as fx:
            for line in fx:
                line = line.strip()
                if line.startswith('#') or len(line) < 2:
                    continue
                yearmmdd = line.replace('-', '')
                year_mm_dd = yearmmdd[0:4]+yearmmdd[4:6]+yearmmdd[6:8]
                dates.append(year_mm_dd)

            #- add pre- and post- dates for date range bookkeeping
        if dates[0] < min(dateobs[0]):
                dates.insert(0, dateobs[0])

        dates.append('9999-99-99')
        print(dates)

        self.n_epochs = len(dates) - 1


        for i in range(len(dates)-1):
            ii = (dateobs >= dates[i]) & (dateobs < dates[i+1])
            epoch_tiles = list()
            for tile in self.exposures['TILEID'][ii]:
                if tile not in epoch_tiles:
                    epoch_tiles.append(tile)
            self.epoch_tiles.append(epoch_tiles)
            print('tiles in epoch {} [{} to {}]: {}'.format(i,dates[i], dates[i+1], len(self.epoch_tiles[i])))


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

    def epoch_data_exists(self, epoch_id=0):
        """Check epoch directory for zcat.fits and mtl.fits files.
        """
        backup_path = os.path.join(self.output_path, '{}'.format(epoch_id))
        mtl_file = os.path.join(backup_path, 'mtl.fits')
        zcat_file = os.path.join(backup_path, 'zcat.fits')
        
        if os.path.isfile(mtl_file) and os.path.isfile(zcat_file):
            return True
        else:
            return False

        
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
        tiles = self.epoch_tiles[epoch]
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

    def update_observed_tiles(self, epoch):
        """Creates the list of tilefiles to be gathered to buikd the redshift catalog.        

        """        
        self.tilefiles = list()
        tiles = self.epoch_tiles[epoch]
        for i in tiles:
            tilename = os.path.join(self.tmp_fiber_path, 'tile_%05d.fits'%(i))
            if os.path.isfile(tilename):
                self.tilefiles.append(tilename)
            else:
                print('Suggested but does not exist {}'.format(tilename))
        print("{} {} tiles to gather in zcat".format(asctime(), len(self.tilefiles)))

        
    def simulate_epoch(self, epoch, truth, targets, perfect=False, zcat=None):
        """Core routine simulating a DESI epoch,

        Args:
            epoch (int): epoch to simulate
            perfect (boolean): Default: False. Selects whether how redshifts are taken from the truth file.
                True: redshifts are taken without any error from the truth file.
                False: redshifts include uncertainties.
            truth (Table): Truth data
            targets (Table): Targets data
            zcat (Table): Redshift Catalog Data
        Notes:
            This routine simulates three steps:
            * Merged target list creation
            * Fiber allocation
            * Redshift catalogue construction
        """

        # create the MTL file
        print("{} Starting MTL".format(asctime()))
        self.mtl_file = os.path.join(self.tmp_output_path, 'mtl.fits')
        if zcat is None:
            mtl = desitarget.mtl.make_mtl(targets)
        else:
            mtl = desitarget.mtl.make_mtl(targets, zcat)
            
        mtl.write(self.mtl_file, overwrite=True)
        del mtl
        gc.collect()
        print("{} Finished MTL".format(asctime()))
                
        # clean files and prepare fiberasign inputs
        tilefiles = sorted(glob.glob(self.tmp_fiber_path+'/tile*.fits'))
        if tilefiles:
            for tilefile in tilefiles:
                os.remove(tilefile)

        # setup the tileids for the current observation epoch
        self.create_surveyfile(epoch)
        self.create_fiberassign_input()

        # launch fiberassign
        print("{} Launching fiberassign".format(asctime()))
        f = open('fiberassign.log','a')
        p = subprocess.call([self.fiberassign_exec, os.path.join(self.tmp_output_path, 'fa_features.txt')], stdout=f)# stdout=subprocess.PIPE)
        print("{} Finished fiberassign".format(asctime()))
        f.close()

        # create a list of fiberassign tiles to read and update zcat
        self.update_observed_tiles(epoch)

        #update obsconditions from progress_data
#        progress_data = Table.read(self.progress_files[epoch + 1])
#        ii = np.in1d(progress_data['TILEID'], self.observed_tiles)
#        obsconditions = progress_data[ii]
        obsconditions = None

        # write the zcat, it uses the tilesfiles constructed in the last step
        self.zcat_file = os.path.join(self.tmp_output_path, 'zcat.fits')
        print("{} starting quickcat".format(asctime()))
        newzcat = quickcat(self.tilefiles, targets, truth, zcat=zcat,
                           obsconditions=obsconditions, perfect=perfect)
        print("{} writing zcat".format(asctime()))
        newzcat.write(self.zcat_file, format='fits', overwrite=True)
        print("{} Finished zcat".format(asctime()))
        del newzcat
        gc.collect()
        return 


    def simulate(self):
        """Simulate the DESI setup described by a SimSetup object.
        """
        self.create_directories()

        truth = Table.read(os.path.join(self.targets_path,'truth.fits'))
        targets = Table.read(os.path.join(self.targets_path,'targets.fits'))

        print(truth.keys())
        #- Drop columns that aren't needed to save memory while manipulating
        truth.remove_columns(['SEED', 'MAG', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'HBETAFLUX', 'TEFF', 'LOGG', 'FEH'])
        targets.remove_columns([ 'SHAPEEXP_R', 'SHAPEEXP_E1', 'SHAPEEXP_E2', 'SHAPEDEV_R',
                                 'SHAPEDEV_E1', 'SHAPEDEV_E2', 'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
                                 'MW_TRANSMISSION_G','MW_TRANSMISSION_R','MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2'])
        gc.collect()
        if 'MOCKID' in truth.colnames:
            truth.remove_column('MOCKID')


        for epoch in range(self.start_epoch, self.n_epochs):
            print('--- Epoch {} ---'.format(epoch))
            
            if not self.epoch_data_exists(epoch_id=epoch):
                
                # Initializes mtl and zcat
                if epoch == 0:
                    zcat = None
                else:
                    print('INFO: Running Epoch {}'.format(epoch))
                    print('INFO: reading zcat from previous epoch')
                    epochdir = os.path.join(self.output_path, str(epoch-1))
                    zcat = Table.read(os.path.join(epochdir, 'zcat.fits'))

                # Update mtl and zcat
                self.simulate_epoch(epoch, truth, targets, perfect=True, zcat=zcat)

                # copy mtl and zcat to epoch directory
                self.backup_epoch_data(epoch_id=epoch)
                del zcat
                gc.collect()
            else:
                print('--- Epoch {} Already Exists ---'.format(epoch))
                
        self.cleanup_directories()




def print_efficiency_stats(truth, mtl_initial, zcat):
    print('Overall efficiency')
    tmp_init = join(mtl_initial, truth, keys='TARGETID')
    total = join(zcat, tmp_init, keys='TARGETID')

    true_types = ['LRG', 'ELG', 'QSO']
    zcat_types = ['GALAXY', 'GALAXY', 'QSO']

    for true_type, zcat_type in zip(true_types, zcat_types):
        i_initial = ((tmp_init['DESI_TARGET'] & desi_mask.mask(true_type)) != 0) & (tmp_init['TRUESPECTYPE'] == zcat_type)
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
