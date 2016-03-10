'''
Provides utility functions for batch processing of pixel-level simulations at
NERSC.  This is a temporary pragmatic package -- after desispec.pipeline code
is merged and vetted, this should use that infrastructure for more rigorous
logging, environment setup, and scaling flexibility.

Example:

#- From python
import desisim.batch.pixsim
flavors = ['arc', 'flat', 'dark', 'dark', 'gray', 'gray', 'bright', 'bright']
expids = range(len(flavors))
desisim.batch.pixsim.batch_newexp('newexp-blat.sh', flavors, expids=expids)
desisim.batch.pixsim.batch_pixsim('pixsim-blat.sh', flavors, expids=expids)

#- then from the command line
[edison] sbatch newexp-batch.sh
Submitted batch job 233895
[edison] sbatch -d afterok:233895 pixsim-batch.sh
Submitted batch job 233901
'''

from __future__ import absolute_import, division, print_function
import os
from desisim import obs
from desisim.batch import calc_nodes

from desispec.log import get_logger
log = get_logger()

def batch_newexp(batchfile, flavors, nspec=5000, night=None, expids=None,
    nodes=None, pixprod=None, desi_spectro_sim=None):
    '''
    Write a slurm batch script for run newexp-desi for the list of flavors
    '''
    nexp = len(flavors)
    timestr = '00:30:00'
    logfile = '{}.%j.log'.format(batchfile)
    
    if night is None:
        night = obs.get_night()
        
    if expids is None:
        expids = obs.get_next_expid(nexp)
        
    if nodes is None:
        nodes = calc_nodes(nexp, tasktime=1.5, maxtime=20)
        
    if pixprod is None:
        if 'PIXPROD' in os.environ:
            pixprod = os.environ['PIXPROD']
        else:
            raise ValueError('must provide pixprod or set $PIXPROD')
        
    if desi_spectro_sim is None:
        if 'DESI_SPECTRO_SIM' in os.environ:
            desi_spectro_sim = os.environ['DESI_SPECTRO_SIM']
        else:
            raise ValueError('must provide desi_spectro_sim or set $DESI_SPECTRO_SIM')
        
    log.info('output dir {}/{}/{}'.format(desi_spectro_sim, pixprod, night))
    
    assert len(expids) == len(flavors)
    
    cmd = "srun -n 1 -N 1 -c $nproc /usr/bin/time newexp-desi --night {night} --nspec {nspec} --flavor {flavor} --expid {expid}"
    with open(batchfile, 'w') as fx:
        fx.write("#!/bin/bash -l\n\n")
        fx.write("#SBATCH --partition=debug\n")
        fx.write("#SBATCH --account=desi\n")
        fx.write("#SBATCH --nodes={}\n".format(nodes))
        fx.write("#SBATCH --time={}\n".format(timestr))
        fx.write("#SBATCH --job-name=newexp\n")
        fx.write("#SBATCH --output={}\n".format(logfile))
        
        fx.write("if [ ${NERSC_HOST} = edison ]; then\n")
        fx.write("  nproc=24\n")
        fx.write("else\n")
        fx.write("  nproc=32\n")
        fx.write("fi\n\n")
        
        fx.write('export DESI_SPECTRO_SIM={}\n'.format(desi_spectro_sim))
        fx.write('export PIXPROD={}\n'.format(pixprod))
        fx.write('\n')
        
        for expid, flavor in zip(expids, flavors):
            fx.write(cmd.format(nspec=nspec, night=night, expid=expid, flavor=flavor)+' &\n')
            
        fx.write('\nwait\n')

    return expids

def batch_pixsim(batchfile, flavors, nspec=5000, night=None, expids=None,
    cosmics_dir=None, nodes=None, pixprod=None, desi_spectro_sim=None):
    '''
    Write a slurm batch script for run newexp-desi for the list of flavors
    '''
    nexp = len(flavors)
    nspectrographs = (nspec-1) // 500 + 1
    ntasks = nexp * nspectrographs * 3
    timestr = '00:30:00'
    logfile = '{}.%j.log'.format(batchfile)
    
    if night is None:
        night = obs.get_night()
        
    if expids is None:
        expids = obs.get_next_expid(nexp)

    assert len(expids) == len(flavors)

    if pixprod is None:
        if 'PIXPROD' in os.environ:
            pixprod = os.environ['PIXPROD']
        else:
            raise ValueError('must provide pixprod or set $PIXPROD')
        
    if desi_spectro_sim is None:
        if 'DESI_SPECTRO_SIM' in os.environ:
            desi_spectro_sim = os.environ['DESI_SPECTRO_SIM']
        else:
            raise ValueError('must provide desi_spectro_sim or set $DESI_SPECTRO_SIM')

    log.info('output dir {}/{}/{}'.format(desi_spectro_sim, pixprod, night))

    #- HARDCODE !!!
    if cosmics_dir is None:
        cosmics_dir = os.getenv('DESI_ROOT') + '/spectro/templates/cosmics/v0.2/'

    if nodes is None:
        nodes = calc_nodes(ntasks, tasktime=5, maxtime=20)
        
    cmd = "srun -n 1 -N 1 -c $nproc /usr/bin/time pixsim-desi --verbose --night {night} --expid {expid} --cameras {camera} --cosmics {cosmics}"    
    with open(batchfile, 'w') as fx:
        fx.write("#!/bin/bash -l\n\n")
        fx.write("#SBATCH --partition=debug\n")
        fx.write("#SBATCH --account=desi\n")
        fx.write("#SBATCH --nodes={}\n".format(nodes))
        fx.write("#SBATCH --time={}\n".format(timestr))
        fx.write("#SBATCH --job-name=newexp\n")
        fx.write("#SBATCH --output={}\n".format(logfile))

        fx.write("if [ ${NERSC_HOST} = edison ]; then\n")
        fx.write("  nproc=24\n")
        fx.write("else\n")
        fx.write("  nproc=32\n")
        fx.write("fi\n\n")
        
        fx.write('export DESI_SPECTRO_SIM={}\n'.format(desi_spectro_sim))
        fx.write('export PIXPROD={}\n'.format(pixprod))

        for expid, flavor in zip(expids, flavors):
            fx.write('\n#--- Exposure {} ({})\n'.format(expid, flavor))
            for spectrograph in range(nspectrographs):
                for channel in ['b', 'r', 'z']:
                    if flavor in ('arc', 'flat'):
                        cosmics = cosmics_dir + '/cosmics-bias-{}.fits'.format(channel)
                    else:
                        cosmics = cosmics_dir + '/cosmics-dark-{}.fits'.format(channel)
                        
                    camera = '{}{}'.format(channel, spectrograph)
                    
                    cx = cmd.format(night=night, expid=expid, cosmics=cosmics,
                        camera=camera,
                    )
                    fx.write(cx + ' &\n')
            
        fx.write('\nwait\n')


            
