"""
desisim.scripts.pixsim
======================

This is a module.
"""
from __future__ import absolute_import, division, print_function

import os,sys
import os.path
import shutil

import random
from time import asctime

import numpy as np

import desimodel.io
from desiutil.log import get_logger
import desispec.io
from desispec.parallel import stdouterr_redirected

from . import pixsim

from ..pixsim import simulate
from ..io import SimSpec
from .. import obs, io

import argparse

import desispec.io as specio

from .. import io as simio

log = get_logger()


def parse(options=None):
    parser = argparse.ArgumentParser(
        description = 'Generate pixel-level simulated DESI data for one or more nights',
        )

    parser.add_argument("--nights", type=str, default=None, required=False, help="YEARMMDD,YEARMMDD,YEARMMDD")
    parser.add_argument("--verbose", action="store_true", help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw and simpix files")
    parser.add_argument("--cosmics_dir", type=str, help="Input directory with cosmics templates") 
    parser.add_argument("--cosmics", action="store_true", help="Add simulated cosmics")
    parser.add_argument("--preproc", action="store_true", help="Run the preprocessing")
    parser.add_argument("--seed", type=int, default=123456, required=False, help="random number seed")
    parser.add_argument("--rawfile", type=str, help="output raw data file")
    parser.add_argument("--simpixfile", type=str, help="output truth image file")

    #need these so simulate will run, but they won't really work properly
    parser.add_argument("--ncpu", type=int, help="Number of cpu cores per thread to use", default=0)
    parser.add_argument("--nspec", type=int, help="Number of spectra to simulate per camera", default=0)
    parser.add_argument("--wavemin", type=float, help="Minimum wavelength to simulate")
    parser.add_argument("--wavemax", type=float, help="Maximum wavelength to simulate")

    #cameras may be specified, but camera_procs must be specified!
    parser.add_argument("--cameras", type=str, default=None, help="cameras, e.g. b0,r5,z9")
    parser.add_argument("--camera_procs", type=int, default=1, required=True,
        help="Number of MPI processes to use per camera")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        options = [str(x) for x in options]
        args = parser.parse_args(options)

    return args

def main(args, comm=None):
    if args.verbose:
        import logging
        log.setLevel(logging.DEBUG)
    #we do this so we can use operator.itemgetter
    import operator
    #we do this so our print statements can have timestamps
    import time

    rank = 0
    nproc = 1
    if comm is not None:
        import mpi4py
        rank = comm.rank
        nproc = comm.size

    if rank == 0:
        log.info('Starting pixsim at {}'.format(asctime()))


    #no preflight check here, too complicated.   
    #we'll assume the user knows what he or she is doing...

    # Determine which nights we are using
    nights = None
    if args.nights is not None:
        nights = args.nights.split(",")
    else:
        rawdir = os.path.abspath(specio.rawdata_root())
        nights = []
        nightpat = re.compile(r"\d{8}")
        for root, dirs, files in os.walk(rawdir, topdown=True):
            for d in dirs:
                nightmat = nightpat.match(d)
                if nightmat is not None:
                    nights.append(d)

    # Get the list of exposures for each night
    night_expid = {}
    all_expid = []
    exp_to_night = {}
    for nt in nights:
        night_expid[nt] = specio.get_exposures(nt, raw=True)

    #this is currently a dict, which is all well and good, but we need 
    #a list of tuples of (night,expid) that we can evenly divide between communicators
    night_exposure_list=list()
    for nt in nights:
        for exp in night_expid[nt]:
            night_exposure_list.append([nt,exp])              

    #then this sub communicator will be split into the number of exposures per node
    #instead of mpi_cameras we will use camera_procs... but maybe we also need to specify
    #a finer grain paralleism, we'll see...
    comm_group = comm
    comm_rank = None
    group = 0
    ngroup = 1
    group_rank = 0
    if comm is not None:
        if args.camera_procs > 1:
            ngroup = int(comm.size / args.camera_procs)
            group = int(comm.rank / args.camera_procs)
            group_rank = comm.rank % args.camera_procs
            comm_group = comm.Split(color=group, key=group_rank)
            comm_rank = comm.Split(color=group_rank, key=group)
        else:
            group = comm.rank
            ngroup = comm.size
            comm_group = MPI.COMM_SELF
            comm_rank = comm

    #divide lists of exposures between split communicators
    #this is where the magic happens
    if ngroup > 1:
        node_exposures=None
        for i in range(ngroup):
            if i == group: 
                node_exposures=night_exposure_list[i::ngroup]
    if ngroup == 1:
        node_exposures=night_exposure_list

    #the way we are splitting the exposures assumes that they can be evenly 
    #divided between the nodes. if this is not the case, the barrier logic
    #will cause a hang during the file write (i think)
    #first let's check to see if it is a problem
    nexposures=len(night_exposure_list)
    if nexposures % ngroup != 0:
        if comm.rank == 0:
            msg = ("nexposures {} does not divide evenly into ngroup {}, simulation aborting".format(nexposures, ngroup))
            log.error(msg)
            raise ValueError(msg)
            comm.Abort()
    elif nexposures % ngroup == 0:
        if comm.rank == 0:
            log.info("nexposures {} divides evenly into ngroup {}, simulation will proceed".format(nexposures, ngroup))         

    comm.Barrier()
    #print("node_exposures=  %s for comm_group %s" %(node_exposures,comm_group))

    # Get the list of cameras
    # should be the same on every node
    cams = None
    if args.cameras is not None:
        cams = args.cameras
    else:
        cams = []
        #do this with band first so we can avoid re-broadcasting cosmics
        for band in ['b', 'r', 'z']:
            for spec in range(10):
                cams.append('{}{}'.format(band, spec))

    for j in range(len(node_exposures)):

        #for each exposure, handle group spectro
        group_spectro = np.array([ int(c[1]) for c in cams], dtype=np.int32)

        # Use original seed to generate different random seeds for each camera
        np.random.seed(args.seed)
        seeds = np.random.randint(0, 2**32-1, size=len(cams))
        seed_counter=0 #need to initialize counter
       
        #now that each communicator (group) knows which cameras it
        #is supposed to process, we can get started
        previous_camera='a' #need to initialize

        #the inner loop handles the frames for each exposure
        for i in range(len(cams)):
            if i > 1:
                #keep track in case we can avoid re-broadcasting stuff
                previous_camera=cams[i-1]
            current_camera=cams[i]
            camera=current_camera[0] + current_camera[1]
            channel=current_camera[0]

            #since we clear, we need to pre-allocate every time
            simspec = None
            image = {}
            rawpix = {}
            truepix = {}
            fibers = None
          
            #figure out which simspec file we need
            args.simspec=io.findfile('simspec',node_exposures[j][0], node_exposures[j][1])

            #need to sort out raw and simpix files
            if args.rawfile is None:
                rawfile = os.path.basename(desispec.io.findfile('raw', node_exposures[j][0], node_exposures[j][1]))
                args.rawfile = os.path.join(os.path.dirname(args.simspec), rawfile)

            if args.simpixfile is None:
                args.simpixfile = io.findfile(
                    'simpix', night=node_exposures[j][0], expid=node_exposures[j][1],
                    outdir=os.path.dirname(os.path.abspath(args.rawfile)))

            #clean up files before we start
            # Remove outputs and or temp files
            rawtemp = "{}.tmp".format(args.rawfile)
            simpixtemp = "{}.tmp".format(args.simpixfile)

            if group_rank == 0:
                if args.overwrite and os.path.exists(args.rawfile):
                    log.debug('removing {}'.format(args.rawfile))
                    os.remove(args.rawfile)

                if args.overwrite and os.path.exists(args.simpixfile):
                    log.debug('removing {}'.format(args.simpixfile))
                    os.remove(args.simpixfile)

                # cleanup stale temp files
                if os.path.isfile(rawtemp):
                    os.remove(rawtemp)
                if os.path.isfile(simpixtemp):
                    os.remove(simpixtemp)

            #since we handle only one spectra at a time, just load the one we need 
            simspec = io.read_simspec_mpi(args.simspec, comm_group, channel,
                          spectrographs=group_spectro[i])

            #then get the fibers, which requires simspec
            if group_rank == 0:
                # Get the fiber list from the simspec file
                from astropy.io import fits
                from astropy.table import Table
                fx = fits.open(args.simspec, memmap=True)
                if 'FIBERMAP' in fx:
                    fibermap = Table(fx['FIBERMAP'].data)
                    fibers = np.array(fibermap['FIBER'], dtype=np.int32)
                else:
                    # Get the number of fibers from one of the photon HDUs
                    fibers = np.arange(fx['PHOT_B'].header['NAXIS2'], 
                        dtype=np.int32)
                fx.close()
            #then need to broadcast within communicator so all ranks have fibers
            fibers = comm_group.bcast(fibers, root=0)

            if group_rank == 0:
                log.debug('Processing camera {}'.format(camera))
    
            # Set the seed for this camera (regardless of which process is
            # performing the simulation).
            np.random.seed(seeds[(seed_counter)])
    
            # Get the random cosmic expids.  The actual values will be
            # remapped internally with the modulus operator.
            cosexpid = np.random.randint(0, 100, size=1)[0]
    
            # Read inputs for this camera.  Unfortunately psf
            # objects are not serializable, so we read it on all
            # processes.
            psf = desimodel.io.load_psf(channel)
            
            #if we have already loaded the right cosmics camera, don't bcast again
            cosmics=None
            if args.cosmics:
                if current_camera[0] != previous_camera[0]:
                    cosmics=None
                    if group_rank == 0:
                        cosmics_file = io.find_cosmics(camera, 
                            simspec.header['EXPTIME'],
                            cosmics_dir=args.cosmics_dir)
                        log.info('cosmics templates {}'.format(cosmics_file))
    
                        shape = (psf.npix_y, psf.npix_x)
                        cosmics = io.read_cosmics(cosmics_file, cosexpid, shape=shape)
                
                    cosmics = comm_group.bcast(cosmics, root=0)

            if group_rank == 0:
                group_size = comm_group.size
                log.info("Group {} ({} processes) simulating camera "
                    "{} exposure {} night {}".format(group, group_size, camera, 
                    node_exposures[j][1], node_exposures[j][0]))

            #calc group_fibers inside the loop
            fs = np.in1d(fibers//500, group_spectro[i])
            group_fibers = fibers[fs]

            image[camera], rawpix[camera], truepix[camera] = \
                simulate(camera, simspec, psf, fibers=group_fibers,
                    ncpu=args.ncpu, nspec=args.nspec, cosmics=cosmics,
                    wavemin=args.wavemin, wavemax=args.wavemax, preproc=False,
                    comm=comm_group)

            #to avoid overflowing the memory let's write after each instance of simulate
            #the data are already at rank 0 in each communicator
            #then have each communicator open and write to the file one at a time
            #this is the part that will fail if the number of nodes divided by number of frames 
            #is not evenly divisible (hence the check above)
            for i in range(ngroup):#number of total communicators
                if group == i:#only one group at a time should open/write/close
                    if group_rank == 0: #write only from rank 0 where the data are
                        #write the raw file using desispec write_raw in raw.py
                        desispec.io.write_raw(rawtemp, rawpix[camera],
                           camera=camera, header=image[camera].meta,
                           primary_header=simspec.header)
                        log.info('Wrote {} image {} exposure {} night to {} at time {}'
                            .format(camera, node_exposures[j][1], node_exposures[j][0], 
                            rawtemp, time.asctime()))
                        #write the simpix file using desisim write_simpix in io.py 
                        io.write_simpix(simpixtemp, truepix[camera],camera=camera, meta=simspec.header)
                        log.info('Wrote {} image {} exposure {} night to {} at time {}'
                            .format(camera, node_exposures[j][1], node_exposures[j][0],
                            simpixtemp, time.asctime()))
                        #delete for each group after we have written the output files
                    del rawpix, image, truepix, simspec
                #to avoid bugs this barrier statement must align with if group==i!! 
                comm.Barrier()#all ranks should be done writing

            if group_rank == 0:
                os.rename(simpixtemp, args.simpixfile)
                os.rename(rawtemp, args.rawfile)

            #iterate random number counter
            seed_counter=seed_counter+1
            #end of inner loop

    comm.Barrier()

    if rank == 0:
        log.info('Finished pixsim nights {}'.format(args.nights, asctime()))
