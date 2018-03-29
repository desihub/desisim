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

from ..pixsim import simulate
from ..io import SimSpec
from .. import obs, io

log = get_logger()

def expand_args(args):
    '''expand camera string into list of cameras
    '''
    # if simspec:
    #     if not night:
    #         get night from simspec
    #     if not expid:
    #         get expid from simspec
    # else:
    #     assert night and expid are set
    #     get simspec from (night, expid)
    #
    # if not outrawfile:
    #     get outrawfile from (night, expid)
    #
    # if outpixfile or outsimpixfile:
    #     assert len(cameras) == 1

    if args.simspec is None:
        if args.night is None or args.expid is None:
            msg = 'Must set --simspec or both --night and --expid'
            log.error(msg)
            raise ValueError(msg)
        args.simspec = io.findfile('simspec', args.night, args.expid)

    if args.fibermap is None:
        if (args.night is not None) and (args.expid is not None):
            args.fibermap = io.findfile('simfibermap', args.night, args.expid)

    if (args.cameras is None) and (args.spectrographs is None):
        from astropy.io import fits
        try:
            data = fits.getdata(args.simspec, 'B')
            nspec = data['PHOT'].shape[1]
        except KeyError:
            #- Try old specsim format instead
            hdr = fits.getheader(args.simspec, 'PHOT_B')
            nspec = hdr['NAXIS2']

        nspectrographs = (nspec-1) // 500 + 1
        args.spectrographs = list(range(nspectrographs))

    if (args.night is None) or (args.expid is None):
        from astropy.io import fits
        hdr = fits.getheader(args.simspec)
        if args.night is None:
            args.night = str(hdr['NIGHT'])
        if args.expid is None:
            args.expid = int(hdr['EXPID'])

    if isinstance(args.spectrographs, str):
        args.spectrographs = [int(x) for x in args.spectrographs.split(',')]

    #- expand camera list
    if args.cameras is None:
        args.cameras = list()
        for arm in args.arms.split(','):
            for ispec in args.spectrographs:
                args.cameras.append(arm+str(ispec))
    else:
        args.cameras = args.cameras.split(',')

    #- write to same directory as simspec
    if args.rawfile is None:
        rawfile = os.path.basename(desispec.io.findfile('raw', args.night, args.expid))
        args.rawfile = os.path.join(os.path.dirname(args.simspec), rawfile)

    if args.preproc:
        if args.preproc_dir is None:
            args.preproc_dir = os.path.dirname(args.rawfile)

    if args.simpixfile is None:
        args.simpixfile = io.findfile(
            'simpix', night=args.night, expid=args.expid,
            outdir=os.path.dirname(os.path.abspath(args.rawfile)))


#-------------------------------------------------------------------------
#- Parse options
def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Generates simulated DESI pixel-level raw data',
        )

    #- Input files
    parser.add_argument("--psf", type=str, help="PSF filename")
    parser.add_argument("--cosmics", action="store_true", help="Add cosmics")
    parser.add_argument("--cosmics_dir", type=str, 
        help="Input directory with cosmics templates")
    parser.add_argument("--cosmics_file", type=str, 
        help="Input file with cosmics templates")
    parser.add_argument("--simspec", type=str, help="input simspec file")
    parser.add_argument("--fibermap", type=str, 
        help="fibermap file (optional)")


    parser.add_argument("--rawfile", type=str, help="output raw data file")
    parser.add_argument("--simpixfile", type=str, 
        help="output truth image file")
    parser.add_argument("--preproc", action="store_true", 
        help="preprocess raw -> pix files")
    parser.add_argument("--preproc_dir", type=str, 
        help="directory for output preprocessed pix files")

    #- Alternately derive inputs/outputs from night, expid, and cameras
    parser.add_argument("--night", type=str, help="YEARMMDD")
    parser.add_argument("--expid", type=int, help="exposure id")
    parser.add_argument("--cameras", type=str, help="cameras, e.g. b0,r5,z9")

    parser.add_argument("--spectrographs", type=str, 
        help="spectrograph numbers, e.g. 0,1,9")
    parser.add_argument("--arms", type=str, 
        help="spectrograph arms, e.g. b,r,z", default='b,r,z')

    parser.add_argument("--ccd_npix_x", type=int, 
        help="for testing; number of x (columns) to include in output", 
        default=None)
    parser.add_argument("--ccd_npix_y", type=int, 
        help="for testing; number of y (rows) to include in output", 
        default=None)

    parser.add_argument("--verbose", action="store_true", 
        help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", 
        help="Overwrite existing raw and simpix files")
    parser.add_argument("--seed", type=int, help="random number seed")

    parser.add_argument("--ncpu", type=int, 
        help="Number of cpu cores per thread to use", default=0)
    parser.add_argument("--wavemin", type=float, 
        help="Minimum wavelength to simulate")
    parser.add_argument("--wavemax", type=float, 
        help="Maximum wavelength to simulate")
    parser.add_argument("--nspec", type=int, 
        help="Number of spectra to simulate per camera", 
                        default=0)

    parser.add_argument("--mpi_camera", type=int, default=1, help="Number of "
        "MPI processes to use per camera")

    if options is None:
        args = parser.parse_args()
    else:
        options = [str(x) for x in options]
        args = parser.parse_args(options)

    expand_args(args)
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
    else:
        if args.ncpu == 0:
            import multiprocessing as mp
            args.ncpu = mp.cpu_count() // 2

    if rank == 0:
        log.info('Starting pixsim at {}'.format(asctime()))

    #- Pre-flight check that these cameras haven't been done yet
    if (rank == 0) and (not args.overwrite) and os.path.exists(args.rawfile):
        log.debug('Checking if cameras are already in output file')
        from astropy.io import fits
        fx = fits.open(args.rawfile)
        oops = False
        for camera in args.cameras:
            if camera.upper() in fx:
                log.error('Camera {} already in {}'.format(camera, 
                    args.rawfile))
                oops = True
        fx.close()
        if oops:
            log.fatal('Exiting due to repeat cameras already in output file')
            if comm is not None:
                comm.Abort()
            else:
                sys.exit(1)

    #assign communicators, groups, etc, used for both mpi and multiprocessing
    ncamera = len(args.cameras)
    comm_group = comm
    comm_rank = None
    group = 0
    ngroup = 1
    group_rank = 0
    if comm is not None:
        if args.mpi_camera > 1:
            ngroup = int(comm.size / args.mpi_camera)
            group = int(comm.rank / args.mpi_camera)
            group_rank = comm.rank % args.mpi_camera
            comm_group = comm.Split(color=group, key=group_rank)
            comm_rank = comm.Split(color=group_rank, key=group)
        else:
            group = comm.rank
            ngroup = comm.size
            comm_group = MPI.COMM_SELF
            comm_rank = comm

    #check to make sure that the number of frames is evenly divisible by nubmer of nodes
    #if not, abort and provide a helpful error message
    #otherwise due to barrier logic the program will hang
    if comm is not None:
        if comm.rank == 0:
            if ncamera % ngroup != 0:
                msg = 'Number of frames (ncamera) must be evenly divisible by number of nodes (N)'
                log.error(msg)
                raise ValueError(msg)
                comm.Abort()

    # Remove outputs and or temp files
    rawtemp = "{}.tmp".format(args.rawfile)
    simpixtemp = "{}.tmp".format(args.simpixfile)

    if rank == 0:
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
    
    if comm is not None:
        comm.barrier()

    #load psf for both mpi and multiprocessing
    psf = None
    if args.psf is not None:
        from specter.psf import load_psf
        psf = load_psf(args.psf)

    # create list of tuples
    camera_channel_args=[]
    camera_channel_list=[]
    if comm is not None:
    #need to parse these together so we preserve the order
        for item in args.cameras:
            #0th element is camera (b,r,z)
            #1st element is channel (0-9)
            camera_channel_args.append((item[0],item[1]))
            #divide cameras among nodes
            #first sort so zs get distributed first 
            #reverse the order so z comes first
            #sort in reverse order so we get z first
            camera_channel_list=np.asarray(sorted(camera_channel_args, key=operator.itemgetter(0), reverse=True))
    #also handle multiprocessing case
    if comm is None:
        camera_channel_list=[]
        for item in args.cameras:
            camera_channel_list.append((item[0],item[1]))

    #if mpi and N>1, divide cameras between nodes
    if comm is not None: 
        if ngroup > 1:
            node_cameras=None
            for i in range(ngroup):
                if i == group: 
                    node_cameras=camera_channel_list[i::ngroup]
        if ngroup == 1:
            node_cameras=camera_channel_list
    #also handle multiprocessing case
    if comm is None:
        node_cameras=camera_channel_list

    group_spectro = np.array([ int(c[1]) for c in node_cameras ], dtype=np.int32)

    #preallocate things needed for multiprocessing (we handle mpi later)
    if comm is None:
        simspec = None
        cosmics = None
        image = {}
        rawpix = {}
        truepix = {}
        lastcamera = None

    # Read the fibermap
    #this is cheap, do here both mpi non mpi
    fibers = None
    if rank == 0:
        if args.fibermap is not None:
            fibermap = desispec.io.read_fibermap(args.fibermap)
            fibers = np.array(fibermap['FIBER'], dtype=np.int32)
        else:
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
        if args.nspec>0 :
            fibers = fibers[0:args.nspec]
    if comm is not None:
        fibers = comm.bcast(fibers, root=0)

    #do multiprocessing fibers now, handle mpi inside the loop
    if comm is None:
        fs = np.in1d(fibers//500, group_spectro)
        group_fibers = fibers[fs]
    
    # Use original seed to generate different random seeds for each camera
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 2**32-1, size=ncamera)
    seed_counter=0 #need to initialize counter

    #this loop handles the mpi case
    if comm is not None:
        #now that each communicator (group) knows which cameras it
        #is supposed to process, we can get started
        previous_camera='a' #need to initialize
        for i in range(len(node_cameras)): #may be different in each group
            if i > 1:
                #keep track in case we can avoid re-broadcasting stuff
                previous_camera=node_cameras[i-1]
            current_camera=node_cameras[i]
            camera=current_camera[0] + current_camera[1]
            channel=current_camera[0]

            #since we clear, we need to pre-allocate every time
            simspec = None
            image = {}
            rawpix = {}
            truepix = {}

            #since we handle only one spectra at a time, just load the one we need 
            simspec = io.read_simspec_mpi(args.simspec, comm_group, channel,
                          spectrographs=group_spectro[i])

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

            if args.psf is None:
                psf = desimodel.io.load_psf(channel)
                if args.ccd_npix_x is not None:
                    psf.npix_x = args.ccd_npix_x
                if args.ccd_npix_y is not None:
                    psf.npix_y = args.ccd_npix_y
            
            #if we have already loaded the right cosmics camera, don't bcast again
            cosmics=None
            if args.cosmics:
                if current_camera[0] != previous_camera[0]:
                    cosmics=None
                    if group_rank == 0:
                        if args.cosmics_file is None:
                            cosmics_file = io.find_cosmics(camera, 
                                simspec.header['EXPTIME'],
                                cosmics_dir=args.cosmics_dir)
                            log.info('cosmics templates {}'.format(cosmics_file))
                        else:
                            cosmics_file = args.cosmics_file
    
                        shape = (psf.npix_y, psf.npix_x)
                        cosmics = io.read_cosmics(cosmics_file, cosexpid, shape=shape)
                
                    cosmics = comm_group.bcast(cosmics, root=0)

            if group_rank == 0:
                group_size = comm_group.size
                log.info("Group {} ({} processes) simulating camera "
                    "{}".format(group, group_size, camera))

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
                        log.info('Wrote {} image to {} at time {}'.format(camera, args.rawfile, time.asctime()))
                        #write the simpix file using desisim write_simpix in io.py 
                        io.write_simpix(simpixtemp, truepix[camera],camera=camera, meta=simspec.header)
                        log.info('Wrote {} image to {} at time {}'.format(camera, args.simpixfile, time.asctime()))
                        #delete for each group after we have written the output files
                    del rawpix, image, truepix, simspec
                #to avoid bugs this barrier statement must align with if group==i!! 
                comm.Barrier()#all ranks should be done writing


            if args.psf is None:
                del psf
           
            #iterate random number counter
            seed_counter=seed_counter+1  

    #initalize random number seeds
    seed_counter=0 #need to initialize counter
    if comm is None:
        simspec = io.read_simspec(args.simspec)
        previous_camera='a'
        for i in range(len(node_cameras)):

            current_camera=node_cameras[i]
            camera=current_camera[0] + current_camera[1]
            channel=current_camera[0]

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
            if args.psf is None:
                #add this so that it won't fail if we enter two repeated channels
                if camera != previous_camera:
                    psf = desimodel.io.load_psf(channel)
                    if args.ccd_npix_x is not None:
                        psf.npix_x = args.ccd_npix_x
                    if args.ccd_npix_y is not None:
                        psf.npix_y = args.ccd_npix_y
                previous_camera = camera
   
            if args.cosmics:
                if group_rank == 0:
                    if args.cosmics_file is None:
                        cosmics_file = io.find_cosmics(camera, 
                            simspec.header['EXPTIME'],
                            cosmics_dir=args.cosmics_dir)
                        log.info('cosmics templates {}'.format(cosmics_file))
                    else:
                        cosmics_file = args.cosmics_file
    
                    shape = (psf.npix_y, psf.npix_x)
                    cosmics = io.read_cosmics(cosmics_file, cosexpid, 
                        shape=shape)


            #- Do the actual simulation
            # Each process in the group is responsible for a subset of the
            # fibers.

            if group_rank == 0:
                group_size = 1
                log.info("Group {} ({} processes) simulating camera "
                    "{}".format(group, group_size, camera))

            image[camera], rawpix[camera], truepix[camera] = \
                simulate(camera, simspec, psf, fibers=group_fibers, 
                    ncpu=args.ncpu, nspec=args.nspec, cosmics=cosmics, 
                    wavemin=args.wavemin, wavemax=args.wavemax, preproc=False,
                    comm=comm_group)

            #like we did in mpi, move the write in here so we don't have to 
            #keep accumulating all the data
            if group_rank == 0:
                desispec.io.write_raw(rawtemp, rawpix[camera],
                    camera=camera, header=image[camera].meta,
                    primary_header=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, args.rawfile))
                io.write_simpix(simpixtemp, truepix[camera],
                    camera=camera, meta=simspec.header)
                log.info('Wrote {} image to {}'.format(camera,
                    args.simpixfile))


            if args.psf is None:
                del psf
    
            #iterate random number counter
            seed_counter=seed_counter+1

    #done with both mpi and multiprocessing
        
    # Move temp files into place
    if rank == 0:
        # Copy the original files into place if we are appending
        if not args.overwrite and os.path.exists(args.rawfile):
            shutil.copy2(args.rawfile, rawtemp)
        if not args.overwrite and os.path.exists(args.simpixfile):
            shutil.copy2(args.simpixfile, simpixtemp)

    if rank == 0:
        os.rename(simpixtemp, args.simpixfile)
        os.rename(rawtemp, args.rawfile)
    if comm is not None:
        comm.barrier()

    # Apply preprocessing
    if args.preproc:
        if rank == 0:
            log.info('Preprocessing raw -> pix files')
        from desispec.scripts import preproc
        if len(node_cameras) > 0:
            if group_rank == 0:
                for c in node_cameras:
                    camera = args.cameras[c]
                    pixfile = desispec.io.findfile('pix', night=args.night,
                        expid=args.expid, camera=camera)
                    preproc_opts = ['--infile', args.rawfile, '--outdir',
                        args.preproc_dir, '--pixfile', pixfile]
                    preproc_opts += ['--cameras', camera]
                    preproc.main(preproc.parse(preproc_opts))

    # Python is terrible with garbage collection, but at least
    # encourage it...
    if comm is None:
        del image
        del rawpix
        del truepix
    #have already deleted these for the mpi case

    if rank == 0:
        log.info('Finished pixsim {} expid {} at {}'.format(args.night, args.expid, asctime()))
