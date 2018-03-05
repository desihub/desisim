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

    group_cameras = np.array_split(np.arange(ncamera, dtype=np.int32), 
        ngroup)[group]
   
    # Compute which spectrographs our group needs to store based on which
    # cameras we are processing.

    group_spectro = np.unique(np.array([ int(args.cameras[c][1]) for c in \
        group_cameras ], dtype=np.int32))

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

    psf = None
    if args.psf is not None:
        from specter.psf import load_psf
        psf = load_psf(args.psf)

    # Read and distribute the simspec data
    # create list of tuples
    camera_channel_list=[]
    if comm is not None:
        if rank == 0:
        #need to parse these together so we preserve the order
            for item in args.cameras:
                #0th element is camera (b,r,z)
                #1st element is channel (0-9)
                camera_channel_list.append((item[0],item[1]))
        camera_channel_list=comm.bcast(camera_channel_list,root=0)    
        comm.Barrier()

    #preallocate things needed by both mpi and non mpi
    simspec = None
    cosmics = None
    image = {}
    rawpix = {}
    truepix = {}
    lastcamera = None

    # Read the fibermap

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

    fs = np.in1d(fibers//500, group_spectro)
    group_fibers = fibers[fs]
    
    # Use original seed to generate different random seeds for each camera
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 2**32-1, size=ncamera)
    seed_counter=0 #need to initialize counter

    #regroup and put all mpi broadcasting together! 
    #no mpi version is below
    if comm is not None:
        for entry in camera_channel_list: #split up channels for mpi Bcast speed
            channel=entry[0]
            #keep consistent definintion of camera
            camera = entry[0]+entry[1]

            simspec = io.read_simspec_mpi(args.simspec, comm, channel,
                      spectrographs=group_spectro)

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
                    cosmics = io.read_cosmics(cosmics_file, cosexpid, shape=shape)
                #must broadcast on all ranks
                cosmics = comm_group.bcast(cosmics, root=0)

            if group_rank == 0:
                group_size = comm_group.size
                log.info("Group {} ({} processes) simulating camera "
                    "{}".format(group, group_size, camera))
    
            image[camera], rawpix[camera], truepix[camera] = \
                simulate(camera, simspec, psf, fibers=group_fibers, 
                    ncpu=args.ncpu, nspec=args.nspec, cosmics=cosmics, 
                    wavemin=args.wavemin, wavemax=args.wavemax, preproc=False,
                    comm=comm_group)
    
            if args.psf is None:
                del psf
            
            #iterate random number counter
            seed_counter=seed_counter+1      

         
    #no mpi (multiprocessing) version                  
    else:
        simspec = io.read_simspec(args.simspec)

        for c in group_cameras:
            #define camera differently for multiprocessing than mpi
            camera = args.cameras[c]

            if group_rank == 0:
                log.debug('Processing camera {}'.format(camera))
            channel = camera[0].lower()
    
            # Set the seed for this camera (regardless of which process is
            # performing the simulation).
            np.random.seed(seeds[c])
    
            # Get the random cosmic expids.  The actual values will be
            # remapped internally with the modulus operator.
            cosexpid = np.random.randint(0, 100, size=1)[0]

            # Read inputs for this camera.  Unfortunately psf
            # objects are not serializable, so we read it on all
            # processes.
            if args.psf is None:
                #add this so that it won't fail if we enter two repeated channels
                if camera != lastcamera:
                    psf = desimodel.io.load_psf(channel)
                    if args.ccd_npix_x is not None:
                        psf.npix_x = args.ccd_npix_x
                    if args.ccd_npix_y is not None:
                        psf.npix_y = args.ccd_npix_y
                lastcamera = camera
   
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

            if args.psf is None:
                del psf
            
    #end of split mpi/no-mpi section#        

    # Write the cameras in order.  Only the rank zero process in each
    # group has the data.  If we are appending new cameras to an existing
    # raw file, then copy the original to the temporary output first.
    # Move temp files into place
    if rank == 0:
        # Copy the original files into place if we are appending
        if not args.overwrite and os.path.exists(args.rawfile):
            shutil.copy2(args.rawfile, rawtemp)
        if not args.overwrite and os.path.exists(args.simpixfile):
            shutil.copy2(args.simpixfile, simpixtemp)
    
    # Wait for all processes to finish their cameras
    if comm is not None:
        comm.barrier()

    #first write non-mpi data    
    if comm is None:
        for c in np.arange(ncamera, dtype=np.int32):
            camera = args.cameras[c]
            if c in group_cameras:
                if group_rank == 0:
                    desispec.io.write_raw(rawtemp, rawpix[camera], 
                        camera=camera, header=image[camera].meta, 
                        primary_header=simspec.header)
                    log.info('Wrote {} image to {}'.format(camera, args.rawfile))
                    io.write_simpix(simpixtemp, truepix[camera], 
                        camera=camera, meta=simspec.header)
                    log.info('Wrote {} image to {}'.format(camera, 
                        args.simpixfile))

    #otherwise write MPI data (easier to handle seperately)
    if comm is not None: 
        if comm.rank == 0:
            for entry in camera_channel_list:
                #keep consistent definition of channel
                channel=entry[0]
                camera=entry[0]+entry[1] #keep consistent definition of camera 
                desispec.io.write_raw(rawtemp, rawpix[camera], 
                camera=camera, header=image[camera].meta, 
                primary_header=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, args.rawfile))
                io.write_simpix(simpixtemp, truepix[camera], 
                    camera=camera, meta=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, 
                    args.simpixfile))

        comm.barrier()

    # Move temp files into place
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
        if len(group_cameras) > 0:
            if group_rank == 0:
                for c in group_cameras:
                    camera = args.cameras[c]
                    pixfile = desispec.io.findfile('pix', night=args.night,
                        expid=args.expid, camera=camera)
                    preproc_opts = ['--infile', args.rawfile, '--outdir',
                        args.preproc_dir, '--pixfile', pixfile]
                    preproc_opts += ['--cameras', camera]
                    preproc.main(preproc.parse(preproc_opts))

    if comm is not None:
        comm.barrier()
    
    # Python is terrible with garbage collection, but at least
    # encourage it...
    del image
    del rawpix
    del truepix

    if rank == 0:
        log.info('Finished pixsim {} expid {} at {}'.format(args.night, args.expid, asctime()))
