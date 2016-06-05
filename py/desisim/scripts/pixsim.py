from __future__ import absolute_import, division, print_function

import os
import os.path
import multiprocessing as mp
import random
from time import asctime

import numpy as np

import desimodel.io
import desispec.io

import desisim
from desisim import obs, io
from desispec.log import get_logger
log = get_logger()

def expand_args(args):
    '''
    expand camera string into list of cameras
    if simspec:
        if not night:
            get night from simspec
        if not expid:
            get expid from simspec
    else:
        assert night and expid are set
        get simspec from (night, expid)
    
    if not outrawfile:
        get outrawfile from (night, expid)
        
    if outpixfile or outsimpixfile:
        assert len(cameras) == 1
    '''
    if args.simspec is None:
        if args.night is None or args.expid is None:
            msg = 'Must set --simspec or both --night and --expid'
            log.error(msg)
            raise ValueError(msg)
        args.simspec = io.findfile('simspec', args.night, args.expid)

    if (args.cameras is None) and (args.spectrographs is None):
        from astropy.io import fits
        hdr = fits.getheader(args.simspec, 'PHOT_B')
        nspec = hdr['NAXIS2']
        nspectrographs = (nspec-1) // 500 + 1
        args.spectrographs = range(nspectrographs)

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
            outdir=os.path.dirname(args.rawfile))

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
    parser.add_argument("--cosmics_dir", type=str, help="Input directory with cosmics templates")
    parser.add_argument("--cosmics_file", type=str, help="Input file with cosmics templates")
    parser.add_argument("--simspec", type=str, help="input simspec file")
    parser.add_argument("--fibermap", type=str, help="fibermap file (optional)")
        
    #- Output options
    parser.add_argument("--rawfile", type=str, help="output raw data file")
    parser.add_argument("--simpixfile", type=str, help="output truth image file")
    parser.add_argument("--preproc", action="store_true", help="preprocess raw -> pix files")
    parser.add_argument("--preproc_dir", type=str, help="directory for output preprocessed pix files")
        
    #- Alternately derive inputs/outputs from night, expid, and cameras
    parser.add_argument("--night", type=str, help="YEARMMDD")
    parser.add_argument("--expid", type=int, help="exposure id")
    parser.add_argument("--cameras", type=str, help="cameras, e.g. b0,r5,z9")

    parser.add_argument("--spectrographs", type=str, help="spectrograph numbers, e.g. 0,1,9")
    parser.add_argument("--arms", type=str, help="spectrograph arms, e.g. b,r,z", default='b,r,z')

    # parser.add_argument("--trimxy", action="store_true", help="Trim image to fit spectra")
    parser.add_argument("--verbose", action="store_true", help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw and simpix files")
    parser.add_argument("--seed", type=int, help="random number seed")
    parser.add_argument("--nspec", type=int, help="Number of spectra to simulate per camera %(default)s", default=500)
    parser.add_argument("--mpi", action="store_true", help="Use MPI parallelism")
    parser.add_argument("--ncpu",  type=int, help="Number of cpu cores per thread to use %(default)s", default=mp.cpu_count() // 2)
    parser.add_argument("--wavemin",  type=float, help="Minimum wavelength to simulate")
    parser.add_argument("--wavemax",  type=float, help="Maximum wavelength to simulate")

    if options is None:
        args = parser.parse_args()
    else:
        options = [str(x) for x in options]        
        args = parser.parse_args(options)

    expand_args(args)
    return args

def main(args=None):
    log.info('Starting pixsim at {}'.format(asctime()))
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if args.verbose:
        import logging
        log.setLevel(logging.DEBUG)

    if args.mpi:
        from mpi4py import MPI
        mpicomm = MPI.COMM_WORLD
        log.debug('Using mpi4py MPI communicator')
    else:
        from desisim.util import _FakeMPIComm
        log.debug('Using fake MPI communicator')
        mpicomm = _FakeMPIComm()

    log.info('Starting pixsim rank {} at {}'.format(mpicomm.rank, asctime()))
    log.debug('MPI rank {} size {} / {} {}'.format(
        mpicomm.rank, mpicomm.size, mpicomm.Get_rank(), mpicomm.Get_size()))

    #- Pre-flight check that these cameras haven't been done yet
    if (mpicomm.rank == 0) and (not args.overwrite) and os.path.exists(args.rawfile):
        log.debug('Checking if cameras are already in output file')
        from astropy.io import fits
        fx = fits.open(args.rawfile)
        oops = False
        for camera in args.cameras:
            if camera.upper() in fx:
                log.error('Camera {} already in {}'.format(camera, args.rawfile))
                oops = True
        
        fx.close()
        if oops:
            log.fatal('Exiting due to repeat cameras already in output file')
            mpicomm.Abort()

    ncameras = len(args.cameras)
    if ncameras % mpicomm.size != 0:
        log.fatal('Processing cameras {}'.format(args.cameras))
        log.fatal('Number of cameras {} must be evenly divisible by MPI size {}'.format(ncameras, mpicomm.size))
        mpicomm.Abort()

    #- Use original seed to generate different random seeds for each MPI rank
    np.random.seed(args.seed)
    while True:
        seeds = np.random.randint(0, 2**32-1, size=mpicomm.size)
        if np.unique(seeds).size == size:
            random.seed(seeds[mpicomm.rank])
            np.random.seed(seeds[mpicomm.rank])
            break

    if args.psf is not None:
        import specter.io
        psf = specter.io.load_psf(args.psf)

    simspec = io.read_simspec(args.simspec)

    if args.fibermap:
        fibermap = desispec.io.read_fibermap(args.fibermap)
        fibers = fibermap['FIBER']
    else:
        fibers = None

    if args.overwrite and os.path.exists(args.rawfile):
        log.debug('removing {}'.format(args.rawfile))
        os.remove(args.rawfile)

    if args.overwrite and os.path.exists(args.simpixfile):
        log.debug('removing {}'.format(args.simpixfile))
        os.remove(args.simpixfile)

    for i in range(mpicomm.rank, ncameras, mpicomm.size):
        camera = args.cameras[i]
        log.debug('Rank {} processing camera {}'.format(mpicomm.rank, camera))
        channel = camera[0].lower()
        assert channel in ('b', 'r', 'z'), "Unknown camera {} doesn't start with b,r,z".format(camera)
        
        #- Read inputs for this camera
        if args.psf is None:
            psf = desimodel.io.load_psf(channel)
            
        if args.cosmics:
            if args.cosmics_file is None:
                cosmics_file = io.find_cosmics(camera, simspec.header['EXPTIME'],
                                               cosmics_dir=args.cosmics_dir)
                log.info('cosmics templates {}'.format(cosmics_file))
            else:
                cosmics_file = args.cosmics_file

            shape = (psf.npix_y, psf.npix_x)
            cosmics = io.read_cosmics(cosmics_file, args.expid, shape=shape)
        else:
            cosmics = None
        
        #- Do the actual simulation
        image, rawpix, truepix = desisim.pixsim.simulate(
            camera, simspec, psf, fibers=fibers,
            nspec=args.nspec, ncpu=args.ncpu, cosmics=cosmics,
            wavemin=args.wavemin, wavemax=args.wavemax)

        #- Synchronize with other MPI threads before continuing with output
        mpicomm.Barrier()

        #- Loop over MPI ranks, letting each one take its turn writing output
        for rank in range(mpicomm.size):
            if rank == mpicomm.rank:
                desispec.io.write_raw(args.rawfile, rawpix, camera=camera,
                    header=image.meta, primary_header=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, args.rawfile))
                io.write_simpix(args.simpixfile, truepix, camera=camera,
                    meta=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, args.simpixfile))
                mpicomm.Barrier()
            else:
                mpicomm.Barrier()

    if args.preproc and mpicomm.rank == 0:
        log.info('Preprocessing raw -> pix files')
        from desispec.scripts import preproc
        preproc_opts = ['--infile', args.rawfile, '--outdir', args.preproc_dir]
        preproc_opts += ['--cameras', ','.join(args.cameras)]
        preproc.main(preproc.parse(preproc_opts))

    if mpicomm.rank == 0:
        log.info('Finished pixsim {}'.format(asctime()))

