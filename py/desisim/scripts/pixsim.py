"""
desisim.scripts.pixsim
======================

This is a module.
"""
from __future__ import absolute_import, division, print_function

import os
import os.path
import multiprocessing as mp
import random
from time import asctime

import numpy as np

import desimodel.io
import desispec.io
from desispec.parallel import stdouterr_redirected

import desisim
import desisim.pixsim
from desisim.io import SimSpec
from desisim import obs, io
from desiutil.log import get_logger
from scipy.sparse import lil_matrix

log = get_logger()


class SparseArray(object) :
    def __init__(self,array,begin,end) :
        self.mat = lil_matrix(array.shape)
        self.mat[begin:end] = array[begin:end]
    def __getitem__(self, index):
        return self.mat[index] #.toarray()

class SparseSimSpec(object) :
    """ a slice of a desisim.io.SimSpec object """
    def __init__(self, simspec, begin, end, rank=0) :
        self.flavor = simspec.flavor
        self.obs = simspec.obs
        self.header = simspec.header
        self.nspec = simspec.nspec
        self.wave = simspec.wave
        self.metadata = simspec.metadata
        self.fibermap = simspec.fibermap

        if rank==0 :
            self.flux     = simspec.flux
            self.skyflux  = simspec.skyflux
        else :
            self.flux     = None
            self.skyflux  = None

        # now, for the rest we only want to allocate the memory and fill vectors in range begin:end
        # but 'pretending' it's a full array
        
        self.phot = dict()
        self.skyphot = dict()
        for channel in ('b', 'r', 'z'):
            self.phot[channel]     = SparseArray(simspec.phot[channel],begin,end).mat
            self.skyphot[channel]  = SparseArray(simspec.skyphot[channel],begin,end).mat
        
        
            
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

    parser.add_argument("--ccd_npix_x", type=int, help="for testing; number of x (columns) to include in output", default=None)
    parser.add_argument("--ccd_npix_y", type=int, help="for testing; number of y (rows) to include in output", default=None)

    # parser.add_argument("--trimxy", action="store_true", help="Trim image to fit spectra")
    parser.add_argument("--verbose", action="store_true", help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw and simpix files")
    parser.add_argument("--seed", type=int, help="random number seed")
    parser.add_argument("--nspec", type=int, help="Number of spectra to simulate per camera %(default)s", default=500)
    parser.add_argument("--ncpu", type=int, help="Number of cpu cores per thread to use %(default)s", default=mp.cpu_count() // 2)
    parser.add_argument("--wavemin", type=float, help="Minimum wavelength to simulate")
    parser.add_argument("--wavemax", type=float, help="Maximum wavelength to simulate")
    parser.add_argument("--no-mpi", action="store_true", help="disable MPI")

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
    
    if args.no_mpi :
        comm=None
    
    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size
    
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

    mycameras = np.array_split(np.arange(ncamera, dtype=np.int32), 
        ngroup)[group]

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
    
    if comm is not None:
        comm.barrier()

    psf = None
    if args.psf is not None:
        from specter.psf import load_psf
        psf = load_psf(args.psf)
    
    
    # trick to have only one copy of the simspec
    # rank 0 reads the data and then distributes to each rank
    # a sparse sparsesimspec
    simspec = None
    if comm is None or rank == 0 :
        log.debug("reading simspec")
        simspec = io.read_simspec(args.simspec)
    
    if comm is not None : 
        if rank == 0:
            log.debug("rank 0 : distributing sparsesimspec")
            cams=np.array(args.cameras)[mycameras]
            log.debug("my cameras = {}".format(cams))
            
            specids=[]
            for camera in cams : specids.append(int(camera[1]))
            uspecids=np.unique(specids)
            log.debug("my spectrographs = {}".format(uspecids))
                        
            tmp = np.array_split(np.arange(500, dtype=np.int32), comm.size) # splitting fibers for a given camera among processes
            spec_of_ranks = []
            for orank in range(comm.size) :
                spec_of_rank = np.array([])
                for specid in uspecids :
                    spec_of_rank = np.append(spec_of_rank,tmp[orank]+500*specid)
                spec_of_ranks.append(spec_of_rank.astype(int))
            
            for other_rank in range(1,comm.size) :
                log.debug("rank 0 -> {} : sending {} spectra from {} to {}".format(other_rank,spec_of_ranks[other_rank].size,spec_of_ranks[other_rank][0],spec_of_ranks[other_rank][-1]))
                comm.send(SparseSimSpec(simspec,spec_of_ranks[other_rank][0],spec_of_ranks[other_rank][-1]+1,other_rank), dest=other_rank, tag=11)
            simspec = SparseSimSpec(simspec,spec_of_ranks[0][0],spec_of_ranks[0][-1],0)
        else :
            simspec = comm.recv(source=0, tag=11)
        if rank == 0:
            log.debug("rank {} : done".format(rank))
        
    
    """
    
    if comm is not None:
        # Broadcast one array at a time, since this is a 
        # very large object.
        flv = None
        wv = None
        pht = None
        if simspec is not None:
            flv = simspec.flavor
            wv = simspec.wave
            pht = simspec.phot
        flv = comm.bcast(flv, root=0)
        wv = comm.bcast(wv, root=0)
        pht = comm.bcast(pht, root=0)
        if simspec is None:
            simspec = SimSpec(flv, wv, pht)
        simspec.flux = comm.bcast(simspec.flux, root=0)
        simspec.skyflux = comm.bcast(simspec.skyflux, root=0)
        simspec.skyphot = comm.bcast(simspec.skyphot, root=0)
        simspec.metadata = comm.bcast(simspec.metadata, root=0)
        simspec.fibermap = comm.bcast(simspec.fibermap, root=0)
        simspec.obs = comm.bcast(simspec.obs, root=0)
        simspec.header = comm.bcast(simspec.header, root=0)
    """
    
    fibers = None
    if args.fibermap:
        if rank == 0:
            fibermap = desispec.io.read_fibermap(args.fibermap)
            fibers = fibermap['FIBER']
            if args.nspec is not None:
                fibers = fibers[0:args.nspec]
        if comm is not None:
            fibers = comm.bcast(fibers, root=0)

    # Use original seed to generate different random seeds for each camera
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 2**32-1, size=ncamera)

    image = {}
    rawpix = {}
    truepix = {}

    for c in mycameras:
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
            psf = desimodel.io.load_psf(channel)
            if args.ccd_npix_x is not None:
                psf.npix_x = args.ccd_npix_x
            if args.ccd_npix_y is not None:
                psf.npix_y = args.ccd_npix_y

        cosmics = None
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
            if comm_group is not None:
                cosmics = comm_group.bcast(cosmics, root=0)

        #- Do the actual simulation
        image[camera], rawpix[camera], truepix[camera] = \
            desisim.pixsim.simulate(camera, simspec, psf, fibers=fibers,
            nspec=args.nspec, ncpu=args.ncpu, cosmics=cosmics,
            wavemin=args.wavemin, wavemax=args.wavemax, preproc=False,
            comm=comm_group)

        if args.psf is None:
            del psf

    # Wait for all processes to finish their cameras
    if comm is not None:
        comm.barrier()

    # Write the cameras in order.  Only the rank zero process in each
    # group has the data.
    for c in np.arange(ncamera, dtype=np.int32):
        camera = args.cameras[c]
        if c in mycameras:
            if group_rank == 0:
                desispec.io.write_raw(rawtemp, rawpix[camera], 
                    camera=camera, header=image[camera].meta, 
                    primary_header=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, args.rawfile))
                io.write_simpix(simpixtemp, truepix[camera], 
                    camera=camera, meta=simspec.header)
                log.info('Wrote {} image to {}'.format(camera, 
                    args.simpixfile))
        if comm is not None:
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
        if len(mycameras) > 0:
            if group_rank == 0:
                for c in mycameras:
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
