"""
desisim.pixsim
==============

Tools for DESI pixel level simulations using specter
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import os.path
import random
from time import asctime
import socket

import astropy.units as u

import numpy as np

import desimodel.io
import desispec.io
from desispec.image import Image
import desispec.cosmics

from . import obs, io
from desiutil.log import get_logger
log = get_logger()

# Inhibit download of IERS-A catalog, even from a good server.
# Note that this is triggered by a call to astropy.time.Time(),
# which is subsequently used to compute sidereal_time().
# It's the initialization of astropy.time.Time() itself that makes the call.
from desiutil.iers import freeze_iers
from astropy.time import Time

def simulate_exposure(simspecfile, rawfile, cameras=None,
        ccdshape=None, simpixfile=None, addcosmics=None, comm=None,
        **kwargs):
    """
    Simulate frames from an exposure, including I/O

    Args:
        simspecfile: input simspec format file with spectra
        rawfile: output raw data file to write

    Options:
        cameras: str or list of str, e.g. b0, r1, .. z9
        ccdshape: (npix_y, npix_x) primarily used to limit memory while testing
        simpixfile: output file for noiseless truth pixels
        addcosmics: if True (must be specified via command input), add cosmics from real data
        comm: MPI communicator object

    Additional keyword args are passed to pixsim.simulate()

    For a lower-level pixel simulation interface that doesn't perform I/O,
    see pixsim.simulate()

    Note: call desi_preproc or desispec.preproc.preproc to pre-process the
    output desi*.fits file for overscan subtraction, noise estimation, etc.
    """
    #- Split communicator by nodes; each node processes N frames
    #- Assumes / requires equal number of ranks per node
    if comm is not None:
        rank, size = comm.rank, comm.size
        num_nodes = mpi_count_nodes(comm)
        comm_node, node_index, num_nodes = mpi_split_by_node(comm, 1)
        node_rank = comm_node.rank
        node_size = comm_node.size
    else:
        log.debug('Not using MPI')
        rank, size = 0, 1
        comm_node = None
        node_index = 0
        num_nodes = 1
        node_rank = 0
        node_size = 1

    if rank == 0:
        log.debug('Starting simulate_exposure at {}'.format(asctime()))

    if cameras is None:
        if rank == 0:
            from astropy.io import fits
            fibermap = fits.getdata(simspecfile, 'FIBERMAP')
            cameras = io.fibers2cameras(fibermap['FIBER'])
            log.debug('Found cameras {} in input simspec file'.format(cameras))
            if len(cameras) % num_nodes != 0:
                raise ValueError('Number of cameras {} should be evenly divisible by number of nodes {}'.format(
                    len(cameras), num_nodes))

    if comm is not None:
        cameras = comm.bcast(cameras, root=0)

    #- Fail early if camera alreaady in output file
    if rank == 0 and os.path.exists(rawfile):
        from astropy.io import fits
        err = False
        fx = fits.open(rawfile)
        for camera in cameras:
            if camera in fx:
                log.error('Camera {} already in {}'.format(camera, rawfile))
                err = True
        if err:
            raise ValueError('Some cameras already in output file')

    #- Read simspec input; I/O layer handles MPI broadcasting
    if rank == 0:
        log.debug('Reading simspec at {}'.format(asctime()))

    mycameras = cameras[node_index::num_nodes]
    if node_rank == 0:
        log.info("Assigning cameras {} to comm_exp node {}".format(mycameras, node_index))

    simspec = io.read_simspec(simspecfile, cameras=mycameras,
        readflux=False, comm=comm)
    night = simspec.header['NIGHT']
    expid = simspec.header['EXPID']

    if rank == 0:
        log.debug('Reading PSFs at {}'.format(asctime()))

    psfs = dict()
    #need to initialize previous channel
    previous_channel = 'a'
    for camera in mycameras:
        #- Note: current PSF object can't be pickled and thus every
        #- rank must read it instead of rank 0 read + bcast
        channel = camera[0]
        if channel not in psfs:
            log.info('Reading {} PSF at {}'.format(channel, asctime()))
            psfs[channel] = desimodel.io.load_psf(channel)

            #- Trim effective CCD size; mainly to limit memory for testing
            if ccdshape is not None:
                psfs[channel].npix_y, psfs[channel].npix_x = ccdshape

        psf = psfs[channel]

        cosmics=None
        #avoid re-broadcasting cosmics if we can
        if previous_channel != channel:
            if (addcosmics is True) and (node_rank == 0):
                cosmics_file = io.find_cosmics(camera, simspec.header['EXPTIME'])
                log.info('Reading cosmics templates {} at {}'.format(
                    cosmics_file, asctime()))
                shape = (psf.npix_y, psf.npix_x)
                cosmics = io.read_cosmics(cosmics_file, expid, shape=shape)
            if (addcosmics is True) and (comm_node is not None):
                if node_rank == 0:
                    log.info('Broadcasting cosmics at {}'.format(asctime()))
                cosmics = comm_node.bcast(cosmics, root=0)
            else:
                log.debug("Cosmics not requested")

        if node_rank == 0:
            log.info("Starting simulate for camera {} on node {}".format(camera,node_index))
        image, rawpix, truepix = simulate(camera, simspec, psf, comm=comm_node, preproc=False, cosmics=cosmics, **kwargs)

        #- Use input communicator as barrier since multiple sub-communicators
        #- will write to the same output file
        if rank == 0:
            log.debug('Writing outputs at {}'.format(asctime()))

        tmprawfile = rawfile + '.tmp'
        if comm is not None:
            for i in range(comm.size):
                if (i == comm.rank) and (comm_node.rank == 0):
                    desispec.io.write_raw(tmprawfile, rawpix, image.meta,
                                          camera=camera)
                    if simpixfile is not None:
                        io.write_simpix(simpixfile, truepix, camera=camera,
                                        meta=image.meta)
                comm.barrier()
        else:
            desispec.io.write_raw(tmprawfile, rawpix, image.meta, camera=camera)
            if simpixfile is not None:
                io.write_simpix(simpixfile, truepix, camera=camera,
                                meta=image.meta)

        if rank == 0:
            log.info('Wrote {}'.format(rawfile))
            log.debug('done at {}'.format(asctime()))

        previous_channel = channel

    #- All done; rename temporary raw file to final location
    if comm is None or comm.rank == 0:
        os.rename(tmprawfile, rawfile)


def simulate(camera, simspec, psf, nspec=None, ncpu=None,
    cosmics=None, wavemin=None, wavemax=None, preproc=True, comm=None):
    """Run pixel-level simulation of input spectra

    Args:
        camera (string) : b0, r1, .. z9
        simspec : desispec.io.SimSpec object from desispec.io.read_simspec()
        psf : subclass of specter.psf.psf.PSF, e.g. from desimodel.io.load_psf()

    Options:
        nspec (int): number of spectra to simulate
        ncpu (int): number of CPU cores to use in parallel
        cosmics (desispec.image.Image): e.g. from desisim.io.read_cosmics()
        wavemin (float): minimum wavelength range to simulate
        wavemax (float): maximum wavelength range to simulate
        preproc (boolean, optional) : also preprocess raw data (default True)

    Returns:
        (image, rawpix, truepix) tuple, where image is the preproc Image object
            (only header is meaningful if preproc=False), rawpix is a 2D
            ndarray of unprocessed raw pixel data, and truepix is a 2D ndarray
            of truth for image.pix
    """

    freeze_iers()
    if (comm is None) or (comm.rank == 0):
        log.info('Starting pixsim.simulate camera {} at {}'.format(camera,
            asctime()))
    #- parse camera name into channel and spectrograph number
    channel = camera[0].lower()
    ispec = int(camera[1])
    assert channel in 'brz', \
        'unrecognized channel {} camera {}'.format(channel, camera)
    assert 0 <= ispec < 10, \
        'unrecognized spectrograph {} camera {}'.format(ispec, camera)
    assert len(camera) == 2, \
        'unrecognized camera {}'.format(camera)

    #- Load DESI parameters
    params = desimodel.io.load_desiparams()

    #- this is not necessarily true, the truth in is the fibermap
    nfibers = params['spectro']['nfibers']

    phot = simspec.cameras[camera].phot
    if simspec.cameras[camera].skyphot is not None:
        phot += simspec.cameras[camera].skyphot

    if nspec is not None:
        phot = phot[0:nspec]
    else:
        nspec = phot.shape[0]

    #- Trim wavelengths if needed
    wave = simspec.cameras[camera].wave
    if wavemin is not None:
        ii = (wave >= wavemin)
        phot = phot[:, ii]
        wave = wave[ii]
    if wavemax is not None:
        ii = (wave <= wavemax)
        phot = phot[:, ii]
        wave = wave[ii]

    #- Project to image and append that to file
    if (comm is None) or (comm.rank == 0):
        log.info('Starting {} projection at {}'.format(camera, asctime()))

    # The returned true pixel values will only exist on rank 0 in the
    # MPI case.  Otherwise it will be None.
    truepix = parallel_project(psf, wave, phot, ncpu=ncpu, comm=comm)

    if (comm is None) or (comm.rank == 0):
        log.info('Finished {} projection at {}'.format(camera,
            asctime()))

    image = None
    rawpix = None
    if (comm is None) or (comm.rank == 0):
        #- Start metadata header
        header = simspec.header.copy()
        header['CAMERA'] = camera
        header['DOSVER'] = 'SIM'
        header['FEEVER'] = 'SIM'
        header['DETECTOR'] = 'SIM'

        #- Add cosmics from library of dark images
        ny = truepix.shape[0] // 2
        nx = truepix.shape[1] // 2
        if cosmics is not None:
            # set to zeros values with mask bit 0 (= dead column or hot pixels)
            cosmics_pix = cosmics.pix*((cosmics.mask&1)==0)
            pix = np.random.poisson(truepix) + cosmics_pix
            try:  #- cosmics templates >= v0.3
                rdnoiseA = cosmics.meta['OBSRDNA']
                rdnoiseB = cosmics.meta['OBSRDNB']
                rdnoiseC = cosmics.meta['OBSRDNC']
                rdnoiseD = cosmics.meta['OBSRDND']
            except KeyError:  #- cosmics templates <= v0.2
                print(cosmic.meta)
                rdnoiseA = cosmics.meta['RDNOISE0']
                rdnoiseB = cosmics.meta['RDNOISE1']
                rdnoiseC = cosmics.meta['RDNOISE2']
                rdnoiseD = cosmics.meta['RDNOISE3']
        else:
            pix = truepix
            readnoise = params['ccd'][channel]['readnoise']
            rdnoiseA = rdnoiseB = rdnoiseC = rdnoiseD = readnoise

        #- data already has noise if cosmics were added
        noisydata = (cosmics is not None)

        #- Split by amplifier and expand into raw data
        nprescan = params['ccd'][channel]['prescanpixels']
        if 'overscanpixels' in params['ccd'][channel]:
            noverscan = params['ccd'][channel]['overscanpixels']
        else:
            noverscan = 50

        #- Reproducibly random overscan bias level offsets across diff exp
        assert channel in 'brz'
        if channel == 'b':
            irand = ispec
        elif channel == 'r':
            irand = 10 + ispec
        elif channel == 'z':
            irand = 20 + ispec

        seeds = np.random.RandomState(0).randint(2**32-1, size=30)
        rand = np.random.RandomState(seeds[irand])

        nyraw = ny
        nxraw = nx + nprescan + noverscan
        rawpix = np.empty( (nyraw*2, nxraw*2), dtype=np.int32 )

        gain = params['ccd'][channel]['gain']

        #- Amp A/1 Lower Left
        rawpix[0:nyraw, 0:nxraw] = \
            photpix2raw(pix[0:ny, 0:nx], gain, rdnoiseA,
                readorder='lr', nprescan=nprescan, noverscan=noverscan,
                offset=rand.uniform(100, 200),
                noisydata=noisydata)

        #- Amp B/2 Lower Right
        rawpix[0:nyraw, nxraw:nxraw+nxraw] = \
            photpix2raw(pix[0:ny, nx:nx+nx], gain, rdnoiseB,
                readorder='rl', nprescan=nprescan, noverscan=noverscan,
                offset=rand.uniform(100, 200),
                noisydata=noisydata)

        #- Amp C/3 Upper Left
        rawpix[nyraw:nyraw+nyraw, 0:nxraw] = \
            photpix2raw(pix[ny:ny+ny, 0:nx], gain, rdnoiseC,
                readorder='lr', nprescan=nprescan, noverscan=noverscan,
                offset=rand.uniform(100, 200),
                noisydata=noisydata)

        #- Amp D/4 Upper Right
        rawpix[nyraw:nyraw+nyraw, nxraw:nxraw+nxraw] = \
            photpix2raw(pix[ny:ny+ny, nx:nx+nx], gain, rdnoiseD,
                readorder='rl', nprescan=nprescan, noverscan=noverscan,
                offset=rand.uniform(100, 200),
                noisydata=noisydata)

        def xyslice2header(xyslice):
            '''
            convert 2D slice into IRAF style [a:b,c:d] header value

            e.g. xyslice2header(np.s_[0:10, 5:20]) -> '[6:20,1:10]'
            '''
            yy, xx = xyslice
            value = '[{}:{},{}:{}]'.format(xx.start+1, xx.stop,
                                           yy.start+1, yy.stop)
            return value

        #- Amp order from DESI-1964 (previously 1-4 instead of A-D)
        #-   C D
        #-   A B
        xoffset = nprescan+nx+noverscan
        header['PRESECA']  = xyslice2header(np.s_[0:nyraw, 0:0+nprescan])
        header['DATASECA'] = xyslice2header(np.s_[0:nyraw, nprescan:nprescan+nx])
        header['BIASSECA'] = xyslice2header(np.s_[0:nyraw, nprescan+nx:nprescan+nx+noverscan])
        header['CCDSECA']  = xyslice2header(np.s_[0:ny, 0:nx])

        header['PRESECB']  = xyslice2header(np.s_[0:nyraw, xoffset+noverscan+nx:xoffset+noverscan+nx+nprescan])
        header['DATASECB'] = xyslice2header(np.s_[0:nyraw, xoffset+noverscan:xoffset+noverscan+nx])
        header['BIASSECB'] = xyslice2header(np.s_[0:nyraw, xoffset:xoffset+noverscan])
        header['CCDSECB']  = xyslice2header(np.s_[0:ny, nx:2*nx])

        header['PRESECC']  = xyslice2header(np.s_[nyraw:2*nyraw, 0:0+nprescan])
        header['DATASECC'] = xyslice2header(np.s_[nyraw:2*nyraw, nprescan:nprescan+nx])
        header['BIASSECC'] = xyslice2header(np.s_[nyraw:2*nyraw, nprescan+nx:nprescan+nx+noverscan])
        header['CCDSECC']  = xyslice2header(np.s_[ny:2*ny, 0:nx])

        header['PRESECD']  = xyslice2header(np.s_[nyraw:2*nyraw, xoffset+noverscan+nx:xoffset+noverscan+nx+nprescan])
        header['DATASECD'] = xyslice2header(np.s_[nyraw:2*nyraw, xoffset+noverscan:xoffset+noverscan+nx])
        header['BIASSECD'] = xyslice2header(np.s_[nyraw:2*nyraw, xoffset:xoffset+noverscan])
        header['CCDSECD']  = xyslice2header(np.s_[ny:2*ny, nx:2*nx])

        #- Add additional keywords to mimic real raw data
        header['INSTRUME'] = 'DESI'
        header['PROCTYPE'] = 'RAW'
        header['PRODTYPE'] = 'image'
        header['EXPFRAME'] = 0
        header['REQTIME'] = simspec.header['EXPTIME']
        header['TIMESYS'] = 'UTC'
        #- DATE-OBS format YEAR-MM-DDThh:mm:ss.sss -> OBSID kpnoYEARMMDDthhmmss
        header['OBSID']='kp4m'+header['DATE-OBS'][0:19].replace('-','').replace(':','').lower()
        header['TIME-OBS'] = header['DATE-OBS'].split('T')[1]
        header['DELTARA'] = 0.0
        header['DELTADEC'] = 0.0
        header['SPECGRPH'] = ispec
        header['CCDNAME'] = 'CCDS' + str(ispec) + str(channel).upper()
        header['CCDPREP'] = 'purge,clear'
        header['CCDSIZE'] = str(rawpix.shape)
        header['CCDTEMP'] = 850.0
        header['CPUTEMP'] = 63.7
        header['CASETEMP'] = 62.8
        header['CCDTMING'] = 'sim_timing.txt'
        header['CCDCFG'] = 'sim.cfg'
        header['SETTINGS'] = 'sim_detectors.json'
        header['VESSEL'] = 7  #- I don't know what this is
        header['FEEBOX'] = 'sim097'
        header['PGAGAIN'] = 5
        header['OCSVER'] = 'SIM'
        header['CONSTVER'] = 'SIM'
        header['BLDTIME'] = 0.35
        header['DIGITIME'] = 61.9

        #- Remove some spurious header keywords from upstream
        if 'BUNIT' in header and header['BUNIT'] == 'Angstrom':
            del header['BUNIT']

        if 'MJD' in header and 'MJD-OBS' not in header:
            header['MJD-OBS'] = header['MJD']
            del header['MJD']

        for key in ['RA', 'DEC']:
            if key in header:
                del header[key]

        #- Drive MJD-OBS from DATE-OBS if needed
        if 'MJD-OBS' not in header:
            header['MJD-OBS'] = Time(header['DATE-OBS']).mjd

        #- from http://www-kpno.kpno.noao.edu/kpno-misc/mayall_params.html
        kpno_longitude = -(111. + 35/60. + 59.6/3600) * u.deg

        #- Convert DATE-OBS to sexigesimal (sigh) Local Sidereal Time
        #- Use mean ST as close enough for sims to avoid nutation calc
        t = Time(header['DATE-OBS'])
        st = t.sidereal_time('mean', kpno_longitude).to('deg').value
        hour = st/15
        minute = (hour % 1)*60
        second = (minute % 1)*60
        header['ST'] = '{:02d}:{:02d}:{:0.3f}'.format(
                int(hour), int(minute), second)

        if preproc:
            log.debug('Running preprocessing at {}'.format(asctime()))
            image = desispec.preproc.preproc(rawpix, header, primary_header=simspec.header)
        else:
            log.debug('Skipping preprocessing')
            image = Image(np.zeros(truepix.shape), np.zeros(truepix.shape), meta=header)

    if (comm is None) or (comm.rank == 0):
        log.info('Finished pixsim.simulate for camera {} at {}'.format(camera,
            asctime()))

    return image, rawpix, truepix


def photpix2raw(phot, gain=1.0, readnoise=3.0, offset=None,
    nprescan=7, noverscan=50, readorder='lr', noisydata=True):
    '''
    Add prescan, overscan, noise, and integerization to an image

    Args:
        phot: 2D float array of mean input photons per pixel
        gain (float, optional): electrons/ADU
        readnoise (float, optional): CCD readnoise in electrons
        offset (float, optional): bias offset to add
        nprescan (int, optional): number of prescan pixels to add
        noverscan (int, optional): number of overscan pixels to add
        readorder (str, optional): 'lr' or 'rl' to indicate readout order
            'lr' : add prescan on left and overscan on right of image
            'rl' : add prescan on right and overscan on left of image
        noisydata (boolean, optional) : if True, don't add noise,
            e.g. because input signal already had noise from a cosmics image

    Returns 2D integer ndarray:
        image = int((poisson(phot) + offset + gauss(readnoise))/gain)

    Integerization happens twice: the mean photons are poisson sampled
    into integers, but then offets, readnoise, and gain are applied before
    resampling into ADU integers

    This is intended to be used per-amplifier, not for an entire CCD image.
    '''
    ny = phot.shape[0]
    nx = phot.shape[1] + nprescan + noverscan

    #- reading from right to left is effectively swapping pre/overscan counts
    if readorder.lower() in ('rl', 'rightleft'):
        nprescan, noverscan = noverscan, nprescan

    img = np.zeros((ny, nx), dtype=float)
    img[:, nprescan:nprescan+phot.shape[1]] = phot

    if offset is None:
        offset = np.random.uniform(100, 200)

    if noisydata:
        #- Data already has noise; just add offset and noise to pre/overscan
        img += offset
        img[0:ny, 0:nprescan] += np.random.normal(scale=readnoise, size=(ny, nprescan))
        ix = phot.shape[1] + nprescan
        img[0:ny, ix:ix+noverscan] += np.random.normal(scale=readnoise, size=(ny, noverscan))
        img /= gain

    else:
        #- Add offset and noise to everything
        noise = np.random.normal(loc=offset, scale=readnoise, size=img.shape)
        img = np.random.poisson(img) + noise
        img /= gain

    return img.astype(np.int32)


#- Helper function for multiprocessing parallel project
def _project(args):
    """
    Helper function to project photons onto a subimage

    Args:
        tuple/array of [psf, wave, phot, specmin]

    Returns (xyrange, subimage) such that
        xmin, xmax, ymin, ymax = xyrange
        image[ymin:ymax, xmin:xmax] += subimage
    """
    try:
        psf, wave, phot, specmin = args
        nspec = phot.shape[0]
        if phot.shape[-1] != wave.shape[-1]:
            raise ValueError('phot.shape {} vs. wave.shape {} mismatch'.format(phot.shape, wave.shape))

        xyrange = psf.xyrange( [specmin, specmin+nspec], wave )
        img = psf.project(wave, phot, specmin=specmin, xyrange=xyrange)
        return (xyrange, img)
    except Exception as e:
        if os.getenv('UNITTEST_SILENT') is None:
            import traceback
            print('-'*60)
            print('ERROR in _project', psf.wmin, psf.wmax, wave[0], wave[-1], phot.shape, specmin)
            traceback.print_exc()
            print('-'*60)

        raise e


#- Move this into specter itself?
def parallel_project(psf, wave, phot, specmin=0, ncpu=None, comm=None):
    """
    Using psf, project phot[nspec, nw] vs. wave[nw] onto image

    Return 2D image
    """
    img = None
    if comm is not None:
        # MPI version

        # Get a smaller communicator if not enough spectra
        nspec = phot.shape[0]
        if nspec < comm.size:
            keep = int(comm.rank < nspec)
            comm = comm.Split(color=keep)
            if not keep:
                return None

        specs = np.arange(phot.shape[0], dtype=np.int32)
        myspecs = np.array_split(specs, comm.size)[comm.rank]
        nspec = phot.shape[0]
        iispec = np.linspace(specmin, nspec, int(comm.size+1)).astype(int)

        args = list()
        if comm.rank == 0:
            for i in range(comm.size):
                if iispec[i+1] > iispec[i]:
                    args.append( [psf, wave, phot[iispec[i]:iispec[i+1]], iispec[i]] )

        args=comm.scatter(args,root=0)
        #now that all ranks have args, we can call _project
        xy_subimg=_project(args)
        #_project calls project calls spotgrid etc

        xy_subimg=comm.gather(xy_subimg,root=0)

        if comm.rank ==0:
            #now all the data should be back at rank 0
            # use same technique as multiprocessing to recombine the data
            img = np.zeros( (psf.npix_y, psf.npix_x) )
            for xyrange, subimg in xy_subimg:
                xmin, xmax, ymin, ymax = xyrange
                img[ymin:ymax, xmin:xmax] += subimg

    #end of mpi section

    else:
        import multiprocessing as mp
        if ncpu is None:
            # Avoid hyperthreading
            ncpu = mp.cpu_count() // 2
        if ncpu <= 1:
            #- Serial version
            log.debug('Not using multiprocessing (ncpu={})'.format(ncpu))
            img = psf.project(wave, phot, specmin=specmin)
        else:
            #- multiprocessing version
            #- Split the spectra into ncpu groups
            log.debug('Using multiprocessing (ncpu={})'.format(ncpu))
            nspec = phot.shape[0]
            iispec = np.linspace(specmin, nspec, ncpu+1).astype(int)
            args = list()
            for i in range(ncpu):
                if iispec[i+1] > iispec[i]:  #- can be false if nspec < ncpu
                    args.append( [psf, wave, phot[iispec[i]:iispec[i+1]], iispec[i]] )

            #- Create pool of workers to do the projection using _project
            #- xyrange, subimg = _project( [psf, wave, phot, specmin] )
            pool = mp.Pool(ncpu)
            xy_subimg = pool.map(_project, args)

            #print("xy_subimg from pool")
            #print(xy_subimg)
            #print(len(xy_subimg))

            img = np.zeros( (psf.npix_y, psf.npix_x) )
            for xyrange, subimg in xy_subimg:
                xmin, xmax, ymin, ymax = xyrange
                img[ymin:ymax, xmin:xmax] += subimg

            #- Prevents hangs of Travis tests
            pool.close()
            pool.join()

    return img


def get_nodes_per_exp(nnodes,nexposures,ncameras,user_nodes_per_comm_exp=None):
    """
    Calculate how many nodes to use per exposure

    Args:
        nnodes: number of nodes in MPI COMM_WORLD (not number of ranks)
        nexposures: number of exposures to process
        ncameras: number of cameras per exposure
        user_nodes_per_comm_exp (int, optional): user override of number of
            nodes to use; used to check requirements

    Returns number of nodes to include in sub-communicators used to process
    individual exposures

    Notes:
        * Uses the largest number of nodes per exposure that will still
          result in efficient node usage
        * requires that (nexposures*ncameras) / nnodes = int
        * the derived nodes_per_comm_exp * nexposures / nodes = int
        * See desisim.test.test_pixsim.test_get_nodes_per_exp() for examples
        * if user_nodes_per_comm_exp is given, requires that
          GreatestCommonDivisor(nnodes, ncameras) / user_nodes_per_comm_exp = int
    """

    from math import gcd
    import desiutil.log as logging
    log = logging.get_logger()
    log.setLevel(logging.INFO)

    #check if nframes is evenly divisible by nnodes
    nframes = ncameras*nexposures
    if nframes % nnodes !=0:
        ### msg=("nframes {} must be evenly divisible by nnodes {}, try again".format(nframes, nnodes))
        ### raise ValueError(msg)
        msg=("nframes {} is not evenly divisible by nnodes {}; packing will be inefficient".format(nframes, nnodes))
        log.warning(msg)
    else:
        log.debug("nframes {} is evenly divisible by nnodes {}, check passed".format(nframes, nnodes))

    #find greatest common divisor between nnodes and ncameras
    #greatest common divisor = greatest common factor
    #we use python's built in gcd
    greatest_common_factor=gcd(nnodes,ncameras)
    #the greatest common factor must be greater than one UNLESS we are on one node
    if nnodes > 1:
        if greatest_common_factor == 1:
            msg=("greatest common factor {} between nnodes {} and nframes {} must be larger than one, try again".format(greatest_common_factor, nnodes, nframes))
            raise ValueError(msg)
        else:
            log.debug("greatest common factor {} between nnodes {} and nframes {} is greater than one, check passed".format(greatest_common_factor, nnodes, nframes))

    #check to make sure the user hasn't specified a really asinine value of user_nodes_per_comm_exp
    if user_nodes_per_comm_exp is not None:
        if greatest_common_factor % user_nodes_per_comm_exp !=0:
            msg=("user-specified value of user_nodes_per_comm_exp {} is bad, try again".format(user_nodes_per_comm_exp))
            raise ValueError(msg)
        else:
            log.debug("user-specified value of user_nodes_per_comm_exp {} is good, check passed".format(user_nodes_per_comm_exp))
            nodes_per_comm_exp=user_nodes_per_comm_exp
    #if the user didn't specify anything, use the greatest common factor
    if user_nodes_per_comm_exp is None:
        nodes_per_comm_exp=greatest_common_factor

    #finally check to make sure exposures*gcf/nnodes is an integer to avoid inefficient node use
    if (nexposures*nodes_per_comm_exp) % nnodes != 0:
        ### msg=("nexposures {} * nodes_per_comm_exp {} does not divide evenly into nnodes {}, try again".format(nexposures, nodes_per_comm_exp, nnodes))
        ### raise ValueError(msg)
        msg=("nexposures {} * nodes_per_comm_exp {} does not divide evenly into nnodes {}; packing will be inefficient".format(nexposures, nodes_per_comm_exp, nnodes))
        log.warning(msg)
    else:
        log.debug("nexposures {} * nodes_per_comm_exp {} divides evenly into nnodes {}, check passed".format(nexposures, nodes_per_comm_exp, nnodes))


    return nodes_per_comm_exp

#-------------------------------------------------------------------------
#- MPI utility functions
#- These functions assist with splitting a communicator across node boundaries.
#- That constraint isn't required by MPI, but can be convenient for humans
#- thinking about "I want to process one camera with one node" or "I want to
#- process 6 exposures with 20 nodes using 10 nodes per exposure"

def mpi_count_nodes(comm):
    '''
    Return the number of nodes in this communicator
    '''
    nodenames = comm.allgather(socket.gethostname())
    num_nodes=len(set(nodenames))
    return num_nodes

def mpi_split_by_node(comm, nodes_per_communicator):
    '''
    Split an MPI communicator into sub-communicators with integer numbers
    of nodes per communicator

    Args:
        comm: MPI communicator
        nodes_per_communicator: number of nodes per sub-communicator

    Returns:
        MPI sub-communicator, node_index, total_num_nodes

    Notes:
      * total number of nodes in original communicator must be an integer
        multiple of nodes_per_communicator
      * if comm is split into N sub-communicators, node_index is the index
        of which of the N is returned for this rank
      * total_num_nodes = number of nodes in original communicator
    '''
    num_nodes = mpi_count_nodes(comm)

    if comm.size % num_nodes != 0:
        raise ValueError('Variable number of ranks per node')
    if num_nodes % nodes_per_communicator != 0:
        raise ValueError('Input number of nodes {} must be divisible by nodes_per_communicator {}'.format(
            num_nodes, nodes_per_communicator))

    ranks_per_communicator = comm.size // (num_nodes // nodes_per_communicator)
    node_index = comm.rank // ranks_per_communicator
    comm_node = comm.Split(color = node_index)

    return comm_node, node_index, num_nodes
