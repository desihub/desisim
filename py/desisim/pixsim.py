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

import numpy as np

import desimodel.io
import desispec.io
from desispec.image import Image
import desispec.cosmics

from . import obs, io
from desiutil.log import get_logger
log = get_logger()


def simulate_frame(night, expid, camera, ccdshape=None, **kwargs):
    """
    Simulate a single frame, including I/O

    Args:
        night: YEARMMDD string
        expid: integer exposure ID
        camera: b0, r1, .. z9

    Options:
        ccdshape = (npix_y, npix_x) primarily used to limit memory while testing

    Additional keyword args are passed to pixsim.simulate()

    Reads:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits

    Writes:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simpix-{camera}-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/desi-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/pix-{camera}-{expid}.fits

    For a lower-level pixel simulation interface that doesn't perform I/O,
    see pixsim.simulate()
    """
    #- night, expid, camera -> input file names
    simspecfile = io.findfile('simspec', night=night, expid=expid)

    #- Read inputs
    psf = desimodel.io.load_psf(camera[0])
    simspec = io.read_simspec(simspecfile)

    #- Trim effective CCD size; mainly to limit memory for testing
    if ccdshape is not None:
        psf.npix_y, psf.npix_x = ccdshape

    if 'cosmics' in kwargs:
        shape = (psf.npix_y, psf.npix_x)
        kwargs['cosmics'] = io.read_cosmics(kwargs['cosmics'], expid, shape=shape)

    image, rawpix, truepix = simulate(camera, simspec, psf, **kwargs)

    #- Outputs; force "real" data files into simulation directory
    simpixfile = io.findfile('simpix', night=night, expid=expid, camera=camera)
    io.write_simpix(simpixfile, truepix, camera=camera, meta=image.meta)

    simdir = io.simdir(night=night)
    rawfile = desispec.io.findfile('desi', night=night, expid=expid)
    rawfile = os.path.join(simdir, os.path.basename(rawfile))
    desispec.io.write_raw(rawfile, rawpix, image.meta, camera=camera)

    pixfile = desispec.io.findfile('pix', night=night, expid=expid, camera=camera)
    pixfile = os.path.join(simdir, os.path.basename(pixfile))
    desispec.io.write_image(pixfile, image)


def simulate(camera, simspec, psf, fibers=None, nspec=None, ncpu=None,
    cosmics=None, wavemin=None, wavemax=None, preproc=True, comm=None):
    """Run pixel-level simulation of input spectra

    Args:
        camera (string) : b0, r1, .. z9
        simspec : desispec.io.SimSpec object e.g. from desispec.io.read_simspec()
        psf : subclass of specter.psf.psf.PSF, e.g. from desimodel.io.load_psf()
        fibers (array_like, optional):  fibers included in this simspec
        nspec (int, optional) : number of spectra to simulate
        ncpu (int, optional) : number of CPU cores to use in parallel
        cosmics (optional): desispec.image.Image object from desisim.io.read_cosmics()
        wavemin, wavemax (float, optional) : min/max wavelength range to simulate
        preproc (boolean, optional) : also preprocess raw data (default True)

    Returns:
        (image, rawpix, truepix) tuple, where image is the preprocessed Image object
            (only header is meaningful if preproc=False), rawpix is a 2D
            ndarray of unprocessed raw pixel data, and truepix is a 2D ndarray
            of truth for image.pix
    """

    if (comm is None) or (comm.rank == 0):
        log.info('Starting pixsim.simulate camera {} at {}'.format(camera,
            asctime()))
    #- parse camera name into channel and spectrograph number
    channel = camera[0].lower()

    ispec = int(camera[1])
    assert channel in 'brz', 'unrecognized channel {} camera {}'.format(channel, camera)
    assert 0 <= ispec < 10, 'unrecognized spectrograph {} camera {}'.format(ispec, camera)
    assert len(camera) == 2, 'unrecognized camera {}'.format(camera)

    #- Load DESI parameters
    params = desimodel.io.load_desiparams()

    #- this is not necessarily true, the truth in is the fibermap
    nfibers = params['spectro']['nfibers']

    if fibers is not None:
        fibers = np.asarray(fibers)
        allphot = simspec.phot[channel] + simspec.skyphot[channel]

        #- Trim to just fibers on this spectrograph
        ii = np.where(fibers//500 == ispec)[0]
        fibers = fibers[ii]

        phot = np.zeros((nfibers, allphot.shape[1]))
        phot[fibers%500] = allphot[ii]
    else:
        if ispec*nfibers >= simspec.nspec:
            msg = "camera {} not covered by simspec with {} spectra".format(
                camera, simspec.nspec)
            log.error(msg)
            raise ValueError(msg)

        phot = simspec.phot[channel] + simspec.skyphot[channel]

        #- Trim to just the fibers for this spectrograph
        if nspec is None:
            ii = slice(nfibers*ispec, nfibers*(ispec+1))
        else:
            ii = slice(nfibers*ispec, nfibers*ispec + nspec)

        phot = phot[ii]

    #- Trim wavelenths if needed
    wave = simspec.wave[channel]
    if wavemin is not None:
        ii = (wave >= wavemin)
        phot = phot[:, ii]
        wave = wave[ii]
    if wavemax is not None:
        ii = (wave <= wavemax)
        phot = phot[:, ii]
        wave = wave[ii]

    #- check if simulation has less than 500 input spectra
    if phot.shape[0] < nspec:
        nspec = phot.shape[0]

    #- Project to image and append that to file
    if (comm is None) or (comm.rank == 0):
        log.info('Starting {} projection at {}'.format(camera,
            asctime()))

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
        gain = params['ccd'][channel]['gain']
        for amp in ('1', '2', '3', '4'):
            header['GAIN'+amp] = gain

        #- Add cosmics from library of dark images
        ny = truepix.shape[0] // 2
        nx = truepix.shape[1] // 2
        if cosmics is not None:
            # set to zeros values with mask bit 0 (= dead column or hot pixels)
            cosmics_pix = cosmics.pix*((cosmics.mask&1)==0)
            pix = np.random.poisson(truepix) + cosmics_pix
            header['RDNOISE1'] = cosmics.meta['RDNOISE1']
            header['RDNOISE2'] = cosmics.meta['RDNOISE2']
            header['RDNOISE3'] = cosmics.meta['RDNOISE3']
            header['RDNOISE4'] = cosmics.meta['RDNOISE4']
        else:
            pix = truepix
            readnoise = params['ccd'][channel]['readnoise']
            header['RDNOISE1'] = readnoise
            header['RDNOISE2'] = readnoise
            header['RDNOISE3'] = readnoise
            header['RDNOISE4'] = readnoise

        if (comm is None) or (comm.rank == 0):
            log.info('RDNOISE1 {}'.format(header['RDNOISE1']))
            log.info('RDNOISE2 {}'.format(header['RDNOISE2']))
            log.info('RDNOISE3 {}'.format(header['RDNOISE3']))
            log.info('RDNOISE4 {}'.format(header['RDNOISE4']))

        #- data already has noise if cosmics were added
        noisydata = (cosmics is not None)

        #- Split by amplifier and expand into raw data
        nprescan = params['ccd'][channel]['prescanpixels']
        if 'overscanpixels' in params['ccd'][channel]:
            noverscan = params['ccd'][channel]['overscanpixels']
        else:
            noverscan = 50

        nyraw = ny
        nxraw = nx + nprescan + noverscan
        rawpix = np.empty( (nyraw*2, nxraw*2), dtype=np.int32 )

        #- Amp 0 Lower Left
        rawpix[0:nyraw, 0:nxraw] = \
            photpix2raw(pix[0:ny, 0:nx], gain, header['RDNOISE1'], 
                readorder='lr', nprescan=nprescan, noverscan=noverscan,
                noisydata=noisydata)

        #- Amp 2 Lower Right
        rawpix[0:nyraw, nxraw:nxraw+nxraw] = \
            photpix2raw(pix[0:ny, nx:nx+nx], gain, header['RDNOISE2'],
                readorder='rl', nprescan=nprescan, noverscan=noverscan,
                noisydata=noisydata)

        #- Amp 3 Upper Left
        rawpix[nyraw:nyraw+nyraw, 0:nxraw] = \
            photpix2raw(pix[ny:ny+ny, 0:nx], gain, header['RDNOISE3'],
                readorder='lr', nprescan=nprescan, noverscan=noverscan,
                noisydata=noisydata)

        #- Amp 4 Upper Right
        rawpix[nyraw:nyraw+nyraw, nxraw:nxraw+nxraw] = \
            photpix2raw(pix[ny:ny+ny, nx:nx+nx], gain, header['RDNOISE4'],
                readorder='rl', nprescan=nprescan, noverscan=noverscan,
                noisydata=noisydata)

        def xyslice2header(xyslice):
            '''
            convert 2D slice into IRAF style [a:b,c:d] header value

            e.g. xyslice2header(np.s_[0:10, 5:20]) -> '[6:20,1:10]'
            '''
            yy, xx = xyslice
            value = '[{}:{},{}:{}]'.format(xx.start+1, xx.stop, yy.start+1, yy.stop)
            return value

        #- Amp order from DESI-1964
        #-   3 4
        #-   1 2
        xoffset = nprescan+nx+noverscan
        header['PRESEC1']  = xyslice2header(np.s_[0:nyraw, 0:0+nprescan])
        header['DATASEC1'] = xyslice2header(np.s_[0:nyraw, nprescan:nprescan+nx])
        header['BIASSEC1'] = xyslice2header(np.s_[0:nyraw, nprescan+nx:nprescan+nx+noverscan])
        header['CCDSEC1']  = xyslice2header(np.s_[0:ny, 0:nx])

        header['PRESEC2']  = xyslice2header(np.s_[0:nyraw, xoffset+noverscan+nx:xoffset+noverscan+nx+nprescan])
        header['DATASEC2'] = xyslice2header(np.s_[0:nyraw, xoffset+noverscan:xoffset+noverscan+nx])
        header['BIASSEC2'] = xyslice2header(np.s_[0:nyraw, xoffset:xoffset+noverscan])
        header['CCDSEC2']  = xyslice2header(np.s_[0:ny, nx:2*nx])

        header['PRESEC3']  = xyslice2header(np.s_[nyraw:2*nyraw, 0:0+nprescan])
        header['DATASEC3'] = xyslice2header(np.s_[nyraw:2*nyraw, nprescan:nprescan+nx])
        header['BIASSEC3'] = xyslice2header(np.s_[nyraw:2*nyraw, nprescan+nx:nprescan+nx+noverscan])
        header['CCDSEC3']  = xyslice2header(np.s_[ny:2*ny, 0:nx])

        header['PRESEC4']  = xyslice2header(np.s_[nyraw:2*nyraw, xoffset+noverscan+nx:xoffset+noverscan+nx+nprescan])
        header['DATASEC4'] = xyslice2header(np.s_[nyraw:2*nyraw, xoffset+noverscan:xoffset+noverscan+nx])
        header['BIASSEC4'] = xyslice2header(np.s_[nyraw:2*nyraw, xoffset:xoffset+noverscan])
        header['CCDSEC4']  = xyslice2header(np.s_[ny:2*ny, nx:2*nx])

        if preproc:
            log.debug('Running preprocessing at {}'.format(asctime()))
            image = desispec.preproc.preproc(rawpix, header, primary_header=simspec.header)
        else:
            log.debug('Skipping preprocessing')
            image = Image(np.zeros(rawpix.shape), np.zeros(rawpix.shape), meta=header)

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
        noisydata (boolean, optional) : if True, don't add noise to the signal region,
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
        from mpi4py import MPI
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
            #we can use the same technique as multiprocessing to add the data back together
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
