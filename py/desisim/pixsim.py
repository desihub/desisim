"""
Tools for DESI pixel level simulations using specter
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import os.path
import multiprocessing as mp
import random
from time import asctime

import numpy as np

import desimodel.io
import desispec.io
from desispec.image import Image
import desispec.cosmics

from desisim import obs, io
from desispec.log import get_logger
log = get_logger()

def simulate_frame(night, expid, camera, **kwargs):
    """
    Simulate a single frame, including I/O
    
    Args:
        night: YEARMMDD string
        expid: integer exposure ID
        camera: b0, r1, .. z9

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
    cosmics=None, wavemin=None, wavemax=None):
    """
    Run pixel-level simulation of input spectra
    
    Args:
        camera (string) : b0, r1, .. z9
        simspec : desispec.io.SimSpec object e.g. from desispec.io.read_simspec()
        psf : subclass of specter.psf.psf.PSF, e.g. from desimodel.io.load_psf()

    Optional:
        fibers (array_like):  fibers included in this simspec
        nspec (int) : number of spectra to simulate
        ncpu (int) : number of CPU cores to use in parallel
        cosmics : desispec.image.Image object from desisim.io.read_cosmics()
        wavemin, wavemax (float) : min/max wavelength range to simulate

    Returns (image, rawpix, truepix) tuple, where
        image : preprocessed Image object
        rawpix : 2D ndarray of unprocessed raw pixel data
        truepix : 2D ndarray of truth for image.pix    
    """

    log.info('Starting pixsim.simulate {}'.format(asctime()))
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
        
        log.debug('Simulating fibers {}'.format(fibers))

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
    log.info("Projecting photons onto {} CCD".format(camera))        
    truepix = parallel_project(psf, wave, phot, ncpu=ncpu)

    #- Start metadata header
    header = simspec.header.copy()
    header['CAMERA'] = camera
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
        photpix2raw(pix[0:ny, 0:nx], gain, header['RDNOISE1'], readorder='lr',
            nprescan=nprescan, noverscan=noverscan, noisydata=noisydata)

    #- Amp 2 Lower Right
    rawpix[0:nyraw, nxraw:nxraw+nxraw] = \
        photpix2raw(pix[0:ny, nx:nx+nx], gain, header['RDNOISE2'], readorder='rl',
            nprescan=nprescan, noverscan=noverscan, noisydata=noisydata)

    #- Amp 3 Upper Left
    rawpix[nyraw:nyraw+nyraw, 0:nxraw] = \
        photpix2raw(pix[ny:ny+ny, 0:nx], gain, header['RDNOISE3'], readorder='lr',
            nprescan=nprescan, noverscan=noverscan, noisydata=noisydata)

    #- Amp 4 Upper Right
    rawpix[nyraw:nyraw+nyraw, nxraw:nxraw+nxraw] = \
        photpix2raw(pix[ny:ny+ny, nx:nx+nx], gain, header['RDNOISE4'], readorder='rl',
            nprescan=nprescan, noverscan=noverscan, noisydata=noisydata)

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

    image = desispec.preproc.preproc(rawpix, header)
    log.info('Finished pixsim.simulate {}'.format(asctime()))

    return image, rawpix, truepix

def photpix2raw(phot, gain=1.0, readnoise=3.0, offset=None,
    nprescan=7, noverscan=50, readorder='lr', noisydata=True):
    '''
    Add prescan, overscan, noise, and integerization to an image

    Args:
        phot: 2D float array of mean input photons per pixel
        
    Options:
        gain (float): electrons/ADU
        readnoise (float): CCD readnoise in electrons
        offset (float): bias offset to add
        nprescan (int): number of prescan pixels to add
        noverscan (int): number of overscan pixels to add
        readorder : 'lr' or 'rl' to indicate readout order
            'lr' : add prescan on left and overscan on right of image
            'rl' : add prescan on right and overscan on left of image
        noisydata (boolean) : if True, don't add noise to the signal region,
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
        xyrange = psf.xyrange( [specmin, specmin+nspec], wave )
        img = psf.project(wave, phot, specmin=specmin, xyrange=xyrange)
        return (xyrange, img)
    except Exception, e:
        if os.getenv('UNITTEST_SILENT') is None:
            import traceback
            print('-'*60)
            print('ERROR in _project', psf.wmin, psf.wmax, wave[0], wave[-1], phot.shape, specmin)
            traceback.print_exc()
            print('-'*60)
        raise e

#- Move this into specter itself?
def parallel_project(psf, wave, phot, ncpu=None):
    """
    Using psf, project phot[nspec, nw] vs. wave[nw] onto image
    
    Return 2D image
    """
    import multiprocessing as mp
    if ncpu is None:
        #- on a Mac, 1/2 cores is about the same speed as all of them
        ncpu = mp.cpu_count() // 2

    if ncpu <= 1:
        #- Serial version
        ### print "Serial project"
        log.debug('Not using multiprocessing (ncpu={})'.format(ncpu))
        img = psf.project(wave, phot)
    else:
        #- multiprocessing version
        #- Split the spectra into ncpu groups
        log.debug('Using multiprocessing (ncpu={})'.format(ncpu))
        nspec = phot.shape[0]
        iispec = np.linspace(0, nspec, ncpu+1).astype(int)
        args = list()
        for i in range(ncpu):
            if iispec[i+1] > iispec[i]:  #- can be false if nspec < ncpu
                args.append( [psf, wave, phot[iispec[i]:iispec[i+1]], iispec[i]] )

        #- Create pool of workers to do the projection using _project
        #- xyrange, subimg = _project( [psf, wave, phot, specmin] )
        ### print "parallel_project {} groups with {} CPU cores".format(len(args), ncpu)

        pool = mp.Pool(ncpu)
        xy_subimg = pool.map(_project, args)
        img = np.zeros( (psf.npix_y, psf.npix_x) )
        for xyrange, subimg in xy_subimg:
            xmin, xmax, ymin, ymax = xyrange
            img[ymin:ymax, xmin:xmax] += subimg
            
    return img
    
