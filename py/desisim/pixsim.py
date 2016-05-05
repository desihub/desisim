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

'''
#- specify night and expid; derive input and output filenames
pixsim --night 20160102 --expid 23

#- get night and expid from input simspec file
pixsim --simspec SIMSPEC

#- optionally override night and expid
pixsim --simspec SIMSPEC --night 20160102 --expid 23

#- override outputs
pixsim --simspec SIMSPEC --outrawfile blat.fits
'''

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
    #- expand camera list
    if args.cameras is None:
        args.cameras = list()
        args.spectrographs = [int(x) for x in args.spectrographs.split(',')]
        for arm in args.arms.split(','):
            for ispec in args.spectrographs:
                args.cameras.append(arm+str(ispec))
    else:
        args.cameras = args.cameras.split(',')

    if args.simspec is None:
        if args.night is None or args.expid is None:
            msg = 'Must set --simspec or both --night and --expid'
            log.error(msg)
            raise ValueError(msg)
        args.simspec = io.findfile('simspec', args.night, args.expid)
    else:
        if (args.night is None) or (args.expid is None):
            from astropy.io import fits
            hdr = fits.getheader(args.simspec)
        if args.night is None:
            args.night = str(hdr['NIGHT'])
        if args.expid is None:
            args.expid = int(hdr['EXPID'])

    #- write to same directory as simspec
    if args.rawfile is None:
        rawfile = os.path.basename(desispec.io.findfile('raw', args.night, args.expid))
        args.rawfile = os.path.join(os.path.dirname(args.simspec), rawfile)
        
    if args.pixfile or args.simpixfile:
        if len(args.cameras) != 1:
            msg = 'if --pixfile or --simpixfile, must specify a single --camera'
            log.error(msg)
            raise ValueError(msg)

#-------------------------------------------------------------------------
#- Parse options
def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Generates simulated DESI pixel-level raw data',
        )

    #- Input files
    parser.add_argument("--psf", type=str, help="PSF filename")
    parser.add_argument("--cosmics", type=str, help="fits file with dark images with cosmics to add")
    parser.add_argument("--simspec", type=str, help="input simspec file")
        
    #- Output files
    parser.add_argument("--rawfile", type=str, help="output raw data file")
    parser.add_argument("--pixfile", type=str, help="output preprocessed image file")
    parser.add_argument("--simpixfile", type=str, help="output truth image file")
        
    parser.add_argument("--night", type=str, help="YEARMMDD")
    parser.add_argument("--expid", type=int, help="exposure id")
    parser.add_argument("--cameras", type=str, help="cameras, e.g. b0,r5,z9")

    parser.add_argument("--spectrographs", type=str, help="spectrograph numbers, e.g. 0,1,9", default='0')
    parser.add_argument("--arms", type=str, help="spectrograph arms, e.g. b,r,z", default='b,r,z')

    parser.add_argument("--preproc", action="store_true", help="preprocess raw -> pix files")
    # parser.add_argument("--trimxy", action="store_true", help="Trim image to fit spectra")
    parser.add_argument("--seed", type=int, help="random number seed")
    parser.add_argument("--nspec", type=int, help="Number of spectra to simulate per camera [%default]", default=500)
    parser.add_argument("--ncpu",  type=int, help="Number of cpu cores to use %(default)s", default=mp.cpu_count() // 2)
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
    log.info('Starting pixsim {}'.format(asctime()))
    if args is None:
        args = parse()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.psf is not None:
        import specter.io
        psf = specter.io.load_psf(args.psf)

    simspec = io.read_simspec(args.simspec)

    for camera in args.cameras:
        channel = camera[0].lower()
        assert channel in ('b', 'r', 'z'), "Unknown camera {} doesn't start with b,r,z".format(camera)

        if args.psf is None:
            psf = desimodel.io.load_psf(channel)
            
        if args.cosmics is not None:
            shape = (psf.npix_y, psf.npix_x)
            cosmics = io.read_cosmics(args.cosmics, args.expid, shape=shape)
        else:
            cosmics = None
        
        image, rawpix, truepix = simulate(camera, simspec, psf,
            nspec=args.nspec, ncpu=args.ncpu, cosmics=cosmics,
            wavemin=args.wavemin, wavemax=args.wavemax)

        #- TODO: this uses the wrong header
        desispec.io.write_raw(args.rawfile, rawpix, image.meta)
        log.info('Wrote {} image to {}'.format(camera, args.rawfile))
        
        if args.pixfile is None:
            if args.preproc:
                pixfile = desispec.io.findfile('pix', args.night, args.expid,
                    camera = camera)
                pixfile = os.path.join(
                    os.path.dirname(args.rawfile), os.path.basename(pixfile))
                desispec.io.write_image(pixfile, image)
                log.info('Wrote '+pixfile)
        else:
            desispec.io.write_image(args.pixfile, image)
            log.info('Wrote ' + args.pixfile)

        if args.simpixfile is None:
            simpixfile = io.findfile('simpix', args.night, args.expid,
                camera=camera)
        else:
            simpixfile = args.simpixfile
            
        io.write_simpix(simpixfile, truepix, meta=image.meta)
        log.info('Wrote '+simpixfile)

    log.info('Finished pixsim {}'.format(asctime()))

def simulate_frame(night, expid, camera, **kwargs):
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
    io.write_simpix(simpixfile, truepix, meta=image.meta)

    simdir = io.simdir(night=night)
    rawfile = desispec.io.findfile('desi', night=night, expid=expid)
    rawfile = os.path.join(simdir, os.path.basename(rawfile))
    desispec.io.write_raw(rawfile, rawpix, image.meta, camera=camera)

    pixfile = desispec.io.findfile('pix', night=night, expid=expid, camera=camera)
    pixfile = os.path.join(simdir, os.path.basename(pixfile))
    desispec.io.write_image(pixfile, image)

def simulate(camera, simspec, psf, nspec=None, ncpu=None,
    trimxy=False, cosmics=None, wavemin=None, wavemax=None):

    log.info('Starting pixsim.simulate {}'.format(asctime()))
    #- parse camera name into channel and spectrograph number
    channel = camera[0].lower()
    ispec = int(camera[1])
    assert channel in 'brz'
    assert 0 <= ispec < 10

    #- Load DESI parameters
    params = desimodel.io.load_desiparams()
    
    #- this is not necessarily true, the truth in is the fibermap
    # nfibers = params['spectro']['nfibers']

    #---------------------------------------------------------------------
    #- This section had a merge conflict; I think it comes from supporting
    #- teststands with only a subset of fibers.  Commenting out while
    #- rebasing rawpixsim, and then we can come back to re-implementing
    #- what this is trying to accomplish.
    #
    # #- Get the list of spectra indices in the simspec.phot file that correspond of this camera
    # ii=np.where(fm["SPECTROID"]==ispec)[0]
    # 
    # #- Truncate if larger than requested nspec
    # if ii.size > nspec :
    #     ii=ii[:nspec]
    #     
    # #- Now we have to place our non empty fibers back to the reference fiber positions of the sims
    # #- that expect nfibers_sim fibers
    # nfibers_sim = params['spectro']['nfibers']
    # simphot = np.zeros((nfibers_sim,phot.shape[1]))
    # simphot[fm["FIBER"][ii]-nfibers_sim*ispec]=phot[ii]
    # #- overwrite phot
    # phot=simphot
    #---------------------------------------------------------------------

    if ispec*nfibers >= simspec.nspec:
        msg = "camera {} not covered by simspec with {} spectra".format(
            camera, simspec.nspec)
        log.error(msg)
        raise ValueError(msg)

    wave = simspec.wave[channel]
    phot = simspec.phot[channel] + simspec.skyphot[channel]

    #- Trim to just the spectra for this spectrograph
    if nspec is None:
        ii = slice(nfibers*ispec, nfibers*(ispec+1))
    else:
        ii = slice(nfibers*ispec, nfibers*ispec + nspec)

    phot = phot[ii]

    #- Trim wavelenths if needed
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
    header = dict()
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
        log.info('RDNOISE1 {}'.format(header['RDNOISE1']))
        log.info('RDNOISE2 {}'.format(header['RDNOISE2']))
        log.info('RDNOISE3 {}'.format(header['RDNOISE3']))
        log.info('RDNOISE4 {}'.format(header['RDNOISE4']))
    else:
        pix = truepix
        readnoise = params['ccd'][channel]['readnoise']
        header['RDNOISE1'] = readnoise
        header['RDNOISE2'] = readnoise
        header['RDNOISE3'] = readnoise
        header['RDNOISE4'] = readnoise

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

def photpix2raw(phot, gain, readnoise, offset=None, nprescan=7, noverscan=50,
    readorder='lr', noisydata=True):
    '''
    Add prescan, overscan, noise, and integerization

    returns image = int((poisson(phot) + offset + gauss(readnoise))/gain)

    readorder = 'lr' or 'rl'
    
    TODO: document     
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

#-------------------------------------------------------------------------
# def simulate(night, expid, camera, nspec=None, verbose=False, ncpu=None,
#     trimxy=False, cosmics=None, wavemin=None, wavemax=None):
#     """
#     Run pixel-level simulation of input spectra
#     
#     Args:
#         night (string) : YEARMMDD
#         expid (integer) : exposure id
#         camera (str) : e.g. b0, r1, z9
# 
#     Optional:
#         nspec (int) : number of spectra to simulate
#         verbose (boolean) : if True, print status messages
#         ncpu (int) : number of CPU cores to use in parallel
#         trimxy (boolean) : trim image to just pixels with input signal
#         cosmics (str) : filename with dark images with cosmics to add
#         wavemin, wavemax (float) : min/max wavelength range to simulate
# 
#     Reads:
#         $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits
#         
#     Writes:
#         $DESI_SPECTRO_SIM/$PIXPROD/{night}/simpix-{camera}-{expid}.fits
#         $DESI_SPECTRO_SIM/$PIXPROD/{night}/pix-{camera}-{expid}.fits
#     """
#     if verbose:
#         log.info("Reading input files")
# 
#     channel = camera[0].lower()
#     ispec = int(camera[1])
#     assert channel in 'brz'
#     assert 0 <= ispec < 10
# 
#     #- Load DESI parameters
#     params = desimodel.io.load_desiparams()
#     nfibers = params['spectro']['nfibers']
# 
#     #- Load simspec file
#     simfile = io.findfile('simspec', night=night, expid=expid)
#     simspec = io.read_simspec(simfile)
#     wave = simspec.wave[channel]
#     if simspec.skyphot is not None:
#         phot = simspec.phot[channel] + simspec.skyphot[channel]
#     else:
#         phot = simspec.phot[channel]
# 
#     if ispec*nfibers >= simspec.nspec:
#         log.fatal("ERROR: camera {} not in the {} spectra in {}/{}".format(
#             camera, simspec.nspec, night, os.path.basename(simfile)))
#         return
# 
#     #- Load PSF
#     psf = desimodel.io.load_psf(channel)
# 
#     #- Trim to just the spectra for this spectrograph
#     if nspec is None:
#         ii = slice(nfibers*ispec, nfibers*(ispec+1))
#     else:
#         ii = slice(nfibers*ispec, nfibers*ispec + nspec)
# 
#     phot = phot[ii]
# 
#     #- Trim wavelenths if needed
#     if wavemin is not None:
#         ii = (wave >= wavemin)
#         phot = phot[:, ii]
#         wave = wave[ii]
#     if wavemax is not None:
#         ii = (wave <= wavemax)
#         phot = phot[:, ii]
#         wave = wave[ii]
# 
#     #- check if simulation has less than 500 input spectra
#     if phot.shape[0] < nspec:
#         nspec = phot.shape[0]
# 
#     #- Project to image and append that to file
#     if verbose:
#         log.info("Projecting photons onto {} CCD".format(camera))
#         
#     img = parallel_project(psf, wave, phot, ncpu=ncpu)
#     
#     if trimxy:
#         xmin, xmax, ymin, ymax = psf.xyrange((0,nspec), wave)
#         img = img[0:ymax, 0:xmax]
#         # img = img[ymin:ymax, xmin:xmax]
#         # hdr['CRVAL1'] = xmin+1
#         # hdr['CRVAL2'] = ymin+1
# 
#     #- Prepare header
#     hdr = simspec.header
#     tmp = '/'.join(simfile.split('/')[-3:])  #- last 3 elements of path
#     hdr['SIMFILE'] = (tmp, 'Input simulation file')
# 
#     #- Strip unnecessary keywords
#     for key in ('EXTNAME', 'LOGLAM', 'AIRORVAC', 'CRVAL1', 'CDELT1'):
#         if key in hdr:
#             del hdr[key]
# 
#     #- Write noiseless output
#     simpixfile = io.findfile('simpix', night=night, expid=expid, camera=camera)
#     io.write_simpix(simpixfile, img, meta=hdr)
# 
#     #- Add cosmics from library of dark images
#     #- in this case, don't add readnoise since the dark image already has it
#     if cosmics is not None:
#         cosmics = io.read_cosmics(cosmics, expid, shape=img.shape)
#         pix = np.random.poisson(img) + cosmics.pix
#         readnoise = cosmics.meta['RDNOISE']
#     #- Or just add noise
#     else:
#         channel = camera[0].lower()
#         readnoise = params['ccd'][channel]['readnoise']
#         pix = np.random.poisson(img) + np.random.normal(scale=readnoise, size=img.shape)
# 
#     ivar = 1.0/(pix.clip(0) + readnoise**2)
#     mask = np.zeros(img.shape, dtype=np.uint16)
# 
#     #- Augment the input header
#     meta = simspec.header.copy()
#     meta['SPECTRO'] = ispec
# 
#     image = Image(pix, ivar, mask, readnoise=readnoise, camera=camera, meta=meta)
# 
#     #- In-place update of the image cosmic ray mask
#     if cosmics is not None:
#         desispec.cosmics.reject_cosmic_rays(image)
# 
#     pixfile = desispec.io.findfile('pix', night=night, camera=camera, expid=expid)
#     pixfile = os.path.join(simdir, os.path.basename(pixfile))
#     desispec.io.write_image(pixfile, image)
# 
#     if verbose:
#         log.info("Wrote "+pixfile)
        
#-------------------------------------------------------------------------

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

    if ncpu < 0:
        #- Serial version
        ### print "Serial project"
        img = psf.project(wave, phot)
    else:
        #- multiprocessing version
        #- Split the spectra into ncpu groups
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
    
