"""
Tools for DESI pixel level simulations using specter
"""

import sys
import os
import os.path
import time
### import multiprocessing as mp

import numpy as np
import yaml
from astropy.io import fits

import desimodel.io
import desispec.io
from desispec.image import Image

from desisim import obs, io

def simulate(night, expid, camera, nspec=None, verbose=False, ncpu=None,
    trimxy=False, cosmics=None):
    """
    Run pixel-level simulation of input spectra
    
    Args:
        night (string) : YEARMMDD
        expid (integer) : exposure id
        camera (str) : e.g. b0, r1, z9
        nspec (int) : number of spectra to simulate
        verbose (boolean) : if True, print status messages
        ncpu (int) : number of CPU cores to use in parallel
        trimxy (boolean) : trim image to just pixels with input signal
        cosmics (str) : filename with dark images with cosmics to add

    Reads:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits
        
    Writes:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simpix-{camera}-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/pix-{camera}-{expid}.fits
    """
    if verbose:
        print "Reading input files"

    channel = camera[0].lower()
    ispec = int(camera[1])
    assert channel in 'brz'
    assert 0 <= ispec < 10

    #- Load DESI parameters
    params = desimodel.io.load_desiparams()
    nfibers = params['spectro']['nfibers']

    #- Load simspec file
    simfile = io.findfile('simspec', night=night, expid=expid)
    simspec = io.read_simspec(simfile)
    wave = simspec.wave[channel]
    if simspec.skyphot is not None:
        phot = simspec.phot[channel] + simspec.skyphot[channel]
    else:
        phot = simspec.phot[channel]

    if ispec*nfibers >= simspec.nspec:
        print "ERROR: camera {} not in the {} spectra in {}/{}".format(
            camera, simspec.nspec, night, os.path.basename(simfile))
        return

    #- Load PSF
    psf = desimodel.io.load_psf(channel)

    #- Trim to just the spectra for this spectrograph
    if nspec is None:
        ii = slice(nfibers*ispec, nfibers*(ispec+1))
    else:
        ii = slice(nfibers*ispec, nfibers*ispec + nspec)

    phot = phot[ii]

    #- check if simulation has less than 500 input spectra
    if phot.shape[0] < nspec:
        nspec = phot.shape[0]

    #- Project to image and append that to file
    if verbose:
        print "Projecting photons onto CCD"
        
    img = parallel_project(psf, wave, phot, ncpu=ncpu)
    
    if trimxy:
        xmin, xmax, ymin, ymax = psf.xyrange((0,nspec), wave)
        img = img[0:ymax, 0:xmax]
        # img = img[ymin:ymax, xmin:xmax]
        # hdr['CRVAL1'] = xmin+1
        # hdr['CRVAL2'] = ymin+1

    #- Prepare header
    hdr = simspec.header
    tmp = '/'.join(simfile.split('/')[-3:])  #- last 3 elements of path
    hdr['SIMFILE'] = (tmp, 'Input simulation file')

    #- Strip unnecessary keywords
    for key in ('EXTNAME', 'LOGLAM', 'AIRORVAC', 'CRVAL1', 'CDELT1'):
        if key in hdr:
            del hdr[key]

    #- Write noiseless output
    simpixfile = io.findfile('simpix', night=night, expid=expid, camera=camera)
    io.write_simpix(simpixfile, img, meta=hdr)

    #- Add cosmics from library of dark images
    #- in this case, don't add readnoise since the dark image already has it
    if cosmics is not None:
        cosmics = io.read_cosmics(cosmics, expid, shape=img.shape)
        pix = np.random.poisson(img) + cosmics.pix
        readnoise = cosmics.meta['RDNOISE']
        ivar = 1.0/(pix.clip(0) + readnoise**2)
        mask = cosmics.mask
    #- Or just add noise
    else:
        params = desimodel.io.load_desiparams()
        channel = camera[0].lower()
        readnoise = params['ccd'][channel]['readnoise']
        pix = np.random.poisson(img) + np.random.normal(scale=readnoise, size=img.shape)
        ivar = 1.0/(pix.clip(0) + readnoise**2)
        mask = np.zeros(img.shape, dtype=np.uint16)
    
    #- Metadata to be included in pix file header is in the fibermap header
    #- TODO: this is fragile; consider updating fibermap to use astropy Table
    #- that includes the header rather than directly assuming FITS as the
    #- underlying format.
    fibermapfile = desispec.io.findfile('fibermap', night=night, expid=expid)
    fm, fmhdr = desispec.io.read_fibermap(fibermapfile, header=True)
    meta = dict()
    try:
        meta['TELRA']  = simspec.header['TELRA']
        meta['TELDEC'] = simspec.header['TELDEC']
    except KeyError:  #- temporary backwards compatibilty
        meta['TELRA']  = fmhdr['TELERA']
        meta['TELDEC'] = fmhdr['TELEDEC']
        
    meta['TILEID'] = simspec.header['TILEID']
    meta['DATE-OBS'] = simspec.header['DATE-OBS']
    meta['FLAVOR'] = simspec.header['FLAVOR']
    meta['EXPTIME'] = simspec.header['EXPTIME']
    meta['AIRMASS'] = simspec.header['AIRMASS']

    image = Image(pix, ivar, mask, readnoise=readnoise, camera=camera, meta=meta)
    pixfile = desispec.io.findfile('pix', night=night, camera=camera, expid=expid)
    desispec.io.write_image(pixfile, image)

    if verbose:
        print "Wrote "+pixfile
        
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
            print '-'*60
            print 'ERROR in _project', psf.wmin, psf.wmax, wave[0], wave[-1], phot.shape, specmin
            traceback.print_exc()
            print '-'*60
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
        ncpu = mp.cpu_count() / 2

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
    
