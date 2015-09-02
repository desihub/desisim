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

from desisim import obs, io

def simulate(night, expid, camera, nspec=None, verbose=False, ncpu=None, trimxy=None):
    """
    Run pixel-level simulation of input spectra
    
    Args:
        night : YEARMMDD string
        expid : integer exposure id
        camera : e.g. b0, r1, z9
        nspec (optional) : number of spectra to simulate
        verbose (optional) : if True, print status messages
        ncpu (optional) : number of CPU cores to use

    Reads:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits
        
    Writes:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simpix-{camera}-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/pix-{camera}-{expid}.fits
    """
    simdir = io.simdir(night)
    simfile = '{}/simspec-{:08d}.fits'.format(simdir, expid)

    if verbose:
        print "Reading input files"

    channel = camera[0].upper()
    ispec = int(camera[1])
    assert channel in 'BRZ'
    assert 0 <= ispec < 10

    #- Load DESI parameters
    params = desimodel.io.load_desiparams()
    nfibers = params['spectro']['nfibers']

    #- Check that this camera has simulated spectra
    hdr = fits.getheader(simfile, 'PHOT_'+channel)
    nspec_in = hdr['NAXIS2']
    if ispec*nfibers >= nspec_in:
        print "ERROR: camera {} not in the {} spectra in {}/{}".format(
            camera, nspec_in, night, os.path.basename(simfile))
        return

    #- Load input photon data
    phot = fits.getdata(simfile, 'PHOT_'+channel)
    try:
        phot += fits.getdata(simfile, 'SKYPHOT_'+channel)
    except KeyError:
        pass  #- arcs and flats don't have SKYPHOT
    
    nwave = phot.shape[1]
    wave = hdr['CRVAL1'] + np.arange(nwave)*hdr['CDELT1']
    
    #- Load PSF
    psf = desimodel.io.load_psf(channel)

    #- Trim to just the spectra for this spectrograph
    if nspec is None:
        ii = slice(nfibers*ispec, nfibers*(ispec+1))
        phot = phot[ii]
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
    hdr = fits.getheader(simfile, 0)
    tmp = '/'.join(simfile.split('/')[-3:])  #- last 3 elements of path
    hdr['SIMFILE'] = (tmp, 'Input simulation file')

    #- Strip unnecessary keywords
    for key in ('EXTNAME', 'LOGLAM', 'AIRORVAC', 'CRVAL1', 'CDELT1'):
        if key in hdr:
            del hdr[key]

    #- Add noise and write output files
    pixfile = io.write_simpix(img, camera, night, expid, header=hdr)

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
    
