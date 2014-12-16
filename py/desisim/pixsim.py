import os
import time

import numpy as np

import yaml

from astropy.io import fits
from astropy.table import Table

import specter
from specter.psf import load_psf
from specter.throughput import load_throughput
import specter.util

from desisim import obs, io

def _parse_filename(filename):
    """
    Parse filename and return (prefix, expid) or (prefix, camera, expid)
    """
    base = os.path.basename(os.path.splitext(filename)[0])
    x = base.split('-')
    if len(x) == 2:
        return x[0], None, int(x[1])
    elif len(x) == 3:
        return x[0], x[1].lower(), int(x[2])
        

def simulate(simfile, nspec=None, verbose=False):
    """
    Simulate spectra
    
    Writes
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simflux-{camera}-{expid}.fits
    
    TODO: more flexible input interface
    """
    assert os.path.exists(simfile)

    if verbose:
        print "Reading input files"
    
    hdr = fits.getheader(simfile, 1)
    spectra = fits.getdata(simfile, 1)
    nspec, nwave = spectra['FLUX'].shape
    
    wave = hdr['CRVAL1'] + np.arange(nwave)*hdr['CDELT1']
    
    #- camera b0 -> channel b, ispec 0
    camera = hdr['camera']
    channel = camera[0]
    ispec = int(camera[1])
    assert channel.lower() in 'brz'
    assert 0 <= ispec < 10
    
    #- Load PSF and DESI parameters
    psf = io.load_psf(channel)
    params = io.load_desiparams()
    nspec = params['spectro']['nfibers']

    #- Project to image and append that to file
    if verbose:
        print "Projecting photons onto CCD"
        
    phot = spectra['PHOT']+spectra['SKYPHOT']
    if nspec is not None:
        phot = phot[0:nspec]
        
    #- Project photons onto the CCD
    img = psf.project(wave, phot)

    outdir = io.simdir(hdr['NIGHT'])
    expid = hdr['EXPID']
    simpixfile = '{}/simpix-{}-{:08d}.fits'.format(outdir, camera, expid)
    
    hdu = fits.PrimaryHDU(img, header=hdr)
    tmp = '/'.join(simfile.split('/')[-3:])  #- last 3 elements of path
    hdu.header['SIMFILE'] = (tmp, 'Input simulation file')
    hdu.header['VSPECTER'] = ('0.0.0', 'TODO: Specter version')
    
    fits.writeto(simpixfile, hdu.data, header=hdu.header, clobber=True)

    if verbose:
        print "Wrote "+simpixfile
        
    #- Add noise
    if verbose:
        print "Adding noise"

    rdnoise = params['ccd'][channel]['readnoise']
    var = img + rdnoise**2
    img += np.random.poisson(img)
    img += np.random.normal(scale=rdnoise, size=img.shape)
    
    #- Write the final noisy image file
    #- Pixels
    outfile = '{}/proc-{}-{:08d}.fits'.format(outdir, camera, expid)
    hdu = fits.ImageHDU(img, header=hdr, name=camera.upper())
    hdu.header.append( ('CAMERA', camera, 'Spectograph Camera') )
    hdu.header.append( ('VSPECTER', '0.0.0', 'TODO: Specter version') )
    hdu.header.append( ('EXPTIME', params['exptime'], 'Exposure time [sec]') )
    fits.writeto(outfile, hdu.data, hdu.header, clobber=True)

    #- Inverse variance (IVAR)
    hdu = fits.ImageHDU(1.0/var, name=camera.upper()+'IVAR')
    fits.append(outfile, hdu.data, hdu.header, clobber=True)

    #- Mask (currently just zeros)
    mask = np.zeros(img.shape, dtype=np.int32)
    hdu = fits.ImageHDU(mask, name=camera.upper()+'MASK')
    fits.append(outfile, hdu.data, hdu.header, clobber=True)

    if verbose:
        print "Wrote "+outfile
        
