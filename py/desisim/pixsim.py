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
        

def simulate(night, expid, camera, nspec=None, verbose=False):
    """
    Simulate spectra

    DOCUMENT
    """
    simdir = io.simdir(night)
    simfile = '{}/simspec-{:08d}.fits'.format(simdir, expid)

    if verbose:
        print "Reading input files"

    channel = camera[0].upper()
    ispec = int(camera[1])
    assert channel in 'BRZ'
    assert 0 <= ispec < 10
    
    hdr = io.fits.getheader(simfile, 'PHOT_'+channel)
    phot  = io.fits.getdata(simfile, 'PHOT_'+channel)
    phot += io.fits.getdata(simfile, 'SKYPHOT_'+channel)
    
    nwave = phot.shape[1]
    wave = hdr['CRVAL1'] + np.arange(nwave)*hdr['CDELT1']
    
    #- Load PSF and DESI parameters
    psf = io.load_psf(channel)
    params = io.load_desiparams()
    nfibers = params['spectro']['nfibers']

    #- Trim to just the spectra for this spectrograph
    if nspec is None:
        ii = slice(nfibers*ispec, nfibers*(ispec+1))
        phot = phot[ii]
    else:
        ii = slice(nfibers*ispec, nfibers*ispec + nspec)
        phot = phot[ii]

    #- Project to image and append that to file
    if verbose:
        print "Projecting photons onto CCD"
        
    #- Project photons onto the CCD
    img = psf.project(wave, phot)

    simpixfile = '{}/simpix-{}-{:08d}.fits'.format(simdir, camera, expid)
    
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

    rdnoise = params['ccd'][channel.lower()]['readnoise']
    var = img + rdnoise**2
    img += np.random.poisson(img)
    img += np.random.normal(scale=rdnoise, size=img.shape)
    
    #- Write the final noisy image file
    #- Pixels
    outfile = '{}/proc-{}-{:08d}.fits'.format(simdir, camera, expid)
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
        
