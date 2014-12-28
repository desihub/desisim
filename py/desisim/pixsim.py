import os
import os.path
import time
### import multiprocessing as mp

import numpy as np
import yaml
from astropy.io import fits

from desisim import obs, io

def _project(args):
    psf, wave, phot, specmin = args
    print 'starting', time.asctime(), specmin, phot.shape
    nspec = phot.shape[0]
    xyrange = psf.xyrange( [specmin, specmin+nspec], wave )
    img = psf.project(wave, phot, specmin=specmin, xyrange=xyrange)
    print 'ending', time.asctime()
    return (xyrange, img)
    
def simulate(night, expid, camera, nspec=None, verbose=False, ncpu=None):
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
    params = io.load_desiparams()
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
    phot += fits.getdata(simfile, 'SKYPHOT_'+channel)
    
    nwave = phot.shape[1]
    wave = hdr['CRVAL1'] + np.arange(nwave)*hdr['CDELT1']
    
    #- Load PSF
    psf = io.load_psf(channel)

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
    if ncpu < 0:
        #- Serial version
        img = psf.project(wave, phot)
    else:
        #- multiprocessing version
        import multiprocessing as mp
        if ncpu is None:
            ncpu = mp.cpu_count() / 2
            
        iispec = np.linspace(0, nspec, ncpu+1).astype(int)
        args = list()
        for i in range(ncpu):
            if iispec[i+1] > iispec[i]:
                print time.asctime(), i, iispec[i], iispec[i+1]
                args.append( [psf, wave, phot[iispec[i]:iispec[i+1]], iispec[i]] )
            
        pool = mp.Pool(ncpu)
        xy_subimg = pool.map(_project, args)
        img = np.zeros( (psf.npix_y, psf.npix_x) )
        for xyrange, subimg in xy_subimg:
            print time.asctime(), xyrange
            xmin, xmax, ymin, ymax = xyrange
            img[ymin:ymax, xmin:xmax] += subimg
    
    #-------------------------------------------------------------------------

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
    outfile = '{}/pix-{}-{:08d}.fits'.format(simdir, camera, expid)
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
        
