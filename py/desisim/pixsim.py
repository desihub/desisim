import os
import os.path
import time
### import multiprocessing as mp

import numpy as np
import yaml
from astropy.io import fits

from desisim import obs, io
from desisim.interpolation import resample_flux

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
        
    img = parallel_project(psf, wave, phot, ncpu=ncpu)

    #- Add noise and write output files
    tmp = '/'.join(simfile.split('/')[-3:])  #- last 3 elements of path
    hdr['SIMFILE'] = (tmp, 'Input simulation file')
    pixfile = _write_simpix(img, camera, 'science', night, expid, header=hdr)

    if verbose:
        print "Wrote "+pixfile

def new_arcexp(nspec=None, nspectrographs=10, ncpu=None):
    """
    Run pixel simulation of new arc lamp exposure
    
    Args:
        nspec : number of spectra to simulate
        nspectrographs : number of spectrographs to simulate [1-10]
        ncpu : number of CPU cores to use
    
    Writes to $DESI_SPECTRO_SIM/$PIXPROD/
        simpix-{camera}-{expid}.fits - noiseless simulated image
        pix-{camera}-{expid}.fits - noisy image, ivar, mask
        
    Updates $DESI_SPECTRO_SIM/$PIXPROD/etc/obslog.sqlite and next_expid.txt

    Bugs:
        Uses hardcoded arc lamp spectrum in
        $DESI_ROOT/spectro/templates/calib/v0.1/arc-lines-average.fits
    """
    
    expid = obs.get_next_expid()
    dateobs = time.gmtime()
    night = obs.get_night(utc=dateobs)
    simdir = io.simdir(night, mkdir=True)

    params = io.load_desiparams()
    if nspec is None:
        nspec = params['spectro']['nfibers']
    
    #- Load input arc template in non-standard format [HARDCODE!]
    infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.1/arc-lines-average.fits'
    d = fits.getdata(infile, 1)
    wave = d['AIRWAVE']
    phot = np.tile(d['ELECTRONS'], nspec).reshape(nspec, len(wave))

    for channel in ('b', 'r', 'z'):
        psf = io.load_psf(channel)

        #- Noiseless image
        ### img = psf.project(w, phot)
        img = parallel_project(psf, wave, phot, ncpu=ncpu)

        for i in range(nspectrographs):
            camera = channel+str(i)
            pixfile = _write_simpix(img, camera, 'arc', night, expid)
            print pixfile

    #- Update obslog that we succeeded with this exposure
    obs.update_obslog('arc', expid, dateobs)

def new_flatexp(nspec=None, nspectrographs=10, ncpu=None):
    """
    Run pixel simulation of new flat lamp exposure
    
    Args:
        nspec : number of spectra to simulate
        nspectrographs : number of spectrographs to simulate [1-10]
        ncpu : number of CPU cores to use
    
    Writes to $DESI_SPECTRO_SIM/$PIXPROD/
        simpix-{camera}-{expid}.fits - noiseless simulated image
        pix-{camera}-{expid}.fits - noisy image, ivar, mask
        
    Updates $DESI_SPECTRO_SIM/$PIXPROD/etc/obslog.sqlite and next_expid.txt
    
    Bugs:
        Uses hardcoded flat lamp spectrum in
        $DESI_ROOT/spectro/templates/calib/v0.1/flat-3100K-quartz-iodine.fits
    """
    expid = obs.get_next_expid()
    dateobs = time.gmtime()
    night = obs.get_night(utc=dateobs)
    simdir = io.simdir(night, mkdir=True)

    params = io.load_desiparams()
    if nspec is None:
        nspec = params['spectro']['nfibers']

    #- Load input arc template in non-standard format [HARDCODE!]
    infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.1/flat-3100K-quartz-iodine.fits'
    flux = fits.getdata(infile, 0)
    hdr = fits.getheader(infile, 0)
    wave = io.load_wavelength(infile, 0)
    
    #- resample to 0.2 A grid
    dw = 0.2
    ww = np.arange(wave[0], wave[-1]+dw/2, dw)
    flux = resample_flux(ww, wave, flux)
    wave = ww
    
    #- Convert to 2D for projection
    flux = np.tile(flux, nspec).reshape(nspec, len(wave))
    
    for channel in ('b', 'r', 'z'):
        psf = io.load_psf(channel)
        thru = io.load_throughput(channel)
        phot = thru.photons(wave, flux, units=hdr['BUNIT'], objtype='CALIB')
        ### img = psf.project(wave, phot)
        ii = (psf.wmin <= wave) & (wave <= psf.wmax)
        img = parallel_project(psf, wave[ii], phot[:,ii], ncpu=ncpu)
        
        for i in range(nspectrographs):
            camera = channel+str(i)
            pixfile = _write_simpix(img, camera, 'arc', night, expid)
            print pixfile

    #- Update obslog that we succeeded with this exposure
    obs.update_obslog('flat', expid, dateobs)
        

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
    psf, wave, phot, specmin = args
    nspec = phot.shape[0]
    xyrange = psf.xyrange( [specmin, specmin+nspec], wave )
    img = psf.project(wave, phot, specmin=specmin, xyrange=xyrange)
    return (xyrange, img)

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
            if iispec[i+1] > iispec[i]:  #- can fail if nspec < ncpu
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
    
def _write_simpix(img, camera, flavor, night, expid, header=None):
    """
    Add noise to input image and write output files.
    
    Args:
        img : 2D noiseless image array
        camera : e.g. b0, r1, z9
        flavor : arc or flat
        night  : YEARMMDD string
        expid  : integer exposure id
        
    Writes to $DESI_SPECTRO_SIM/$PIXPROD/{night}/
        simpix-{camera}-{expid}.fits
        pix-{camera}-{expid}.fits
    """

    simdir = io.simdir(night, mkdir=True)
    params = io.load_desiparams()
    channel = camera[0].lower()

    #- Add noise, generate inverse variance and mask
    rdnoise = params['ccd'][channel]['readnoise']
    pix = np.random.poisson(img) + np.random.normal(scale=rdnoise, size=img.shape)
    ivar = 1.0/(pix.clip(0) + rdnoise**2)
    mask = np.zeros(img.shape, dtype=np.int32)

    #-----
    #- Write noiseless image to simpix file
    simpixfile = '{}/simpix-{}-{:08d}.fits'.format(simdir, camera, expid)

    hdu = fits.PrimaryHDU(img, header=header)
    hdu.header['VSPECTER'] = ('0.0.0', 'TODO: Specter version')
    fits.writeto(simpixfile, hdu.data, header=hdu.header, clobber=True)

    #- Add x y trace locations from PSF
    psffile = '{}/data/specpsf/psf-{}.fits'.format(os.getenv('DESIMODEL'), channel)
    psfxy = fits.open(psffile)
    fits.append(simpixfile, psfxy['XCOEFF'].data, header=psfxy['XCOEFF'].header)
    fits.append(simpixfile, psfxy['YCOEFF'].data, header=psfxy['YCOEFF'].header)

    #-----
    #- Write simulated raw data to pix file

    #- Primary HDU: noisy image
    outfile = '{}/pix-{}-{:08d}.fits'.format(simdir, camera, expid)
    hdu = fits.PrimaryHDU(pix, header=header)
    hdu.header.append( ('CAMERA', camera, 'Spectograph Camera') )
    hdu.header.append( ('VSPECTER', '0.0.0', 'TODO: Specter version') )
    hdu.header.append( ('EXPTIME', params['exptime'], 'Exposure time [sec]') )
    hdu.header.append( ('RDNOISE', rdnoise, 'Read noise [electrons]'))
    hdu.header.append( ('FLAVOR', flavor, 'Exposure type (arc, flat, science)'))
    fits.writeto(outfile, hdu.data, hdu.header, clobber=True)

    #- IVAR: Inverse variance (IVAR)
    hdu = fits.ImageHDU(ivar, name='IVAR')
    hdu.header.append(('RDNOISE', rdnoise, 'Read noise [electrons]'))
    fits.append(outfile, hdu.data, hdu.header, clobber=True)

    #- MASK: currently just zeros
    hdu = fits.ImageHDU(mask, name='MASK')
    fits.append(outfile, hdu.data, hdu.header, clobber=True)

    return outfile
    
