import os
import numpy as np

import yaml

from astropy.io import fits
from astropy.table import Table

import specter
from specter.psf import load_psf
from specter.throughput import load_throughput
import specter.util

from desisim.obs import get_next_obs

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
        

def simulate(simfile, nspec=None):
    """
    Simulate spectra
    """
    #- Load input data
    _, camera, expid = _parse_filename(simfile)
    data = fits.getdata(simfile, camera.upper()).view(np.recarray)

    if nspec is None:
        nspec = data.FLUX.shape[0]
        nspec = 5 ### TEST: Just simulate a few spectra for debugging ###

    #- Load PSF
    psfdir = os.environ['DESIMODEL'] + '/data/specpsf'
    psf = load_psf(psfdir+'/psf-'+camera[0]+'.fits')

    img = psf.project(data.WAVE, (data.PHOT.T+data.SKYPHOT)[0:nspec])
    
    outdir = os.path.split(simfile)[0]
    outfile = '{}/simpix-{}-{:08d}.fits'.format(outdir, camera, expid)
    fits.writeto(outfile, img, clobber=True)

def new_exposure(verbose=False):
    """
    Setup new exposure to simulate
    
    Writes
        $DESI_SPECTRO_SIM/{night}/fibermap-{expid}.fits
        $DESI_SPECTRO_SIM/{night}/simflux-{camera}-{expid}.fits
        
    Returns tuple (night, expid)
    """
    
    #- Load PSFs and throughputs
    if verbose:
        print "Loading PSFs and throughputs"
        
    psfdir = os.environ['DESIMODEL'] + '/data/specpsf'
    thrudir = os.environ['DESIMODEL'] + '/data/throughput'
    psf = dict()
    thru = dict()
    for channel in ('b', 'r', 'z'):
        psf[channel] = load_psf(psfdir+'/psf-'+channel+'.fits')
        thru[channel] = load_throughput(thrudir+'/thru-'+channel+'.fits')

    #- Other DESI parameters
    params = yaml.load(open(os.environ['DESIMODEL']+'/data/desi.yaml'))

    #- What to do
    night, expid, tileid, fibermap = get_next_obs()

    #-----
    #- TEST: Trim for debugging
    fibermap = fibermap[0:1000]
    #-----

    #- Create output directory if needed
    outdir = '{}/{}/'.format(os.getenv('DESI_SPECTRO_SIM'), night)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #- Write fibermap
    fibermap_file = '{}/fibermap-{:08d}.fits'.format(outdir, expid)

    hdr = fits.Header()
    hdr.append( ('TILEID', tileid, 'Tile ID') )
    hdr.append( ('EXPID',  expid, 'Exposure number') )
    #- TODO: code versions...
    
    if verbose:
        print "Writing " + os.path.basename(fibermap_file)
        
    fits.writeto(fibermap_file, fibermap, header=hdr)

    #- Get object spectra
    if verbose:
        print "Getting template spectra"
        
    dw = 0.1
    wavebrz = np.arange(round(psf['b'].wmin, 1), psf['z'].wmax, dw)
    ### spectra = get_templates(wavebrz, fibermap['_SIMTYPE'], fibermap['_SIMZ'])
    nspec = len(fibermap)
    spectra = np.zeros( (nspec, len(wavebrz)) )

    #- Load sky [Magic knowledge of units 1e-17 erg/s/cm2/A/arcsec2]
    skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky.dat'
    skywave, skyflux_brz = np.loadtxt(skyfile, unpack=True)

    for channel in ('b', 'r', 'z'):
        wmin = psf[channel].wmin
        wmax = psf[channel].wmax
        ii = np.where( (wmin <= wavebrz) & (wavebrz <= wmax) )[0]
        wave = wavebrz[ii]
    
        flux = spectra[:, ii]
    
        phot = thru[channel].photons(wave, flux, units='1e-17 erg/s/cm2/A',
                objtype=fibermap['_SIMTYPE'], exptime=params['exptime'])
        
        skyflux = np.interp(wave, skywave, skyflux_brz)
        skyphot = thru[channel].photons(wave, skyflux, units='1e-17 erg/s/cm2/A/arcsec^2',
            objtype='SKY', exptime=params['exptime'])
        
        nspec_per_cam = 500
        for ispec in range(0, nspec, nspec_per_cam):
            camera = channel + str(ispec//nspec_per_cam)
            ii = slice(ispec, ispec+nspec_per_cam)
            fluxtable = Table([wave, flux[ii].T, phot[ii].T, skyflux, skyphot],
                            names=['WAVE', 'FLUX', 'PHOT', 'SKYFLUX', 'SKYPHOT'])

            simflux_file = '{}/simflux-{}-{:08d}.fits'.format(outdir, camera, expid)
            if verbose:
                print "Writing " + os.path.basename(simflux_file)
    
            #- fits bug requires creation of HDU to get all header keywords first
            hdu = fits.BinTableHDU(fluxtable._data, name=camera.upper())
            hdu.header.append( ('CAMERA', camera, 'Spectograph Camera') )
            fits.writeto(simflux_file, hdu.data, hdu.header)

    if verbose:
        print "Wrote output to "+outdir
        
    return night, expid

