import os
import numpy as np

import yaml

from astropy.io import fits
from astropy.table import Table

import specter
from specter.psf import load_psf
from specter.throughput import load_throughput
import specter.util

from desisim import obs

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
        

def simulate(fibermap_file, camera, verbose=False):
    """
    Simulate spectra
    
    Writes
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simflux-{camera}-{expid}.fits
    
    TODO: more flexible input interface
    """
    assert os.path.exists(fibermap_file)
    assert len(camera) == 2
    
    #- camera b0 -> channel b, ispec 0
    channel = camera[0]
    ispec = int(camera[1])
    assert channel.lower() in 'brz'
    assert 0 <= ispec < 10
    
    if verbose:
        print "Reading input files"
    
    #- Load PSF and throughput
    psfdir = os.environ['DESIMODEL'] + '/data/specpsf'
    thrudir = os.environ['DESIMODEL'] + '/data/throughput'
    psf = load_psf(psfdir+'/psf-'+channel+'.fits')
    thru = load_throughput(thrudir+'/thru-'+channel+'.fits')

    #- Other DESI parameters
    params = yaml.load(open(os.environ['DESIMODEL']+'/data/desi.yaml'))
    nspec = params['spectro']['nfibers']

    outdir = os.path.split(fibermap_file)[0]
    fibermap = fits.getdata(fibermap_file, 'FIBERMAP')
    
    #- Trim to just the fibers for this spectrograph
    fibermap = fibermap[nspec*ispec:nspec*(ispec+1)]

    #- Get expid from FIBERMAP
    fmhdr = fits.getheader(fibermap_file, 'FIBERMAP')
    expid = fmhdr['EXPID']

    #-----
    #- TEST: trim for speed while testing
    fibermap = fibermap[0:3]
    #-----

    if verbose:
        print 'Applying throughput'
        
    dw = 0.1
    wave = np.arange(round(psf.wmin, 1), psf.wmax, dw)
    ### flux = get_templates(wave, fibermap['_SIMTYPE'], fibermap['_SIMZ'])
    nspec = len(fibermap)
    flux = np.zeros( (nspec, len(wave)) )

    #- Load sky [Magic knowledge of units 1e-17 erg/s/cm2/A/arcsec2]
    skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky.dat'
    skywave, skyflux = np.loadtxt(skyfile, unpack=True)
    skyflux = np.interp(wave, skywave, skyflux)

    phot = thru.photons(wave, flux, units='1e-17 erg/s/cm2/A',
            objtype=fibermap['_SIMTYPE'], exptime=params['exptime'])
    
    skyphot = thru.photons(wave, skyflux, units='1e-17 erg/s/cm2/A/arcsec2',
        objtype='SKY', exptime=params['exptime'])
    
    tmp = Table([wave, flux.T, phot.T, skyflux, skyphot],
                names=['WAVE', 'FLUX', 'PHOT', 'SKYFLUX', 'SKYPHOT'])
    ### spectra = tmp._data.view(np.recarray)
    spectra = tmp._data

    #- TODO: column units

    #- Write spectra table
    #- fits bug requires creation of HDU to get all header keywords first
    simfile = '{}/sim-{}-{:08d}.fits'.format(outdir, camera, expid)
    hdu = fits.BinTableHDU(spectra, header=fmhdr, name=camera.upper()+'-SPECTRA')
    hdu.header.append( ('CAMERA', camera, 'Spectograph Camera') )
    hdu.header.append( ('VSPECTER', '0.0.0', 'TODO: Specter version') )
    hdu.header.append( ('EXPTIME', params['exptime'], 'Exposure time [sec]') )
    fits.writeto(simfile, hdu.data, hdu.header, clobber=True)

    comments = dict(
        WAVE = 'Wavelength [Angstroms]',
        FLUX = 'Object flux [1e-17 erg/s/cm^2/A]',
        PHOT = 'Object photons per bin (not per A)',
        SKYFLUX = 'Sky flux [1e-17 erg/s/cm^2/A/arcsec^2]',
        SKYPHOT = 'Sky photons per bin (not per A)',
    )
    _add_table_comments(simfile, 1, comments)

    #- Project to image and append that to file
    if verbose:
        print "Projecting photons onto CCD"
    img = psf.project(spectra['WAVE'], (spectra['PHOT'].T+spectra['SKYPHOT']))
    hdu = fits.ImageHDU(img, header=fmhdr, name=camera.upper()+'-PIX')
    hdu.header.append( ('CAMERA', camera, 'Spectograph Camera') )
    hdu.header.append( ('VSPECTER', '0.0.0', 'TODO: Specter version') )
    hdu.header.append( ('EXPTIME', params['exptime'], 'Exposure time [sec]') )
    
    fits.append(simfile, hdu.data, header=hdu.header)

    #- NOTE: this does not enforce that the info in simfile is actually
    #- complete enough for generating the final noisy outfile.
    
    if verbose:
        print "Wrote "+simfile
        
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
    hdu = fits.ImageHDU(img, header=fmhdr, name=camera.upper())
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
    
def new_fibermap_file(night=None, expid=None, tileid=None, verbose=False):
    """
    Setup new exposure to simulate
    
    Writes
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/fibermap-{expid}.fits
        
    Returns full path to filename of fibermap file written
    """
    #- Get night, expid, and tileid if needed
    if night is None:
        night = obs.get_night()
        
    if expid is None:
        expid = obs.get_next_expid()
        
    if tileid is None:
        tileid = obs.get_next_tileid()
    
    #- Get fibermap table
    fibermap, tele_ra, tele_dec = obs.get_fibermap(tileid)
    ### fibermap = obs.get_fibermap(tileid=tileid, nfiber=1000)   #- TEST

    #- Create output directory if needed
    outdir = '{}/{}/{}/'.format(
        os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'), night )
        
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    #- Write fibermap
    fibermap_file = '{}/fibermap-{:08d}.fits'.format(outdir, expid)

    #- Create HDU to get required header keywords
    hdu = fits.BinTableHDU(fibermap, name='FIBERMAP')
    hdu.header.append( ('TILEID', tileid, 'Tile ID') )
    hdu.header.append( ('EXPID',  expid, 'Exposure number') )
    hdu.header.append( ('NIGHT',  str(night), 'Night YEARMMDD') )
    hdu.header.append( ('DATE-OBS',  '2000-01-01T00:00:00', 'TODO: date obs in UTC (or TAI?)') )
    hdu.header.append( ('VPIXSIM',  '0.0.0', 'TODO: pixsim version') )
    hdu.header.append( ('VDMODEL',  '0.0.0', 'TODO: desimodel version') )
    hdu.header.append( ('VOPTICS',  '0.0.0', 'TODO: optics model version') )
    hdu.header.append( ('VFIBVCAM', '0.0.0', 'TODO: fiber view code version') )
    hdu.header.append( ('TELERA',   tele_ra, 'Telescope central RA [deg]') )
    hdu.header.append( ('TELEDEC', tele_dec, 'Telescope central dec [deg]') )
    hdu.header.append( ('HEXPDROT', 0.0, 'TODO: hexapod rotation [deg]') )
    #- TODO: code versions...
    
    if verbose:
        print "Writing " + os.path.basename(fibermap_file)
        
    fits.writeto(fibermap_file, hdu.data, header=hdu.header)
    
    #- Update columns with comment fields
    comments = dict(
        FIBER        = "Fiber ID [0-4999]",
        POSITIONER   = "Positioner ID [0-4999]",
        SPECTROID    = "Spectrograph ID [0-9]",
        TARGETID     = "Unique target ID",
        TARGETCAT    = "Name/version of the target catalog",
        OBJTYPE      = "Target type [ELG, LRG, QSO, STD, STAR, SKY]",
        LAMBDAREF    = "Reference wavelength at which to align fiber",
        TARGET_MASK0 = "Targeting bit mask",
        RA_TARGET    = "Target right ascension [degrees]",
        DEC_TARGET   = "Target declination [degrees]",
        X_TARGET     = "X on focal plane derived from (RA,DEC)_TARGET",
        Y_TARGET     = "Y on focal plane derived from (RA,DEC)_TARGET",
        X_FVCOBS     = "X location observed by Fiber View Cam [mm]",
        Y_FVCOBS     = "Y location observed by Fiber View Cam [mm]",
        X_FVCERR     = "X location uncertainty from Fiber View Cam [mm]",
        Y_FVCERR     = "Y location uncertainty from Fiber View Cam [mm]",
        RA_OBS       = "RA of obs from (X,Y)_FVCOBS and optics [deg]",
        DEC_OBS      = "dec of obs from (X,Y)_FVCOBS and optics [deg]",
        _SIMTYPE     = "True object type to simulate",
        _SIMZ        = "True redshift at which to simulate spectrum",
    )
    _add_table_comments(fibermap_file, 1, comments)
    
    return fibermap_file

def _add_table_comments(filename, hdu, comments):
    """
    Add comments to auto-generated FITS binary table column keywords.
    
    filename : FITS file to update
    hdu : HDU number with the table
    comments : dictionary of colname:comment
    """
    fx = fits.open(filename, mode='update')
    for i in range(1,100):
        key = 'TTYPE'+str(i)
        if key not in fx[hdu].header:
            break
        else:
            value = fx[hdu].header[key]
            if value in comments:
                fx[hdu].header[key] = (value, comments[value])
    
    fx.flush()
    fx.close()
    
