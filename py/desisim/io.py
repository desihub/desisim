import os
import time

import specter.psf
import specter.throughput
import yaml     #- for desi.yaml
from astropy.io import fits
import numpy as np

from desisim.interpolation import resample_flux

#-------------------------------------------------------------------------
#- Fibermap

def write_fibermap(fibermap, expid, night, dateobs, tileid=None):
    """    
    Writes
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/fibermap-{expid}.fits
       
    Inputs:
      - fibermap : ndarray with named columns of fibermap data
      - expid : exposure ID (integer)
      - night : string YEARMMDD
      - dateobs : time tuple with UTC time; see time.gmtime()
        
    Returns full path to filename of fibermap file written
    """
    #- Where should this be written?  Create dir if needed.
    outdir = simdir(night, mkdir=True)      
    outfile = '{}/fibermap-{:08d}.fits'.format(outdir, expid)

    #- Comments for fibermap columns
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
        MAG          = "magitude",
        FILTER       = "SDSS_R, DECAM_Z, WISE1, etc."
    )

    #- Extra header keywords
    hdr = fits.Header()
    if tileid is not None:
        hdr['TILEID']   = (tileid, 'Tile ID')
        tele_ra, tele_dec = get_tile_radec(tileid)
        hdr['TELERA']   = (tele_ra, 'Telescope central RA [deg]')
        hdr['TELEDEC']  = (tele_dec, 'Telescope central dec [deg]')
    else:
        hdr['TELERA']   = (0.0, 'Telescope central RA [deg]')
        hdr['TELEDEC']  = (0.0, 'Telescope central dec [deg]')        
        
    hdr['EXPID']    = (expid, 'Exposure number')
    hdr['NIGHT']    = (str(night), 'Night YEARMMDD')
    hdr['VDMODEL']  = ('0.0.0', 'TODO: desimodel version')
    hdr['VOPTICS']  = ('0.0.0', 'TODO: optics model version')
    hdr['VFIBVCAM'] = ('0.0.0', 'TODO: fiber view code version')
    hdr['HEXPDROT'] = (0.0, 'TODO: hexapod rotation [deg]')
    dateobs_str = time.strftime('%Y-%m-%dT%H:%M:%S', dateobs)
    hdr['DATE-OBS'] = (dateobs_str, 'Date of observation in UTC')

    write_bintable(outfile, fibermap, hdr, comments=comments,
        extname="FIBERMAP", clobber=True)
        
    return outfile

def read_fibermap(night, expid):
    infile = '{}/fibermap-{:08d}.fits'.format(simdir(night), expid)
    
    fibermap = fits.getdata(infile, 'FIBERMAP')
    hdr = fits.getheader(infile, 'FIBERMAP')
    return fibermap, hdr

#-------------------------------------------------------------------------
#- simspec

def write_simspec(meta, truth, expid, night, header=None):
    """
    Write $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits
    
    Args:
        meta : metadata table to write to "METADATA" HDU
        truth : dictionary with keys:
            FLUX - 2D array [nspec, nwave] in erg/s/cm2/A
            WAVE - 1D array of vacuum wavelengths [Angstroms]
            SKYFLUX - array of sky flux [erg/s/cm2/A/arcsec],
                      either 1D [nwave] or 2D [nspec, nwave]
            PHOT_{B,R,Z} - 2D array [nspec, nwave] of object photons/bin
            SKYPHOT_{B,R,Z} - 1D or 2D array of sky photons/bin
    """
    #- Where should this go?
    outdir = simdir(night, mkdir=True)      
    outfile = '{}/simspec-{:08d}.fits'.format(outdir, expid)

    #- Object flux
    hdr = fits.Header()
    if header is not None:
        for key, value in header.items():
            hdr[key] = value
            
    wave = truth['WAVE']
    hdr['CRVAL1']    = (wave[0], 'Starting wavelength [Angstroms]')
    hdr['CDELT1']    = (wave[1]-wave[0], 'Wavelength step [Angstroms]')
    hdr['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
    hdr['LOGLAM']    = (0, 'linear wavelength steps, not log10')
    hdr['EXTNAME']   = ('FLUX', 'Object flux [erg/s/cm2/A]')
    fits.writeto(outfile, truth['FLUX'], header=hdr, clobber=True)
    
    #- Sky flux
    hdr['EXTNAME'] = ('SKYFLUX', 'Sky flux [erg/s/cm2/A/arcsec2]')
    hdu = fits.ImageHDU(truth['SKYFLUX'], header=hdr)
    fits.append(outfile, hdu.data, header=hdu.header)
    
    #- Metadata table
    comments = dict(
        OBJTYPE     = 'Object type (ELG, LRG, QSO, STD, STAR)',
        REDSHIFT    = 'true object redshift',
        TEMPLATEID  = 'input template ID',
        O2FLUX      = '[OII] flux [erg/s/cm2]',
    )
    
    units = dict(
        # OBJTYPE     = 'Object type (ELG, LRG, QSO, STD, STAR)',
        # REDSHIFT    = 'true object redshift',
        # TEMPLATEID  = 'input template ID',
        O2FLUX      = 'erg/s/cm2',
    )
    
    write_bintable(outfile, meta, header=None, extname="METADATA",
        comments=comments, units=units)

    #- Write object photon and sky photons for each channel
    for channel in ['B', 'R', 'Z']:
        hdr = fits.Header()
        wave = truth['WAVE_'+channel]
        hdr['CRVAL1']    = (wave[0], 'Starting wavelength [Angstroms]')
        hdr['CDELT1']    = (wave[1]-wave[0], 'Wavelength step [Angstroms]')
        hdr['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
        hdr['LOGLAM']    = (0, 'linear wavelength steps, not log10')
        
        extname = 'PHOT_'+channel
        hdr['EXTNAME']   = (extname, channel+' channel object photons per bin')
        hdu = fits.ImageHDU(truth[extname], header=hdr)
        fits.append(outfile, hdu.data, header=hdu.header)

        extname = 'SKYPHOT_'+channel
        hdr['EXTNAME']   = (extname, channel+' channel sky photons per bin')
        hdu = fits.ImageHDU(truth[extname], header=hdr)
        fits.append(outfile, hdu.data, header=hdu.header)
                            
    return outfile

#-------------------------------------------------------------------------
#- Parse header to make wavelength array
def load_wavelength(filename, extname):
    hdr = fits.getheader(filename, extname)
    wave = hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1']
    if hdr['LOGLAM'] == 1:
        wave = 10**wave
    return wave
    
#-------------------------------------------------------------------------
#- desimodel
#- These should probably move to desimodel itself,
#- except that brings in extra dependencies for desimodel.

_thru = dict()
def load_throughput(channel):
    channel = channel.lower()
    global _thru
    if channel not in _thru:
        thrudir = os.getenv('DESIMODEL') + '/data/throughput'
        _thru[channel] = specter.throughput.load_throughput(thrudir+'/thru-'+channel+'.fits')
    
    return _thru[channel]

_psf = dict()
def load_psf(channel):
    channel = channel.lower()
    global _psf
    if channel not in _psf:
        psfdir = os.getenv('DESIMODEL') + '/data/specpsf'
        _psf[channel] = specter.psf.load_psf(psfdir+'/psf-'+channel+'.fits')
    
    return _psf[channel]

_params = None
def load_desiparams():
    global _params
    if _params is None:
        _params = yaml.load(open(os.getenv('DESIMODEL')+'/data/desi.yaml'))
        
    return _params

_fiberpos = None
def load_fiberpos():
    global _fiberpos
    if _fiberpos is None:
        _fiberpos = fits.getdata(os.getenv('DESIMODEL')+'/data/focalplane/fiberpos.fits', upper=True)
    
    return _fiberpos

_tiles = None
def load_tiles(onlydesi=True):
    """
    Return DESI tiles structure from desimodel
    
    if onlydesi is True, trim to just the tiles in the DESI footprint
    """
    global _tiles
    if _tiles is None:
        footprint = os.getenv('DESIMODEL')+'/data/footprint/desi-tiles.fits'
        _tiles = fits.getdata(footprint)
    
    if onlydesi:
        return _tiles[_tiles['IN_DESI'] > 0]
    else:
        return _tiles

def get_tile_radec(tileid):
    """
    Return (ra, dec) in degrees for the requested tileid.
    
    If tileid is not in DESI, return (0.0, 0.0)
    TODO: should it raise and exception instead?
    """
    tiles = load_tiles()
    if tileid in tiles['TILEID']:
        i = np.where(tiles['TILEID'] == tileid)[0][0]
        return tiles[i]['RA'], tiles[i]['DEC']
    else:
        return (0.0, 0.0)   

#-------------------------------------------------------------------------
#- spectral templates

def read_templates(wave, objtype, n, randseed=1):
    """
    Returns n templates of type objtype sampled at wave
    
    Inputs:
      - wave : array of wavelengths to sample
      - objtype : 'ELG', 'LRG', 'QSO', 'STD', or 'STAR'
      - n : number of templates to return
    
    Returns flux[n, len(wave)], meta[n]
    
    where meta is a metadata table from the input template file
    with redshift, mags, etc.
    
    Requires $DESI_{objtype}_TEMPLATES to be set, pointing to a file that
    has the observer frame flux in HDU 0 and a metadata table for these
    objects in HDU 1.  This code randomly samples n spectra from that file.
    
    TO DO: add a setable randseed for random reproducibility.
    """    
    key = 'DESI_'+objtype.upper()+'_TEMPLATES'
    if key not in os.environ:
        raise ValueError("ERROR: $"+key+" not set; can't find "+objtype+" templates")
        
    infile = os.getenv(key)
    hdr = fits.getheader(infile)
    flux = fits.getdata(infile, 0)
    meta = fits.getdata(infile, 1).view(np.recarray)
    ww = 10**(hdr['CRVAL1'] + np.arange(hdr['NAXIS1'])*hdr['CDELT1'])

    ntemplates = flux.shape[0]
    randindex = np.arange(ntemplates)
    np.random.shuffle(randindex)
    
    outflux = np.zeros([n, len(wave)])
    outmeta = np.empty(n, dtype=meta.dtype)
    for i in range(n):
        j = randindex[i%ntemplates]
        if 'Z' in meta:
            z = meta['Z'][j]
        else:
            z = 0.0
        outflux[i] = resample_flux(wave, ww*(1+z), flux[j])
        outmeta[i] = meta[j]
        
    return outflux, outmeta
    
    

#-------------------------------------------------------------------------
#- Utility functions

def write_bintable(filename, data, header=None, comments=None, units=None,
                   extname=None, clobber=False):
    """
    Utility function to write a binary table and get the comments and units
    in the FITS header too.
    """
    
    #- Convert data from dictionary of columns to ndarray if needed
    # if isinstance(data, dict):
    #     dtype = list()
    #     for key in data:
    #         dtype.append( (key, data[key].dtype, data[key].shape) )
    #     nrows = len(data[key])  #- use last column to get length
    #     xdata = np.empty(nrows, dtype=dtype)
    #     for key in data:
    #         xdata[key] = data[key]            
    #     data = xdata
    
    #- Write the data and header
    hdu = fits.BinTableHDU(data, header=header, name=extname)
    if clobber:
        fits.writeto(filename, hdu.data, hdu.header, clobber=True)
    else:
        fits.append(filename, hdu.data, hdu.header)

    #- Allow comments and units to be None
    if comments is None:
        comments = dict()
    if units is None:
        units = dict()

    #- Reopen the file to add the comments and units
    fx = fits.open(filename, mode='update')
    hdu = fx[extname]
    for i in xrange(1,999):
        key = 'TTYPE'+str(i)
        if key not in hdu.header:
            break
        else:
            value = hdu.header[key]
            if value in comments:
                hdu.header[key] = (value, comments[value])
            if value in units:
                hdu.header['TUNIT'+str(i)] = (units[value], value+' units')
    
    #- Write updated header and close file
    fx.flush()
    fx.close()

def simdir(night='', mkdir=False):
    """
    Return $DESI_SPECTRO_SIM/$PIXPROD/{night}
    If mkdir is True, create directory if needed
    """
    dirname = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'), night)        
    if mkdir and not os.path.exists(dirname):
        os.makedirs(dirname)
        
    return dirname
    
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
        


#-------------------------------------------------------------------------
# def _add_table_comments(filename, hdu, comments):
#     """
#     Add comments to auto-generated FITS binary table column keywords.
#     
#     filename : FITS file to update
#     hdu : HDU number with the table
#     comments : dictionary of colname:comment
#     """
#     fx = fits.open(filename, mode='update')
#     for i in range(1,100):
#         key = 'TTYPE'+str(i)
#         if key not in fx[hdu].header:
#             break
#         else:
#             value = fx[hdu].header[key]
#             if value in comments:
#                 fx[hdu].header[key] = (value, comments[value])
#     
#     fx.flush()
#     fx.close()
    

