import os
import time

import specter  #- for throughput, psf
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
        _SIMTYPE     = "True object type to simulate",
        _SIMZ        = "True redshift at which to simulate spectrum",
    )

    #- Extra header keywords
    hdr = fits.Header()
    if tileid is not None:
        hdr['TILEID']   = (tileid, 'Tile ID')
        
    hdr['EXPID']    = (expid, 'Exposure number')
    hdr['NIGHT']    = (str(night), 'Night YEARMMDD')
    hdr['VDMODEL']  = ('0.0.0', 'TODO: desimodel version')
    hdr['VOPTICS']  = ('0.0.0', 'TODO: optics model version')
    hdr['VFIBVCAM'] = ('0.0.0', 'TODO: fiber view code version')
    hdr['HEXPDROT'] = (0.0, 'TODO: hexapod rotation [deg]')
    dateobs_str = time.strftime('%Y-%m-%dT%H:%M:%S', dateobs)
    hdr['DATE-OBS'] = (dateobs_str, 'Date of observation in UTC')

    #- TODO: Refactor.  tele_ra,dec comes from obs.get_tile_radec(),
    #- but obs itself imports io.  Can't have circular dependency.
    # hdr['TELERA']   = (tele_ra, 'Telescope central RA [deg]')
    # hdr['TELEDEC']  = (tele_dec, 'Telescope central dec [deg]')
    
    write_bintable(outfile, fibermap, hdr, comments=comments,
        extname="FIBERMAP", clobber=True)
        
    return outfile

def read_fibermap(night, expid):
    infile = '{}/fibermap-{:08d}.fits'.format(simdir(night), expid)
    
    fibermap = fits.getdata(infile, 'FIBERMAP')
    hdr = fits.getheader(infile, 'FIBERMAP')
    return fibermap, hdr

#-------------------------------------------------------------------------
#- desimodel

def load_throughput(channel):
    thrudir = os.getenv('DESIMODEL') + '/data/throughput'
    return specter.throughput.load_throughput(thrudir+'/thru-'+channel+'.fits')

def load_psf(channel):
    psfdir = os.getenv('DESIMODEL') + '/data/specpsf'
    return specter.psf.load_psf(psfdir+'/psf-'+channel+'.fits')

def load_desiparams():
    return yaml.load(open(os.getenv('DESIMODEL')+'/data/desi.yaml'))

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
        raise ValueError(key+" not set; can't find templates")
        
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

def write_bintable(filename, data, header, comments=None, units=None,
                   extname=None, clobber=False):
    """
    Utility function to write a binary table and get the comments and units
    in the FITS header too.
    """
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
                hdu.header['TUNIT'+str(i)] = (value, units[value])
    
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
    

