#- Utility functions related to simulating observations

import os, sys
import numpy as np
import yaml
import sqlite3
import fcntl
import time
from astropy.io import fits
from astropy.table import Table

import desimodel.io
import desispec.io
from desispec.interpolation import resample_flux

from targets import get_targets
from . import io

def new_exposure(flavor, nspec=5000, night=None, expid=None, tileid=None, \
    airmass=1.0, exptime=None):
    """
    Create a new exposure and output input simulation files.
    Does not generate pixel-level simulations or noisy spectra.
    
    Args:
        nspec (optional): integer number of spectra to simulate
        night (optional): YEARMMDD string
        expid (optional): positive integer exposure ID
        tileid (optional): tile ID
        airmass (optional): airmass, default 1.0
    
    Writes:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/fibermap-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits
        
    Returns:
        fibermap numpy structured array
        truth dictionary
    """
    if expid is None:
        expid = get_next_expid()
    
    if tileid is None:
        tileid = get_next_tileid()

    if night is None:
        #- simulation obs time = now, even if sun is up
        dateobs = time.gmtime()
        night = get_night(utc=dateobs)
    else:
        #- 10pm on night YEARMMDD
        dateobs = time.strptime(night+':22', '%Y%m%d:%H')
    
    params = desimodel.io.load_desiparams()    
    if flavor == 'arc':
        infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.2/arc-lines-average.fits'
        d = fits.getdata(infile, 1)
        wave = d['AIRWAVE']
        phot = d['ELECTRONS']
        
        truth = dict(WAVE=wave)
        meta = None
        fibermap = desispec.io.fibermap.empty_fibermap(nspec)
        for channel in ('B', 'R', 'Z'):
            thru = desimodel.io.load_throughput(channel)        
            ii = np.where( (thru.wavemin <= wave) & (wave <= thru.wavemax) )[0]
            truth['WAVE_'+channel] = wave[ii]
            truth['PHOT_'+channel] = np.tile(phot[ii], nspec).reshape(nspec, len(ii))

    elif flavor == 'flat':
        infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.2/flat-3100K-quartz-iodine.fits'
        flux = fits.getdata(infile, 0)
        hdr = fits.getheader(infile, 0)
        wave = desispec.io.util.header2wave(hdr)

        #- resample to 0.2 A grid
        dw = 0.2
        ww = np.arange(wave[0], wave[-1]+dw/2, dw)
        flux = resample_flux(ww, wave, flux)
        wave = ww

        #- Convert to 2D for projection
        flux = np.tile(flux, nspec).reshape(nspec, len(wave))

        truth = dict(WAVE=wave, FLUX=flux)
        meta = None
        fibermap = desispec.io.fibermap.empty_fibermap(nspec)
        for channel in ('B', 'R', 'Z'):
            thru = desimodel.io.load_throughput(channel)
            ii = (thru.wavemin <= wave) & (wave <= thru.wavemax)
            phot = thru.photons(wave[ii], flux[:,ii], units=hdr['BUNIT'],
                            objtype='CALIB', exptime=exptime)
        
            truth['WAVE_'+channel] = wave[ii]
            truth['PHOT_'+channel] = phot
        
    elif flavor == 'science':
        fibermap, truth = get_targets(nspec, tileid=tileid)
            
        flux = truth['FLUX']
        wave = truth['WAVE']
        nwave = len(wave)
    
        if exptime is None:
            exptime = params['exptime']
    
        #- Load sky [Magic knowledge of units 1e-17 erg/s/cm2/A/arcsec2]
        skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky.dat'
        skywave, skyflux = np.loadtxt(skyfile, unpack=True)
        skyflux = np.interp(wave, skywave, skyflux)
        truth['SKYFLUX'] = skyflux

        for channel in ('B', 'R', 'Z'):
            thru = desimodel.io.load_throughput(channel)
        
            ii = np.where( (thru.wavemin <= wave) & (wave <= thru.wavemax) )[0]
        
            #- Project flux to photons
            phot = thru.photons(wave[ii], flux[:,ii], units='1e-17 erg/s/cm2/A',
                    objtype=truth['OBJTYPE'], exptime=exptime,
                    airmass=airmass)
                
            truth['PHOT_'+channel] = phot
            truth['WAVE_'+channel] = wave[ii]
    
            #- Project sky flux to photons
            skyphot = thru.photons(wave[ii], skyflux[ii]*airmass,
                units='1e-17 erg/s/cm2/A/arcsec2',
                objtype='SKY', exptime=exptime, airmass=airmass)
    
            #- 2D version
            ### truth['SKYPHOT_'+channel] = np.tile(skyphot, nspec).reshape((nspec, len(ii)))
            #- 1D version
            truth['SKYPHOT_'+channel] = skyphot.astype(np.float32)
        
        #- NOTE: someday skyflux and skyphot may be 2D instead of 1D
        
        #- Extract the metadata part of the truth dictionary into a table
        columns = (
            'OBJTYPE',
            'REDSHIFT',
            'TEMPLATEID',
            'O2FLUX',
        )
        meta = {key: truth[key] for key in columns}
        
    #- (end indentation for arc/flat/science flavors)
        
    #- Override $DESI_SPECTRO_DATA in order to write to simulation area
    datadir_orig = os.getenv('DESI_SPECTRO_DATA')
    simbase = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'))
    os.environ['DESI_SPECTRO_DATA'] = simbase

    #- Write fibermap
    telera, teledec = io.get_tile_radec(tileid)
    hdr = dict(
        NIGHT = (night, 'Night of observation YEARMMDD'),
        EXPID = (expid, 'DESI exposure ID'),
        TILEID = (tileid, 'DESI tile ID'),
        FLAVOR = (flavor, 'Flavor [arc, flat, science, ...]'),
        TELRA = (telera, 'Telescope pointing RA [degrees]'),
        TELDEC = (teledec, 'Telescope pointing dec [degrees]'),
        )
    fiberfile = desispec.io.findfile('fibermap', night, expid)
    desispec.io.write_fibermap(fiberfile, fibermap, header=hdr)
    print fiberfile
    
    #- Write simspec
    hdr = dict(
        AIRMASS=(airmass, 'Airmass at middle of exposure'),
        EXPTIME=(exptime, 'Exposure time [sec]'),
        FLAVOR=(flavor, 'exposure flavor [arc, flat, science]'),
        )
    simfile = io.write_simspec(meta, truth, expid, night, header=hdr)
    print simfile

    #- Update obslog that we succeeded with this exposure
    update_obslog(flavor, expid, dateobs, tileid)
    
    #- Restore $DESI_SPECTRO_DATA
    if datadir_orig is not None:
        os.environ['DESI_SPECTRO_DATA'] = datadir_orig
    else:
        del os.environ['DESI_SPECTRO_DATA']
    
    return fibermap, truth

def get_next_tileid():
    """
    Return tileid of next tile to observe
    
    Note: simultaneous calls will return the same tileid;
          it does *not* reserve the tileid
    """
    #- Read DESI tiling and trim to just tiles in DESI footprint
    tiles = desimodel.io.load_tiles()

    #- If obslog doesn't exist yet, start at tile 0
    dbfile = io.simdir()+'/etc/obslog.sqlite'
    if not os.path.exists(dbfile):
        obstiles = set()
    else:
        #- Read obslog to get tiles that have already been observed
        db = sqlite3.connect(dbfile)
        result = db.execute('SELECT tileid FROM obslog')
        obstiles = set( [row[0] for row in result] )
        db.close()
    
    #- Just pick the next tile in sequential order
    nexttile = int(min(set(tiles['TILEID']) - obstiles))        
    return nexttile
    
def get_next_expid(n=None):
    """
    Return the next exposure ID to use from {proddir}/etc/next_expid.txt
    and update the exposure ID in that file.
    
    Use file locking to prevent multiple readers from getting the same
    ID or accidentally clobbering each other while writing.
    
    Optional Input:
    n : integer, number of contiguous expids to return as a list.
        If None, return a scalar. Note that n=1 returns a list of length 1.
    
    BUGS:
      * if etc/next_expid.txt doesn't exist, initial file creation is
        probably not threadsafe.
      * File locking mechanism doesn't work on NERSC Edison, to turned off
        for now.
    """
    #- Full path to next_expid.txt file
    filename = io.simdir()+'/etc/next_expid.txt'
    
    if not os.path.exists(io.simdir()+'/etc/'):
        os.makedirs(io.simdir()+'/etc/')

    #- Create file if needed; is this threadsafe?  Probably not.
    if not os.path.exists(filename):
        fw = open(filename, 'w')
        fw.write("0\n")
        fw.close()
    
    #- Open the file, but get exclusive lock before reading
    f0 = open(filename)
    ### fcntl.flock(f0, fcntl.LOCK_EX)
    expid = int(f0.readline())
    
    #- Write update expid to the file
    fw = open(filename, 'w')
    if n is None:
        fw.write(str(expid+1)+'\n')
    else:
        fw.write(str(expid+n)+'\n')        
    fw.close()
    
    #- Release the file lock
    ### fcntl.flock(f0, fcntl.LOCK_UN)
    f0.close()
    
    if n is None:
        return expid
    else:
        return range(expid, expid+n)
    
def get_night(t=None, utc=None):
    """
    Return YEARMMDD for tonight.  The night roles over at local noon.
    i.e. 1am and 11am is the previous date; 1pm is the current date.
    
    Optional inputs:
    t : local time.struct_time tuple of integers
        (year, month, day, hour, min, sec, weekday, dayofyear, DSTflag)
        default is time.localtime(), i.e. now
    utc : time.struct_time tuple for UTC instead of localtime
    
    Note: this only has one second accuracy; good enough for sims but *not*
          to be used for actual DESI ops.
    """
    #- convert t to localtime or fetch localtime if needed
    if utc is not None:
        t = time.localtime(time.mktime(utc) - time.timezone)
    elif t is None:
        t = time.localtime()
        
    #- what date/time was it 12 hours ago? "Night" rolls over at noon local
    night = time.localtime(time.mktime(t) - 12*3600)
    
    #- format that as YEARMMDD
    return time.strftime('%Y%m%d', night)
    
#- I'm not really sure this is a good idea.
#- I'm sure I will want to change the schema later...
def update_obslog(obstype='science', expid=None, dateobs=None,
    tileid=-1, ra=None, dec=None):
    """
    Update obslog with a new exposure
    
    obstype : 'science', 'arc', 'flat', 'bias', 'dark', or 'test'
    expid   : integer exposure ID, default from get_next_expid()
    dateobs : time.struct_time tuple; default time.localtime()
    tileid  : integer TileID, default -1, i.e. not a DESI tile
    ra, dec : float (ra, dec) coordinates, default tile ra,dec or (0,0)
    
    returns tuple (expid, dateobs)
    """
    #- Connect to sqlite database file and create DB if needed
    dbdir = io.simdir() + '/etc'
    if not os.path.exists(dbdir):
        os.makedirs(dbdir)
        
    dbfile = dbdir+'/obslog.sqlite'
    db = sqlite3.connect(dbfile)
    db.execute("""\
    CREATE TABLE IF NOT EXISTS obslog (
        expid INTEGER PRIMARY KEY,
        dateobs DATETIME,                   -- seconds since Unix Epoch (1970)
        night TEXT,                         -- YEARMMDD
        obstype TEXT DEFAULT "science",
        tileid INTEGER DEFAULT -1,
        ra REAL DEFAULT 0.0,
        dec REAL DEFAULT 0.0
    )
    """)
    
    #- Fill in defaults
    if expid is None:
        expid = get_next_expid()
    
    if dateobs is None:
        dateobs = time.localtime()

    if ra is None:
        assert (dec is None)
        if tileid < 0:
            ra, dec = (0.0, 0.0)
        else:
            ra, dec = io.get_tile_radec(tileid)
            
    night = get_night(utc=dateobs)
        
    insert = """\
    INSERT OR REPLACE INTO obslog(expid,dateobs,night,obstype,tileid,ra,dec)
    VALUES (?,?,?,?,?,?,?)
    """
    db.execute(insert, (expid, time.mktime(dateobs), night, obstype, tileid, ra, dec))
    db.commit()
    
    return expid, dateobs

#- for the future
# def ndarray_from_columns(keys, columns):
#     nrow = len(columns[0])
#     dtype = list()
#     for name, col in zip(keys, columns):
#         dtype.append( (name, col.dtype) )
#     
#     result = np.zeros(nrow, dtype=dtype)
#     for name, col in zip(keys, columns):
#         result[name] = col
# 
#     return result



