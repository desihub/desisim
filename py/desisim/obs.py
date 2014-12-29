#- Utility functions related to simulating observations

import os, sys
import numpy as np
import yaml
import sqlite3
import fcntl
import time
from astropy.io import fits
from astropy.table import Table

from targets import get_targets
from . import io

#- Utility function; should probably be moved elsewhere
def _dict2ndarray(data, columns=None):
    """
    Convert a dictionary of ndarrays into a structured ndarray
    
    Args:
        data: input dictionary, each value is an ndarray
        columns: optional list of column names
        
    Notes:
        data[key].shape[0] must be the same for every key
        every entry in columns must be a key of data
    
    Example
        d = dict(x=np.arange(10), y=np.arange(10)/2)
        nddata = _dict2ndarray(d, columns=['x', 'y'])
    """
    if columns is None:
        columns = data.keys()
        
    dtype = list()
    for key in columns:
        ### dtype.append( (key, data[key].dtype, data[key].shape) )
        if data[key].ndim == 1:
            dtype.append( (key, data[key].dtype) )
        else:
            dtype.append( (key, data[key].dtype, data[key].shape[1:]) )
        
    nrows = len(data[key])  #- use last column to get length    
    xdata = np.empty(nrows, dtype=dtype)
    
    for key in columns:
        xdata[key] = data[key]
    
    return xdata
        

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

def new_exposure(nspec=5000, expid=None, tileid=None, airmass=1.0):
    """
    Create a new exposure and output input simulation files.
    Does not generate pixel-level simulations or noisy spectra.
    
    Args:
        nspec (optional): integer number of spectra to simulate
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
        
    dateobs = time.gmtime()
    night = get_night(utc=dateobs)
    
    fibermap, truth = get_targets(nspec, tileid=tileid)
    flux = truth['FLUX']
    wave = truth['WAVE']
    nwave = len(wave)
    
    params = io.load_desiparams()
    
    #- Load sky [Magic knowledge of units 1e-17 erg/s/cm2/A/arcsec2]
    skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky.dat'
    skywave, skyflux = np.loadtxt(skyfile, unpack=True)
    skyflux = np.interp(wave, skywave, skyflux)
    truth['SKYFLUX'] = skyflux

    for channel in ('B', 'R', 'Z'):
        thru = io.load_throughput(channel)
        
        ii = np.where( (thru.wavemin <= wave) & (wave <= thru.wavemax) )[0]
        
        #- Project flux to photons
        phot = thru.photons(wave[ii], flux[:,ii], units='1e-17 erg/s/cm2/A',
                objtype=truth['OBJTYPE'], exptime=params['exptime'],
                airmass=airmass)
                
        truth['PHOT_'+channel] = phot
        truth['WAVE_'+channel] = wave[ii]
    
        #- Project sky flux to photons
        skyphot = thru.photons(wave[ii], skyflux[ii]*airmass,
            units='1e-17 erg/s/cm2/A/arcsec2',
            objtype='SKY', exptime=params['exptime'], airmass=airmass)
    
        #- 2D version
        ### truth['SKYPHOT_'+channel] = np.tile(skyphot, nspec).reshape((nspec, len(ii)))
        #- 1D version
        truth['SKYPHOT_'+channel] = skyphot
        
    #- NOTE: someday skyflux and skyphot may be 2D instead of 1D
    
    #- Convert to ndarrays to get nice column order in output files
    columns = (
        'OBJTYPE',
        'TARGETCAT',
        'TARGETID',
        'TARGET_MASK0',
        'MAG',
        'FILTER',
        'SPECTROID',
        'POSITIONER',
        'FIBER',
        'LAMBDAREF',
        'RA_TARGET',
        'DEC_TARGET',
        'RA_OBS',
        'DEC_OBS',
        'X_TARGET',
        'Y_TARGET',
        'X_FVCOBS',
        'Y_FVCOBS',
        'Y_FVCERR',
        'X_FVCERR',
        )
    fibermap = _dict2ndarray(fibermap, columns)
    
    #- Extract the metadata part of the truth dictionary into a table
    columns = (
        'OBJTYPE',
        'REDSHIFT',
        'TEMPLATEID',
        'O2FLUX',
    )
    meta = _dict2ndarray(truth, columns)
        
    #- Write fibermap
    fiberfile = io.write_fibermap(fibermap, expid, night, dateobs, tileid=tileid)
    print fiberfile
    
    #- Write simfile
    simfile = io.write_simspec(meta, truth, expid, night)
    print simfile

    #- Update obslog that we succeeded with this exposure
    update_obslog('science', expid, dateobs, tileid)
    
    return fibermap, truth
    
    
def get_next_tileid():
    """
    Return tileid of next tile to observe
    
    Note: simultaneous calls will return the same tileid;
          it does *not* reserve the tileid
    """
    #- Read DESI tiling and trim to just tiles in DESI footprint
    tiles = io.load_tiles()

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
    dbfile = io.simdir()+'/etc/obslog.sqlite'
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
    INSERT INTO obslog(expid,dateobs,night,obstype,tileid,ra,dec)
    VALUES (?,?,?,?,?,?,?)
    """
    db.execute(insert, (expid, time.mktime(dateobs), night, obstype, tileid, ra, dec))
    db.commit()
    
    return expid, dateobs
    

