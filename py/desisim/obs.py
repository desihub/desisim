#- Utility functions related to simulating observations

import os, sys
import numpy as np
import yaml
import fitsio
import sqlite3
import fcntl
import time

from . import targets

def get_next_tile():
    """
    Return tileid of next tile to observe
    
    Note: simultaneous calls will return the same tileid;
          it does *not* reserve the tileid
    """
    #- Read DESI tiling and trim to just tiles in DESI footprint
    footprint = os.getenv('DESIMODEL')+'/data/footprint/desi-tiles.fits'
    tiles = fitsio.read(footprint)
    tiles = tiles[tiles['IN_DESI'] > 0]

    #- If obslog doesn't exist yet, start at tile 0
    dbfile = os.getenv('DESI_SPECTRO_SIM')+'/etc/obslog.sqlite'
    if not os.path.exists(dbfile):
        return 0
    
    #- Read obslog to get tiles that have already been observed
    db = sqlite3.connect(dbfile)
    result = db.execute('SELECT tileid FROM obslog')
    obstiles = set( [row[0] for row in result] )
    db.close()
    
    #- Just pick the next tile in sequential order
    nexttile = min(set(tiles['TILEID']) - obstiles)
        
    return nexttile
    
def get_fibermap(tileid=None, nfiber=5000):
    """
    Return a fake fibermap ndarray for this tileid
    
    Columns FIBER OBJTYPE RA DEC XFOCAL YFOCAL TARGET_MASK0
    
    BUGS: 5000 should not be hardcoded default
    """
    true_objtype, target_objtype, z = targets.sample_targets(nfiber)

    #- Load fiber -> positioner mapping
    fiberpos = fitsio.read(os.getenv('DESIMODEL')+'/data/focalplane/fiberpos.fits', upper=True)
    
    #- Make a fake fibermap
    fibermap = dict()
    fibermap['FIBER'] = np.arange(nfiber, dtype='i4')
    fibermap['POSITIONER'] = fiberpos['POSITIONER'][0:nfiber]
    fibermap['TARGETID'] = np.random.randint(sys.maxint, size=nfiber)
    fibermap['OBJTYPE'] = np.array(target_objtype)
    fibermap['_SIMTYPE'] = np.array(true_objtype)
    fibermap['_SIMZ'] = z
    fibermap['TARGET_MASK0'] = np.zeros(nfiber, dtype='i8')
    fibermap['RA'] = np.zeros(nfiber, dtype='f8')
    fibermap['DEC'] = np.zeros(nfiber, dtype='f8')
    fibermap['XFOCAL'] = fiberpos['X'][0:nfiber]
    fibermap['YFOCAL'] = fiberpos['Y'][0:nfiber]

    #- convert fibermap into numpy ndarray with named columns
    ### keys = fibermap.keys()
    #- Ensure a friendly order of columns
    keys = ['FIBER', 'POSITIONER', 'TARGETID', 'OBJTYPE', '_SIMTYPE', '_SIMZ',
        'TARGET_MASK0', 'RA', 'DEC', 'XFOCAL', 'YFOCAL']
    assert set(keys) == set(fibermap.keys())
    dtype = zip(keys, [fibermap[k].dtype for k in keys])
    cols = [fibermap[k] for k in keys]
    rows = zip(*cols)
    fibermap = np.array(rows, dtype)
    
    return fibermap
    
def get_next_obs(t=None):
    """
    Return night, expid, tileid, fibermap for next observation to perform.
    
    Optional inputs:
    t : time.struct_time tuple of integers
        (year, month, day, hour, min, sec, weekday, dayofyear, DSTflag)
        default is time.localtime(), i.e. now
    
    Increments exposure ID counter.
    """
    expid = get_next_expid()
    tileid = get_next_tile()
    return get_night(t), expid, tileid, get_fibermap(tileid)

def get_next_expid(n=None):
    """
    Return the next exposure ID to use from $DESI_SPECTRO_SIM/etc/next_expid.txt
    and update the exposure ID in that file.
    
    Use file locking to prevent multiple readers from getting the same
    ID or accidentally clobbering each other while writing.
    
    Optional Input:
    n : integer, number of contiguous expids to return as a list.
        If None, return a scalar. Note that n=1 returns a list of length 1.
    
    BUGS:
      * if etc/next_expid.txt doesn't exist, initial file creation is
        probably not threadsafe.
    """
    #- Full path to next_expid.txt file
    filename = os.getenv('DESI_SPECTRO_SIM')+'/etc/next_expid.txt'
    
    if not os.path.exists(os.getenv('DESI_SPECTRO_SIM')+'/etc/'):
        os.makedirs(os.getenv('DESI_SPECTRO_SIM')+'/etc/')

    #- Create file if needed; is this threadsafe?  Probably not.
    if not os.path.exists(filename):
        fw = open(filename, 'w')
        fw.write("0\n")
        fw.close()
    
    #- Open the file, but get exclusive lock before reading
    f0 = open(filename)
    fcntl.flock(f0, fcntl.LOCK_EX)
    expid = int(f0.readline())
    
    #- Write update expid to the file
    fw = open(filename, 'w')
    if n is None:
        fw.write(str(expid+1)+'\n')
    else:
        fw.write(str(expid+n)+'\n')        
    fw.close()
    
    #- Release the file lock
    fcntl.flock(f0, fcntl.LOCK_UN)
    f0.close()
    
    if n is None:
        return expid
    else:
        return range(expid, expid+n)
    
def get_night(t=None):
    """
    Return YEARMMDD for tonight.  The night roles over at local noon.
    i.e. 1am and 11am is the previous date; 1pm is the current date.
    
    Optional inputs:
    t : time.struct_time tuple of integers
        (year, month, day, hour, min, sec, weekday, dayofyear, DSTflag)
        default is time.localtime(), i.e. now
    
    Note: this only has one second accuracy; good enough for sims but *not*
          to be used for actual DESI ops.
    """
    if t is None:
        t = time.localtime()
    #- what date/time was it 12 hours ago? "Night" rolls over at noon.
    night = time.localtime(time.mktime(t) - 12*3600)
    #- format that as YEARMMDD
    return time.strftime('%Y%m%d', night)
    
#- I'm not really sure this is a good idea.
#- I'm sure I will want to change the schema later...
def update_obslog(obstype='science', expid=None, dateobs=None, tileid=-1, ra=0.0, dec=0.0):
    """
    Update obslog with a new exposure
    
    obstype : 'science', 'arc', 'flat', 'bias', 'dark', or 'test'
    expid   : integer exposure ID, default from get_next_expid()
    dateobs : time.struct_time tuple; default time.localtime()
    tileid  : integer TileID, default -1, i.e. not a DESI tile
    ra, dec : float (ra, dec) coordinates, default (0,0)
    
    returns tuple (expid, dateobs)
    """
    #- Connect to sqlite database file and create DB if needed
    dbfile = os.path.join(os.environ['DESI_SPECTRO_SIM'], 'etc', 'obslog.sqlite')
    db = sqlite3.connect(dbfile)
    db.execute("""\
    CREATE TABLE IF NOT EXISTS obslog (
        expid INTEGER PRIMARY KEY,
        dateobs DATETIME,                   -- seconds since Unix Epoch (1970)
        night TEXT,                         -- YEAR-MM-DD
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

    night = get_night(dateobs)
        
    insert = """\
    INSERT INTO obslog(expid,dateobs,night,obstype,tileid,ra,dec)
    VALUES (?,?,?,?,?,?,?)
    """
    db.execute(insert, (expid, time.mktime(dateobs), night, obstype, tileid, ra, dec))
    db.commit()
    
    return expid, dateobs
    
