#- Utility functions related to simulating observations

import os, sys
import numpy as np
import yaml
import sqlite3
import fcntl
import time
from astropy.io import fits
from astropy.table import Table

from . import targets
from . import io

def get_fibermap(tileid=None, nfiber=5000):
    """
    Return a fake fibermap ndarray for this tileid
    
    Columns FIBER OBJTYPE RA DEC XFOCAL YFOCAL TARGET_MASK0
    
    TODO:
      - 5000 should not be hardcoded default
      - tileid is currently ignored
    """
    true_objtype, target_objtype = targets.sample_targets(nfiber)

    #- Load fiber -> positioner mapping
    fiberpos = fits.getdata(os.getenv('DESIMODEL')+'/data/focalplane/fiberpos.fits', upper=True)
            
    #- Where is this tile on the sky
    #- NOTE: more file I/O; seems clumsy
    tilera, tiledec = get_tile_radec(tileid)
                        
    #- Make a fake fibermap
    fibermap = dict()
    fibermap['FIBER'] = np.arange(nfiber, dtype='i4')
    fibermap['POSITIONER'] = fiberpos['POSITIONER'][0:nfiber]
    fibermap['SPECTROID'] = fiberpos['SPECTROGRAPH'][0:nfiber]
    fibermap['TARGETID'] = np.random.randint(sys.maxint, size=nfiber)
    fibermap['TARGETCAT'] = np.zeros(nfiber, dtype='|S20')
    fibermap['OBJTYPE'] = np.array(target_objtype)
    fibermap['LAMBDAREF'] = np.ones(nfiber, dtype=np.float32)*5400
    fibermap['TARGET_MASK0'] = np.zeros(nfiber, dtype='i8')
    fibermap['RA_TARGET'] = np.ones(nfiber, dtype='f8') * tilera   #- TODO
    fibermap['DEC_TARGET'] = np.ones(nfiber, dtype='f8') * tiledec #- TODO
    fibermap['X_TARGET'] = fiberpos['X'][0:nfiber]
    fibermap['Y_TARGET'] = fiberpos['Y'][0:nfiber]
    fibermap['X_FVCOBS'] = fibermap['X_TARGET']
    fibermap['Y_FVCOBS'] = fibermap['Y_TARGET']
    fibermap['X_FVCERR'] = np.zeros(nfiber, dtype=np.float32)
    fibermap['Y_FVCERR'] = np.zeros(nfiber, dtype=np.float32)
    fibermap['RA_OBS'] = fibermap['RA_TARGET']
    fibermap['DEC_OBS'] = fibermap['DEC_TARGET']
    fibermap['_SIMTYPE'] = np.array(true_objtype)

    #- convert fibermap into numpy ndarray with named columns
    ### keys = fibermap.keys()
    #- Ensure a friendly order of columns
    keys = ['FIBER', 'POSITIONER', 'SPECTROID', 'TARGETID', 'TARGETCAT',
            'OBJTYPE', 'LAMBDAREF', 'TARGET_MASK0',
            'RA_TARGET', 'DEC_TARGET', 'RA_OBS', 'DEC_OBS',
            'X_TARGET', 'Y_TARGET',
            'X_FVCOBS', 'Y_FVCOBS', 'X_FVCERR', 'Y_FVCERR',
            '_SIMTYPE']
    assert set(keys) == set(fibermap.keys())
    dtype = zip(keys, [fibermap[k].dtype for k in keys])
    cols = [fibermap[k] for k in keys]
    rows = zip(*cols)
    fibermap = np.array(rows, dtype)
    
    return fibermap
    
    
def new_exposure(fibermap_file, dateobs=None, camera=None):
    """
    TODO: document
    
    TODO: refactor to not need to read throughputs and project sky photons
        every time
    """
    #- parse camera b0 -> channel b, ispec 0
    channel = camera[0]
    ispec = int(camera[1])
    assert channel.lower() in 'brz'
    assert 0 <= ispec < 10
    
    #- Load throughput
    thru = io.load_throughput(channel)

    #- Other DESI parameters
    params = io.load_desiparams()
    nspec = params['spectro']['nfibers']

    #- Read fibermap file
    outdir = os.path.split(fibermap_file)[0]
    fibermap = fits.getdata(fibermap_file, 'FIBERMAP')
    fmhdr = fits.getheader(fibermap_file, 'FIBERMAP')
    
    #- Trim to just the fibers for this spectrograph
    fibermap = fibermap[nspec*ispec:nspec*(ispec+1)]

    #- Get expid from FIBERMAP
    expid = fmhdr['EXPID']

    #- Get object flux
    dw = 0.2
    wave = np.arange(round(thru.wavemin, 1), thru.wavemax, dw)
    nwave = len(wave)
    nspec = len(fibermap)
    flux = np.zeros( (nspec, len(wave)) )
    z = np.zeros(nspec, dtype='f4')
    mag = np.zeros(nspec, dtype='f4')
    magtype = np.zeros(nspec, dtype='S8')
    templateid = np.zeros(nspec, dtype='i4')
    o2flux = np.zeros(nspec, dtype='f4')
    for objtype in set(fibermap['_SIMTYPE']):
        if objtype == 'SKY':
            continue
            
        ii = np.where(fibermap['_SIMTYPE'] == objtype)[0]
        try:
            simflux, meta = io.read_templates(wave, objtype, len(ii))
        except ValueError:
            print "ERROR: unable to load {} templates".format(objtype)
            continue
            
        flux[ii] = simflux
        
        #- STD don't have Z; others do
        if 'Z' in meta:
            z[ii] = meta['Z']
            
        templateid[ii] = meta['TEMPLATEID']
        if objtype == 'ELG':
            o2flux[ii] = meta['OII_3727']
            
        for x in ('SDSS_R', 'DECAM_R', 'DECAM_Z'):
            if x in meta.dtype.names:
                mag[ii] = meta[x]
                magtype[ii] = x
                break

    #- Load sky [Magic knowledge of units 1e-17 erg/s/cm2/A/arcsec2]
    skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky.dat'
    skywave, skyflux = np.loadtxt(skyfile, unpack=True)
    skyflux = np.interp(wave, skywave, skyflux)

    #- Project flux to photons
    phot = thru.photons(wave, flux, units='1e-17 erg/s/cm2/A',
            objtype=fibermap['_SIMTYPE'], exptime=params['exptime'])
    
    skyphot = thru.photons(wave, skyflux, units='1e-17 erg/s/cm2/A/arcsec2',
        objtype='SKY', exptime=params['exptime'])
    
    #- Convert sky into 2D; someday it may vary
    skyflux = np.tile(skyflux, nspec).reshape((nspec, nwave))
    skyphot = np.tile(skyphot, nspec).reshape((nspec, nwave))
    
    #- Use astropy Table as a convenient way to create an ndarray
    tmp = Table([flux, phot, skyflux, skyphot, z, mag, magtype, templateid, o2flux],
                names=['FLUX', 'PHOT', 'SKYFLUX', 'SKYPHOT', 'Z', 'MAG', 'MAGTYPE', 'TEMPLATEID', 'OII_3727'])
    spectra = tmp._data


    #- Extend fmhdr with additional keywords
    
    hdr = fmhdr
    hdr['CAMERA'] = (camera, 'Spectograph Camera')
    hdr['VSPECTER'] = ('0.0.0', 'TODO: Specter version')
    hdr['EXPTIME'] = (params['exptime'], 'Exposure time [sec]')
    hdr['CRVAL1'] = (wave[0], 'Starting wavelength [Angstroms]')
    hdr['CDELT1'] = (dw, 'Wavelength step [Angstroms]')
    hdr['LOGLAM'] = (0, 'Linear wavelength grid')
    hdr['WAVEUNIT'] = ('Angstrom', 'wavelength units')
    hdr['AIRORVAC'] = ('vac', 'wavelengths in vacuum (vac) or air')

    comments = dict(
        FLUX = 'Object flux [1e-17 erg/s/cm^2/A]',
        PHOT = 'Object photons per bin (not per A)',
        SKYFLUX = 'Sky flux [1e-17 erg/s/cm^2/A/arcsec^2]',
        SKYPHOT = 'Sky photons per bin (not per A)',
    )
    units = dict(
        FLUX = '1e-17 erg/s/cm^2/A',
        PHOT = 'counts/bin',
        SKYFLUX = '1e-17 erg/s/cm^2/A/arcsec^2',
        SKYPHOT = 'counts/bin',
    )

    simfile = '{}/sim-{}-{:08d}.fits'.format(outdir, camera, expid)
    io.write_bintable(simfile, spectra, hdr,
        comments=comments, units=units,
        extname=camera.upper()+"-SPECTRA", clobber=True)
        
    return simfile


def get_next_tile():
    """
    Return tileid, ra, dec of next tile to observe
    
    Note: simultaneous calls will return the same tileid;
          it does *not* reserve the tileid
    """
    #- Read DESI tiling and trim to just tiles in DESI footprint
    footprint = os.getenv('DESIMODEL')+'/data/footprint/desi-tiles.fits'
    tiles = fits.getdata(footprint)
    tiles = tiles[tiles['IN_DESI'] > 0]

    #- If obslog doesn't exist yet, start at tile 0
    dbfile = io.simdir()+'/etc/obslog.sqlite'
    if not os.path.exists(dbfile):
        return 0
    
    #- Read obslog to get tiles that have already been observed
    db = sqlite3.connect(dbfile)
    result = db.execute('SELECT tileid FROM obslog')
    obstiles = set( [row[0] for row in result] )
    db.close()
    
    #- Just pick the next tile in sequential order
    nexttile = int(min(set(tiles['TILEID']) - obstiles))
    i = np.where(tiles['TILEID'] == nexttile)[0][0]
        
    return nexttile, tiles[i]['RA'], tiles[i]['DEC']
    
def get_tile_radec(tileid):
    footprint = os.getenv('DESIMODEL')+'/data/footprint/desi-tiles.fits'
    tiles = fits.getdata(footprint)
    if tileid in tiles['TILEID']:
        i = np.where(tiles['TILEID'] == tileid)[0][0]
        return tiles[i]['RA'], tiles[i]['DEC']
    else:
        return (0.0, 0.0)
    
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

    night = get_night(dateobs)
        
    insert = """\
    INSERT INTO obslog(expid,dateobs,night,obstype,tileid,ra,dec)
    VALUES (?,?,?,?,?,?,?)
    """
    db.execute(insert, (expid, time.mktime(dateobs), night, obstype, tileid, ra, dec))
    db.commit()
    
    return expid, dateobs
    

