"""
desisim.obs
===========

Utility functions related to simulating observations for DESI
"""
from __future__ import absolute_import, division, print_function

import os, sys
import numpy as np
import yaml
import sqlite3
import fcntl
import time
from astropy.io import fits
from astropy import table
import astropy.units as u

import desimodel.io
import desispec.io
from desispec.interpolation import resample_flux

from desiutil.log import get_logger
log = get_logger()

from .targets import get_targets_parallel
from . import io
import desisim.simexp
from .simexp import simulate_spectra

def new_exposure(program, nspec=5000, night=None, expid=None, tileid=None,
                 seed=None, obsconditions=None, specify_targets=dict(), testslit=False, exptime=None,
                 arc_lines_filename=None, flat_spectrum_filename=None):
    """
    Create a new exposure and output input simulation files.
    Does not generate pixel-level simulations or noisy spectra.

    Args:
        program: 'arc', 'flat', 'bright', 'dark', 'bgs', 'mws', ...

    Options:
        * nspec : integer number of spectra to simulate
        * night : YEARMMDD string
        * expid : positive integer exposure ID
        * tileid : integer tile ID
        * seed : random seed
        * obsconditions: str or dict-like; see options below
        * specify_targets: (dict of dicts)  Define target properties like magnitude and redshift
                                 for each target class. Each objtype has its own key,value pair
                                 see simspec.templates.specify_galparams_dict() 
                                 or simsepc.templates.specify_starparams_dict()
        * exptime: float exposure time [seconds], overrides obsconditions['EXPTIME']
        * testslit : simulate test slit if True, default False; only for arc/flat
        * arc_lines_filename : use alternate arc lines filename (used if program="arc")
        * flat_spectrum_filename : use alternate flat spectrum filename (used if program="flat")

    Writes:
        * $DESI_SPECTRO_SIM/$PIXPROD/{night}/fibermap-{expid}.fits
        * $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits

    Returns:
        * science: sim, fibermap, meta, obsconditions

    input obsconditions can be a string 'dark', 'gray', 'bright', or dict-like
    observation metadata with keys SEEING (arcsec), EXPTIME (sec), AIRMASS,
    MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg).  Output obsconditions is
    is expanded dict-like structure.

    program is used to pick the sky brightness, and is propagated to
    desisim.targets.sample_objtype() to get the correct distribution of
    targets for a given program, e.g. ELGs, LRGs, QSOs for program='dark'.

    if program is 'arc' or 'flat', then `sim` is truth table with keys
    FLUX and WAVE; and meta=None and obsconditions=None.

    Also see simexp.simarc(), .simflat(), and .simscience(), the last of
    which simulates a science exposure given surveysim obsconditions input,
    fiber assignments, and pre-generated mock target spectra.
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
        night = str(night)  #- just in case we got an integer instead of string
        dateobs = time.strptime(night+':22', '%Y%m%d:%H')

    outsimspec = desisim.io.findfile('simspec', night, expid)
    outfibermap = desisim.io.findfile('simfibermap', night, expid)
    
    program = program.lower()
    log.debug('Generating {} targets'.format(nspec))
    
    header = dict(NIGHT=night, EXPID=expid, PROGRAM=program)
    if program in ('arc', 'flat'):
        header['FLAVOR'] = program
    else:
        header['FLAVOR'] = 'science'

    #- ISO 8601 DATE-OBS year-mm-ddThh:mm:ss
    header['DATE-OBS'] = time.strftime('%FT%T', dateobs)
    
    if program == 'arc':
        if arc_lines_filename is None :
            infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.3/arc-lines-average-in-vacuum.fits'
        else :
            infile = arc_lines_filename
        arcdata = fits.getdata(infile, 1)
        if exptime is None:
            exptime = 5
        wave, phot, fibermap = desisim.simexp.simarc(arcdata, nspec=nspec, testslit=testslit)

        header['EXPTIME'] = exptime
        desisim.io.write_simspec_arc(outsimspec, wave, phot, header, fibermap=fibermap)

        fibermap.meta['NIGHT'] = night
        fibermap.meta['EXPID'] = expid
        fibermap.write(outfibermap)
        truth = dict(WAVE=wave, PHOT=phot, UNITS='photon')
        return truth, fibermap, None, None
    
    elif program == 'flat':
        if flat_spectrum_filename is None :
            infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.3/flat-3100K-quartz-iodine.fits'
        else :
            infile = flat_spectrum_filename

        if exptime is None:
            exptime = 10
        sim, fibermap = desisim.simexp.simflat(infile, nspec=nspec, exptime=exptime, testslit=testslit)

        header['EXPTIME'] = exptime
        header['FLAVOR'] = 'flat'
        desisim.io.write_simspec(sim, truth=None, fibermap=fibermap, obs=None,
            expid=expid, night=night, header=header)

        fibermap.meta['NIGHT'] = night
        fibermap.meta['EXPID'] = expid
        fibermap.write(outfibermap)
        # fluxunits = 1e-17 * u.erg / (u.s * u.cm**2 * u.Angstrom)
        fluxunits = '1e-17 erg/(s * cm2 * Angstrom)'
        flux = sim.simulated['source_flux'].to(fluxunits)
        wave = sim.simulated['wavelength'].to('Angstrom')
        truth = dict(WAVE=wave, FLUX=flux, UNITS=str(fluxunits))
        return truth, fibermap, None, None

    #- all other programs
    fibermap, (flux, wave, meta) = get_targets_parallel(nspec, program, tileid=tileid, \
                                          seed=seed, specify_targets=specify_targets)

    if obsconditions is None:
        if program in ['dark', 'lrg', 'qso']:
            obsconditions = desisim.simexp.reference_conditions['DARK']
        elif program in ['elg', 'gray', 'grey']:
            obsconditions = desisim.simexp.reference_conditions['GRAY']
        elif program in ['mws', 'bgs', 'bright']:
            obsconditions = desisim.simexp.reference_conditions['BRIGHT']
        else:
            raise ValueError('unknown program {}'.format(program))
    elif isinstance(obsconditions, str):
        try:
            obsconditions = desisim.simexp.reference_conditions[obsconditions.upper()]
        except KeyError:
            raise ValueError('obsconditions {} not in {}'.format(
                obsconditions.upper(),
                list(desisim.simexp.reference_conditions.keys())))

    if exptime is not None:
        obsconditions['EXPTIME'] = exptime

    sim = simulate_spectra(wave, flux, fibermap=fibermap, obsconditions=obsconditions)

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
        PROGRAM = (program, 'program [dark, bright, ...]'),
        FLAVOR = ('science', 'Flavor [arc, flat, science, zero, ...]'),
        TELRA = (telera, 'Telescope pointing RA [degrees]'),
        TELDEC = (teledec, 'Telescope pointing dec [degrees]'),
        AIRMASS = (obsconditions['AIRMASS'], 'Airmass at middle of exposure'),
        EXPTIME = (obsconditions['EXPTIME'], 'Exposure time [sec]'),
        SEEING = (obsconditions['SEEING'], 'Seeing FWHM [arcsec]'),
        MOONFRAC = (obsconditions['MOONFRAC'], 'Moon illumination fraction 0-1; 1=full'),
        MOONALT  = (obsconditions['MOONALT'], 'Moon altitude [degrees]'),
        MOONSEP  = (obsconditions['MOONSEP'], 'Moon:tile separation angle [degrees]'),
        )
    hdr['DATE-OBS'] = (time.strftime('%FT%T', dateobs), 'Start of exposure')

    simfile = io.write_simspec(sim, meta, fibermap, obsconditions, expid, night, header=hdr)

    #- Write fibermap to $DESI_SPECTRO_SIM/$PIXPROD not $DESI_SPECTRO_DATA
    fiberfile = io.findfile('simfibermap', night, expid)
    desispec.io.write_fibermap(fiberfile, fibermap, header=hdr)
    log.info('Wrote '+fiberfile)

    update_obslog(obstype='science', program=program, expid=expid, dateobs=dateobs, tileid=tileid)

    #- Restore $DESI_SPECTRO_DATA
    if datadir_orig is not None:
        os.environ['DESI_SPECTRO_DATA'] = datadir_orig
    else:
        del os.environ['DESI_SPECTRO_DATA']

    return sim, fibermap, meta, obsconditions

#- Mapping of DESI objtype to things specter knows about
def specter_objtype(desitype):
    '''
    Convert a list of DESI object types into ones that specter knows about
    '''
    intype = np.atleast_1d(desitype)
    desi2specter = dict(
        STAR='STAR', STD='STAR', STD_FSTAR='STAR', FSTD='STAR', MWS_STAR='STAR',
        LRG='LRG', ELG='ELG', QSO='QSO', QSO_BAD='STAR',
        BGS='LRG', # !!!
        SKY='SKY'
    )

    unknown_types = set(intype) - set(desi2specter.keys())
    if len(unknown_types) > 0:
        raise ValueError('Unknown input objtypes {}'.format(unknown_types))

    results = np.zeros(len(intype), dtype=(str, 8))
    for objtype in desi2specter:
        ii = (intype == objtype)
        results[ii] = desi2specter[objtype]

    assert np.count_nonzero(results == '') == 0

    if isinstance(desitype, str):
        return results[0]
    else:
        return results

def get_next_tileid(program='DARK'):
    """
    Return tileid of next tile to observe

    Args:
        program (optional): dark, gray, or bright

    Note:
        Simultaneous calls will return the same tileid;
        it does *not* reserve the tileid.
    """
    program = program.upper()
    if program not in ('DARK', 'GRAY', 'GREY', 'BRIGHT',
                       'ELG', 'LRG', 'QSO', 'LYA', 'BGS', 'MWS'):
        return -1

    #- Read DESI tiling and trim to just tiles in DESI footprint
    tiles = table.Table(desimodel.io.load_tiles())

    #- HACK: update tilelist to include PROGRAM, etc.
    if 'PROGRAM' not in tiles.colnames:
        log.error('You are using an out-of-date desi-tiles.fits file from desimodel')
        log.error('please update your copy of desimodel/data')
        log.warning('proceeding anyway with a workaround for now...')
        tiles['PASS'] -= min(tiles['PASS'])  #- standardize to starting at 0 not 1

        brighttiles = tiles[tiles['PASS'] <= 2].copy()
        brighttiles['TILEID'] += 50000
        brighttiles['PASS'] += 5

        tiles = table.vstack([tiles, brighttiles])

        program_col = table.Column(name='PROGRAM', length=len(tiles), dtype=(str, 6))
        tiles.add_column(program_col)
        tiles['PROGRAM'][tiles['PASS'] <= 3] = 'DARK'
        tiles['PROGRAM'][tiles['PASS'] == 4] = 'GRAY'
        tiles['PROGRAM'][tiles['PASS'] >= 5] = 'BRIGHT'
    else:
        tiles['PROGRAM'] = np.char.strip(tiles['PROGRAM'])

    #- If obslog doesn't exist yet, start at tile 0
    dbfile = io.simdir()+'/etc/obslog.sqlite'
    if not os.path.exists(dbfile):
        obstiles = set()
    else:
        #- Read obslog to get tiles that have already been observed
        db = sqlite3.connect(dbfile)
        result = db.execute('SELECT tileid FROM obslog WHERE program="{}"'.format(program))
        obstiles = set( [row[0] for row in result] )
        db.close()

    #- Just pick the next tile in sequential order
    program_tiles = tiles['TILEID'][tiles['PROGRAM'] == program]
    nexttile = int(min(set(program_tiles) - obstiles))

    log.debug('{} tiles in program {}'.format(len(program_tiles), program))
    log.debug('{} observed tiles'.format(len(obstiles)))

    return nexttile

def get_next_expid(n=None):
    """
    Return the next exposure ID to use from {proddir}/etc/next_expid.txt
    and update the exposure ID in that file.

    Use file locking to prevent multiple readers from getting the same
    ID or accidentally clobbering each other while writing.

    Args:
        n (int, optional): number of contiguous expids to return as a list.
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
        return list(range(expid, expid+n))

def get_night(t=None, utc=None):
    """
    Return YEARMMDD for tonight.  The night roles over at local noon.
    i.e. 1am and 11am is the previous date; 1pm is the current date.

    Args:
        t : local time.struct_time tuple of integers
            (year, month, day, hour, min, sec, weekday, dayofyear, DSTflag)
            default is time.localtime(), i.e. now
        utc : time.struct_time tuple for UTC instead of localtime

    Note:
        this only has one second accuracy; good enough for sims but *not*
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
def update_obslog(obstype='science', program='DARK', expid=None, dateobs=None,
    tileid=-1, ra=None, dec=None):
    """
    Update obslog with a new exposure

    obstype : 'arc', 'flat', 'bias', 'test', 'science', ...
    program : 'DARK', 'GRAY', 'BRIGHT', 'CALIB'
    expid   : integer exposure ID, default from get_next_expid()
    dateobs : time.struct_time tuple; default time.localtime()
    tileid  : integer TileID, default -1, i.e. not a DESI tile
    ra, dec : float (ra, dec) coordinates, default tile ra,dec or (0,0)

    returns tuple (expid, dateobs)

    TODO: normalize obstype vs. program; see desisim issue #97
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
        program TEXT DEFAULT "DARK",
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
    INSERT OR REPLACE INTO obslog(expid,dateobs,night,obstype,program,tileid,ra,dec)
    VALUES (?,?,?,?,?,?,?,?)
    """
    db.execute(insert, (int(expid), time.mktime(dateobs), str(night),
        str(obstype.upper()), str(program.upper()), int(tileid),
        float(ra), float(dec)))
    db.commit()

    return expid, dateobs
