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

from desispec.log import get_logger
log = get_logger()

from .targets import get_targets_parallel
from . import io
import desisim.newexp
from .newexp import simulate_spectra

def testslit_fibermap() :
    # from WBS 1.6 PDR Fiber Slit document
    # science slit has 20 bundles of 25 fibers
    # test slit has 1 fiber per bundle except in the middle where it is fully populated
    nspectro=10
    testslit_nspec_per_spectro=20
    testslit_nspec = nspectro*testslit_nspec_per_spectro
    fibermap = np.zeros(testslit_nspec, dtype=desispec.io.fibermap.fibermap_columns)
    fibermap['FIBER'] = np.zeros((testslit_nspec)).astype(int)
    fibermap['SPECTROID'] = np.zeros((testslit_nspec)).astype(int)
    for spectro in range(nspectro) :
        fiber_index=testslit_nspec_per_spectro*spectro
        first_fiber_id=500*spectro
        for b in range(20) :
            # Fibers at Top of top block or Bottom of bottom block
            if b <= 10:
                fibermap['FIBER'][fiber_index]  = 25*b + first_fiber_id
            else:
                fibermap['FIBER'][fiber_index]  = 25*b + 24 + first_fiber_id

            fibermap['SPECTROID'][fiber_index] = spectro
            fiber_index+=1
    return fibermap

def new_exposure(flavor, nspec=5000, night=None, expid=None, tileid=None,
                 airmass=1.0, exptime=None, seed=None, testslit=False,
                 arc_lines_filename=None, flat_spectrum_filename=None):

    """
    Create a new exposure and output input simulation files.
    Does not generate pixel-level simulations or noisy spectra.

    Args:
        flavor: 'arc', 'flat', 'bright', 'dark', 'bgs', 'mws', ...

    Options:
        nspec : integer number of spectra to simulate
        night : YEARMMDD string
        expid : positive integer exposure ID
        tileid : integer tile ID
        airmass : airmass, default 1.0
        exptime : exposure time in seconds
        seed : random seed
        testslit : simulate test slit if True, default False
        arc_lines_filename : use alternate arc lines filename (used if flavor="arc")
        flat_spectrum_filename : use alternate flat spectrum filename (used if flavor="flat")

    Writes:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/fibermap-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits

    Returns:
        fibermap numpy structured array
        truth dictionary

    Notes:
        flavor is used to pick the sky brightness, and is propagated to
        desisim.targets.sample_objtype() to get the correct distribution of
        targets for a given flavor, e.g. ELGs, LRGs, QSOs for flavor='dark'.
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
    
    flavor = flavor.lower()
    log.debug('Generating {} targets'.format(nspec))
    
    if flavor == 'arc':
        if arc_lines_filename is None :
            infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.3/arc-lines-average-in-vacuum.fits'
        else :
            infile = arc_lines_filename
        arcdata = fits.getdata(infile, 1)
        wave, phot, fibermap = desisim.newexp.newarc(arcdata, nspec=nspec)
        header = dict(NIGHT=night, EXPID=expid, EXPTIME=5, FLAVOR='arc')
        desisim.io.write_simspec_arc(outsimspec, wave, phot, header)
        fibermap.meta['NIGHT'] = night
        fibermap.meta['EXPID'] = expid
        fibermap.write(outfibermap)
        truth = dict(WAVE=wave, PHOT=phot, UNITS='photon')
        return fibermap, truth
    
    elif flavor == 'flat':
        if flat_spectrum_filename is None :
            infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.3/flat-3100K-quartz-iodine.fits'
        else :
            infile = flat_spectrum_filename

        exptime = 10
        sim, fibermap = desisim.newexp.newflat(infile, nspec=nspec, exptime=exptime)
        header = dict(NIGHT=night, EXPID=expid, EXPTIME=exptime, FLAVOR='flat')
        desisim.io.write_simspec(sim, None, expid, night, header=header)
        fibermap.meta['NIGHT'] = night
        fibermap.meta['EXPID'] = expid
        fibermap.write(outfibermap)
        # fluxunits = 1e-17 * u.erg / (u.s * u.cm**2 * u.Angstrom)
        fluxunits = '1e-17 erg/(s * cm2 * Angstrom)'
        flux = sim.simulated['source_flux'].to(fluxunits)
        wave = sim.simulated['wavelength'].to('Angstrom')
        truth = dict(WAVE=wave, FLUX=flux, UNITS=str(fluxunits))
        return fibermap, truth

    #- all other flavors
    fibermap, truth = get_targets_parallel(nspec, flavor, tileid=tileid, seed=seed)

    fibermap = table.Table(fibermap)
    fibermap.remove_columns(['OBJTYPE', 'MAG'])
    meta = table.hstack([fibermap, table.Table(truth['META'])])

    wave = truth['WAVE']
    flux = truth['FLUX']
    if flavor.upper() in ['DARK', 'LRG', 'QSO']:
        obsconditions = 'DARK'
    elif flavor.upper() in ['ELG', 'GRAY', 'GREY']:
        obsconditions = 'GRAY'
    elif flavor.upper() in ['MWS', 'BGS', 'BRIGHT']:
        obsconditions = 'BRIGHT'
    else:
        raise ValueError('unknown flavor {}'.format(flavor))

    obsconditions = desisim.newexp.reference_conditions[obsconditions]
    if exptime is not None:
        obsconditions['EXPTIME'] = exptime
    if airmass is not None:
        obsconditions['AIRMASS'] = airmass

    # sim = simulate_spectra(wave, flux, meta=meta, obsconditions=obsconditions, galsim=False)
    sim = simulate_spectra(wave, 1e-17*flux, meta=fibermap, obsconditions=obsconditions, galsim=False)

    #- Override $DESI_SPECTRO_DATA in order to write to simulation area
    datadir_orig = os.getenv('DESI_SPECTRO_DATA')
    simbase = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'))
    os.environ['DESI_SPECTRO_DATA'] = simbase

    #- Copy some per-camera post-convolution results into the truth dict
    for camera, results in zip(sim.camera_names, sim.camera_output):
        camera = camera.upper()
        truth['WAVE_'+camera] = results['wavelength']
        truth['PHOT_'+camera] = results['num_source_electrons']
        truth['SKYPHOT_'+camera] = results['num_sky_electrons']

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
    #- ISO 8601 DATE-OBS year-mm-ddThh:mm:ss
    fiberfile = desispec.io.findfile('fibermap', night, expid)
    desispec.io.write_fibermap(fiberfile, fibermap, header=hdr)
    log.info('Wrote '+fiberfile)

    #- Write simspec; expand fibermap header
    hdr['AIRMASS'] = (obsconditions['AIRMASS'], 'Airmass at middle of exposure')
    hdr['EXPTIME'] = (obsconditions['EXPTIME'], 'Exposure time [sec]')
    hdr['DATE-OBS'] = (time.strftime('%FT%T', dateobs), 'Start of exposure')

    simfile = io.write_simspec(sim, meta, expid, night, header=hdr)

    update_obslog(obstype='science', program=flavor, expid=expid, dateobs=dateobs, tileid=tileid)

    #- Restore $DESI_SPECTRO_DATA
    if datadir_orig is not None:
        os.environ['DESI_SPECTRO_DATA'] = datadir_orig
    else:
        del os.environ['DESI_SPECTRO_DATA']

    return fibermap, truth

def _orig_new_exposure(flavor, nspec=5000, night=None, expid=None, tileid=None,
                 airmass=1.0, exptime=None, seed=None, testslit=False,
                 arc_lines_filename=None, flat_spectrum_filename=None):

    """
    Create a new exposure and output input simulation files.
    Does not generate pixel-level simulations or noisy spectra.

    Args:
        flavor: 'arc', 'flat', 'bright', 'dark', 'bgs', 'mws', ...

    Options:
        nspec : integer number of spectra to simulate
        night : YEARMMDD string
        expid : positive integer exposure ID
        tileid : integer tile ID
        airmass : airmass, default 1.0
        exptime : exposure time in seconds
        seed : random seed
        testslit : simulate test slit if True, default False
        arc_lines_filename : use alternate arc lines filename (used if flavor="arc")
        flat_spectrum_filename : use alternate flat spectrum filename (used if flavor="flat")

    Writes:
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/fibermap-{expid}.fits
        $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits

    Returns:
        fibermap numpy structured array
        truth dictionary

    Notes:
        flavor is used to pick the sky brightness, and is propagated to
        desisim.targets.sample_objtype() to get the correct distribution of
        targets for a given flavor, e.g. ELGs, LRGs, QSOs for flavor='dark'.
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

    params = desimodel.io.load_desiparams()
    flavor = flavor.lower()
    if flavor == 'arc':
        if arc_lines_filename is None :
            infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.3/arc-lines-average-in-vacuum.fits'
        else :
            infile = arc_lines_filename
        d, arc_header = fits.getdata(infile, 1, header=True)

        #- v0.3 comment "typical exptime for BOSS=5s" but didn't set EXPTIME
        if 'EXPTIME' not in arc_header:
            arc_header['EXPTIME'] = 5.0

        if exptime is None:
            exptime = arc_header['EXPTIME']

        keys = d.columns.names

        wave=None
        phot=None
        elec=None

        if ( 'VACUUM_WAVE' in keys ) :
            wave = d['VACUUM_WAVE']
        elif ( 'WAVE' in keys ) :
            wave = d['WAVE']
        if ("ELECTRONS" in keys) :
            elec = d['ELECTRONS']
        elif ("PHOTONS" in keys) :
            phot= d['PHOTONS']

        if ( ( phot is None ) and (elec is None) ) or wave is None :
            log.error("cannot read arc line fits file '%s' (don't know the format)"%infile)
            raise KeyError("cannot read arc line fits file")


        truth = dict(WAVE=wave)
        meta = None
        if  testslit :
            fibermap = testslit_fibermap()
            if nspec != fibermap.size :
                log.warning("forcing nspec %d -> %d (testslit sim.)"%(nspec,fibermap.size))
                nspec=fibermap.size
        else :
            fibermap = desispec.io.fibermap.empty_fibermap(nspec)

        fibermap['OBJTYPE'] = 'ARC'
        for channel in ('B', 'R', 'Z'):
            thru = desimodel.io.load_throughput(channel)
            ii = np.where( (thru.wavemin <= wave) & (wave <= thru.wavemax) )[0]
            truth['WAVE_'+channel] = wave[ii]
            if phot is not None :
                elec = phot*np.interp(wave,thru._wave,thru._thru,left=0,right=0)

            elec *= exptime / arc_header['EXPTIME']
            truth['PHOT_'+channel] = np.tile(elec[ii], nspec).reshape(nspec, len(ii))

    elif flavor == 'flat':

        if flat_spectrum_filename is None :
            infile = os.getenv('DESI_ROOT')+'/spectro/templates/calib/v0.3/flat-3100K-quartz-iodine.fits'
        else :
            infile = flat_spectrum_filename

        flux, flat_header = fits.getdata(infile, 0, header=True)
        wave = desispec.io.util.header2wave(flat_header)

        #- v0.3 flat doesn't specify EXPTIME;
        #- we've been treating it as 10 sec to not saturate CCDs
        if 'EXPTIME' not in flat_header:
            flat_header['EXPTIME'] = 10.0

        if exptime is None:
            exptime = flat_header['EXPTIME']

        #- resample to 0.2 A grid
        dw = 0.2
        ww = np.arange(wave[0], wave[-1]+dw/2, dw)
        flux = resample_flux(ww, wave, flux)
        wave = ww

        truth = dict(WAVE=wave, FLUX=flux)
        meta = None
        if  testslit :
            fibermap = testslit_fibermap()
            if nspec != fibermap.size :
                log.warning("forcing nspec %d -> %d (testslit sim.)"%(nspec,fibermap.size))
                nspec=fibermap.size
        else :
            fibermap = desispec.io.fibermap.empty_fibermap(nspec)

        #- Convert to 2D for projection
        flux = np.tile(flux, nspec).reshape(nspec, len(wave))

        fibermap['OBJTYPE'] = 'FLAT'
        for channel in ('B', 'R', 'Z'):
            thru = desimodel.io.load_throughput(channel)
            ii = (thru.wavemin <= wave) & (wave <= thru.wavemax)
            phot = thru.photons(wave[ii], flux[:,ii], units=flat_header['BUNIT'],
                            objtype='CALIB', exptime=exptime)

            truth['WAVE_'+channel] = wave[ii]
            truth['PHOT_'+channel] = phot

    else: # checked that flavor is valid in newexp-desi
        log.debug('Generating {} targets'.format(nspec))
        fibermap, truth = get_targets_parallel(nspec, flavor, tileid=tileid, seed=seed)

        flux = truth['FLUX']
        wave = truth['WAVE']
        nwave = len(wave)

        if exptime is None:
            exptime = params['exptime']

        #- Load sky [Magic knowledge of units 1e-17 erg/s/cm2/A/arcsec2]
        if flavor in ('bright', 'bgs', 'mws'):
            skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky-bright.dat'
        elif flavor in ('gray', 'grey'):
            skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky-grey.dat'
        else:
            skyfile = os.getenv('DESIMODEL')+'/data/spectra/spec-sky.dat'

        log.info('skyfile '+skyfile)

        skywave, skyflux = np.loadtxt(skyfile, unpack=True)
        skyflux = np.interp(wave, skywave, skyflux)
        truth['SKYFLUX'] = skyflux

        log.debug('Calculating flux -> photons')
        for channel in ('B', 'R', 'Z'):
            thru = desimodel.io.load_throughput(channel)

            ii = np.where( (thru.wavemin <= wave) & (wave <= thru.wavemax) )[0]

            #- Project flux to photons
            phot = thru.photons(wave[ii], flux[:,ii], units=truth['UNITS'],
                    objtype=specter_objtype(truth['OBJTYPE']),
                    exptime=exptime, airmass=airmass)

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
        meta = truth['META']

        #columns = (
        #    'OBJTYPE',
        #    'REDSHIFT',
        #    'TEMPLATEID',
        #    'D4000',
        #    'OIIFLUX',
        #    'VDISP'
        #)
        #meta = {key: truth[key] for key in columns}

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
    #- ISO 8601 DATE-OBS year-mm-ddThh:mm:ss
    fiberfile = desispec.io.findfile('fibermap', night, expid)
    desispec.io.write_fibermap(fiberfile, fibermap, header=hdr)
    log.info('Wrote '+fiberfile)

    #- Write simspec; expand fibermap header
    hdr['AIRMASS'] = (airmass, 'Airmass at middle of exposure')
    hdr['EXPTIME'] = (exptime, 'Exposure time [sec]')
    hdr['DATE-OBS'] = (time.strftime('%FT%T', dateobs), 'Start of exposure')

    simfile = io.write_simspec(meta, truth, expid, night, header=hdr)
    log.info('Wrote '+simfile)

    #- Update obslog that we succeeded with this exposure
    if flavor in ('arc', 'flat'):
        update_obslog(obstype=flavor, program='calib', expid=expid, dateobs=dateobs, tileid=tileid)
    else:
        update_obslog(obstype='science', program=flavor, expid=expid, dateobs=dateobs, tileid=tileid)

    #- Restore $DESI_SPECTRO_DATA
    if datadir_orig is not None:
        os.environ['DESI_SPECTRO_DATA'] = datadir_orig
    else:
        del os.environ['DESI_SPECTRO_DATA']

    return fibermap, truth

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
            #- TODO: mod 50k is to convert bright tileids to locations of
            #- dark tileids; update this when new tiling file exists
            ra, dec = io.get_tile_radec(tileid % 50000)

    night = get_night(utc=dateobs)

    insert = """\
    INSERT OR REPLACE INTO obslog(expid,dateobs,night,obstype,program,tileid,ra,dec)
    VALUES (?,?,?,?,?,?,?,?)
    """
    db.execute(insert, (expid, time.mktime(dateobs), night, obstype.upper(), program.upper(), tileid, ra, dec))
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
