from __future__ import absolute_import, division, print_function

import sys
import os.path
import warnings
import datetime, time

import numpy as np

import astropy.table
import astropy.time
from astropy.io import fits

import desitarget
import desispec.io
import desispec.io.util
import desimodel.io
import desiutil.depend
import desispec.interpolation
import desisim.io

#- Reference observing conditions for each of dark, gray, bright
#- TODO: verify numbers
reference_conditions = dict(DARK=dict(), GRAY=dict(), BRIGHT=dict())
reference_conditions['DARK']['SEEING']  = 1.1
reference_conditions['DARK']['EXPTIME'] = 1000
reference_conditions['DARK']['AIRMASS'] = 1.25
reference_conditions['DARK']['MOONFRAC'] = 0.0
reference_conditions['DARK']['MOONALT'] = -60
reference_conditions['DARK']['MOONSEP'] = 180

reference_conditions['GRAY']['SEEING']  = 1.1
reference_conditions['GRAY']['EXPTIME'] = 1000
reference_conditions['GRAY']['AIRMASS'] = 1.25
reference_conditions['GRAY']['MOONFRAC'] = 0.1
reference_conditions['GRAY']['MOONALT']  = 10
reference_conditions['GRAY']['MOONSEP'] = 60

reference_conditions['BRIGHT']['SEEING']  = 1.1
reference_conditions['BRIGHT']['EXPTIME'] = 300
reference_conditions['BRIGHT']['AIRMASS'] = 1.25
reference_conditions['BRIGHT']['MOONFRAC'] = 0.7
reference_conditions['BRIGHT']['MOONALT']  = 60
reference_conditions['BRIGHT']['MOONSEP'] = 50

def newarc(arcdata, nspec=5000, nonuniform=False):
    '''
    Simulates an arc lamp exposure

    Args:
        arcdata: Table with columns VACUUM_WAVE and ELECTRONS

    Options:
        nspec: (int) number of spectra to simulate
        nonuniform: (bool) include calibration screen non-uniformity

    Returns: (wave, phot, fibermap)
        wave: 1D[nwave] wavelengths in Angstroms
        phot: 2D[nspec,nwave] photons observed by CCD (i.e. electrons)
        fibermap: fibermap Table

    Note: this bypasses specsim since we don't have an arclamp model in
    surface brightness units; we only have electrons on the CCD.  But it
    does include the effect of varying fiber sizes.

    TODO:
        * add exptime support
        * update inputs to surface brightness and DESI lamp lines (DESI-2674)
    '''
    wave = arcdata['VACUUM_WAVE']
    phot = arcdata['ELECTRONS']

    fibermap = astropy.table.Table(desispec.io.empty_fibermap(nspec))
    fibermap.meta['FLAVOR'] = 'arc'
    fibermap['OBJTYPE'] = 'ARC'

    x = fibermap['X_TARGET']
    y = fibermap['Y_TARGET']
    r = np.sqrt(x**2 + y**2)

    #-----
    #- Determine ratio of fiber sizes relative to larges fiber
    fiber_area = fiber_area_arcsec2(fibermap['X_TARGET'], fibermap['Y_TARGET'])
    size_ratio = fiber_area / np.max(fiber_area)

    #- Correct photons for fiber size
    phot = np.tile(phot, nspec).reshape(nspec, len(wave))
    phot = (phot.T * size_ratio).T

    #- Apply calibration screen non-uniformity
    if nonuniform:
        ratio = _calib_screen_uniformity(radius=r)
        assert np.all(ratio <= 1) and np.all(ratio > 0.99)
        phot = (phot.T * ratio).T

    return wave, phot, fibermap

def newflat(flatfile, nspec=5000, nonuniform=False, exptime=10):
    '''
    Simulates a flat lamp calibration exposure

    Args:
        flatfile: filename with flat lamp spectrum data

    Options:
        nspec: (int) number of spectra to simulate
        nonuniform: (bool) include calibration screen non-uniformity
        exptime: (float) exposure time in seconds

    Returns: (sim, fibermap)
        sim: specsim Simulator object
        fibermap: fibermap Table
    '''
    import astropy.units as u
    import specsim.simulator
    from desiutil.log import get_logger
    log = get_logger()

    log.info('Reading flat lamp spectrum from {}'.format(flatfile))
    sbflux, hdr = fits.getdata(flatfile, header=True)
    wave = desispec.io.util.header2wave(hdr)
    assert len(wave) == len(sbflux)

    #- Trim to DESI wavelength ranges
    #- TODO: is there an easier way to get these parameters?
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    ii = (wavemin <= wave) & (wave <= wavemax)
    wave = wave[ii]
    sbflux = sbflux[ii]

    #- Downsample to 0.2A grid to not blow up memory
    ww = np.arange(wave[0], wave[-1]+0.1, 0.2)
    sbflux = desispec.interpolation.resample_flux(ww, wave, sbflux)
    wave = ww

    fibermap = astropy.table.Table(desispec.io.empty_fibermap(nspec))
    fibermap.meta['FLAVOR'] = 'flat'
    fibermap['OBJTYPE'] = 'FLAT'
    x = fibermap['X_TARGET']
    y = fibermap['Y_TARGET']
    r = np.sqrt(x**2 + y**2)
    xy = np.vstack([x, y]).T * u.mm

    #- Convert to unit-ful 2D
    sbunit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 * u.arcsec ** 2)
    sbflux = np.tile(sbflux, nspec).reshape(nspec, len(wave)) * sbunit

    if nonuniform:
        ratio = _calib_screen_uniformity(radius=r)
        assert np.all(ratio <= 1) and np.all(ratio > 0.99)
        sbflux = (sbflux.T * ratio).T
        tmp = np.min(sbflux) / np.max(sbflux)
        log.info('Adjusting for calibration screen non-uniformity {:.4f}'.format(tmp))

    log.debug('Creating specsim configuration')
    config = _specsim_config_for_wave(wave)
    log.debug('Creating specsim simulator for {} spectra'.format(nspec))
    sim = specsim.simulator.Simulator(config, num_fibers=nspec)
    sim.observation.exposure_time = exptime * u.s
    log.debug('Simulating')
    sim.simulate(calibration_surface_brightness=sbflux, focal_positions=xy)

    return sim, fibermap

def _calib_screen_uniformity(theta=None, radius=None):
    '''
    Returns calibration screen relative non-uniformity as a function
    of theta (degrees) or focal plane radius (mm)
    '''
    if theta is not None:
        assert radius is None
        #- Julien Guy fit to DESI-2761v1 figure 5
        #- ratio lamp/sky = 1 - 9.4e-04*theta - 2.1e-03 * theta**2
        return 1 - 9.4e-04*theta - 2.1e-03 * theta**2
    elif radius is not None:
        import desimodel.io
        ps = desimodel.io.load_platescale()
        theta = np.interp(radius, ps['radius'], ps['theta'])
        return _calib_screen_uniformity(theta=theta)
    else:
        raise ValueError('must provide theta or radius')

def newexp(fiberassign, mockdir, obsconditions=None, expid=None, nspec=None):
    '''
    Simulates a new DESI exposure

    Args:
        fiberassign: Table of fiber assignments
        mockdir: (str) directory containing mock targets and truth spectra
            from select_mock_targets

    Options:
        obsconditions: observation metadata with keys
            SEEING (arcsec), EXPTIME (sec), AIRMASS,
            MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
        expid: (int) exposure ID
        nspec: (int) number of spectra to simulate

    obsconditions can be one of the following
        dict or row of Table
        str: DARK or GRAY or BRIGHT
        Table including EXPID for subselection of which row to use
        filename with obsconditions Table; expid must also be set

    Returns sim, fibermap, meta
        sim: specsim.simulate.Simulator object
        fibermap: Table
        meta: target metadata truth table
    '''
    from desiutil.log import get_logger
    log = get_logger()

    if nspec is not None:
        fiberassign = fiberassign[0:nspec]

    flux, wave, meta = get_mock_spectra(fiberassign, mockdir=mockdir)

    assert np.all(fiberassign['TARGETID'] == meta['TARGETID'])
    fiberassign.remove_column('TARGETID')
    fibermeta = astropy.table.hstack([meta, fiberassign])
    fibermap = fibermeta2fibermap(fibermeta)

    #- Parse multiple options for obsconditions
    if obsconditions is None:
        log.warning('Assuming DARK obsconditions')
        obsconditions = reference_conditions['DARK']
    elif isinstance(obsconditions, str):
        #- DARK GRAY BRIGHT
        if obsconditions.upper() in reference_conditions:
            log.info('Using reference {} obsconditions'.format(obsconditions.upper()))
            obsconditions = reference_conditions[obsconditions.upper()]
        #- filename
        elif os.path.exists(obsconditions):
            log.info('Loading obsconditions from {}'.format(obsconditions.upper()))
            if obsconditions.endswith('.ecsv'):
                allobs = astropy.table.Table.read(obsconditions, format='ascii.ecsv')
            else:
                allobs = astropy.table.Table.read(obsconditions)

            #- trim down to just this exposure
            if (expid is not None) and 'EXPID' in allobs.colnames:
                obsconditions = allobs[allobs['EXPID'] == expid]
            else:
                raise ValueError('unable to select which exposure from obsconditions file')
        else:
            raise ValueError('bad obsconditions {}'.format(obsconditions))
    elif isinstance(obsconditions, (astropy.table.Table, np.ndarray)):
        #- trim down to just this exposure
        if (expid is not None) and ('EXPID' in obsconditions):
            obsconditions = allobs[allobs['EXPID'] == expid]
        else:
            raise ValueError('must provide expid when providing obsconditions as a Table')

    #- Validate obsconditions keys
    try:
        obskeys = set(obsconditions.dtype.names)
    except AttributeError:
        obskeys = set(obsconditions.keys())
    missing_keys = set(reference_conditions['DARK'].keys()) - obskeys
    if len(missing_keys) > 0:
        raise ValueError('obsconditions missing keys {}'.format(missing_keys))

    sim = simulate_spectra(wave, flux, meta=fibermeta, obsconditions=obsconditions)
    header = dict(FLAVOR='science')

    return sim, fibermap, meta

def fibermeta2fibermap(fibermeta):
    '''
    Convert a fiberassign + targeting metadata table into a fibermap Table

    A future refactor will standardize the column names of fiber assignment,
    target catalogs, and fibermaps, but in the meantime this is needed.
    '''
    from desitarget import desi_mask

    #- Copy column names in common
    fibermap = desispec.io.empty_fibermap(len(fibermeta))
    for c in ['FIBER', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET',
              'BRICKNAME']:
        fibermap[c] = fibermeta[c]

    #- MAG and FILTER; ignore warnings from negative flux
    #- these are deprecated anyway and will be replaced with FLUX_G, FLUX_R, etc.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ugrizy = 22.5 - 2.5 * np.log10(fibermeta['DECAM_FLUX'].data)
        wise = 22.5 - 2.5 * np.log10(fibermeta['WISE_FLUX'].data)

    fibermap['FILTER'][:, :5] = ['DECAM_G', 'DECAM_R', 'DECAM_Z', 'WISE_W1', 'WISE_W2']
    fibermap['MAG'][:, 0] = ugrizy[:, 1]
    fibermap['MAG'][:, 1] = ugrizy[:, 2]
    fibermap['MAG'][:, 2] = ugrizy[:, 4]
    fibermap['MAG'][:, 3] = wise[:, 0]
    fibermap['MAG'][:, 4] = wise[:, 1]

    #- set OBJTYPE
    #- TODO: what about MWS science targets that are also standard stars?
    stdmask = (desi_mask.STD_FSTAR | desi_mask.STD_WD | desi_mask.STD_BRIGHT)
    isSTD = (fibermeta['DESI_TARGET'] & stdmask) != 0
    isSKY = (fibermeta['DESI_TARGET'] & desi_mask.SKY) != 0
    isSCI = (~isSTD & ~isSKY)
    fibermap['OBJTYPE'][isSTD] = 'STD'
    fibermap['OBJTYPE'][isSKY] = 'SKY'
    fibermap['OBJTYPE'][isSCI] = 'SCIENCE'

    fibermap['LAMBDAREF'] = 5400.0
    fibermap['RA_TARGET'] = fibermeta['RA']
    fibermap['DEC_TARGET'] = fibermeta['DEC']
    fibermap['RA_OBS']   = fibermeta['RA']
    fibermap['DEC_OBS']  = fibermeta['DEC']
    fibermap['X_TARGET'] = fibermeta['XFOCAL_DESIGN']
    fibermap['Y_TARGET'] = fibermeta['YFOCAL_DESIGN']
    fibermap['X_FVCOBS'] = fibermeta['XFOCAL_DESIGN']
    fibermap['Y_FVCOBS'] = fibermeta['YFOCAL_DESIGN']

    #- TODO: POSITIONER -> LOCATION
    #- TODO: TARGETCAT (how should we propagate this info into here?)
    #- TODO: NaNs in fibermap for unassigned positioners targets

    return fibermap

#-------------------------------------------------------------------------
#- specsim related routines

def simulate_spectra(wave, flux, meta=None, obsconditions=None, galsim=False,
    dwave_out=None):
    '''
    Simulates an exposure without reading/writing data files

    Args:
        wave (array): 1D wavelengths in Angstroms
        flux (array): 2D[nspec,nwave] flux in erg/s/cm2/Angstrom

    Optional:
        meta: table from fiberassign tiles file; uses X/YFOCAL_DESIGN
            TODO: document other columns for get_source_type
        obsconditions: (dict-like) observation metadata including
            SEEING (arcsec), EXPTIME (sec), AIRMASS,
            MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)

    TODO:
        galsim

    Returns a specsim.simulator.Simulator object
    '''
    import specsim.simulator
    import specsim.config
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from desiutil.log import get_logger
    log = get_logger('DEBUG')

    nspec, nwave = flux.shape

    #- Convert to unit-ful quantities for specsim
    if not isinstance(flux, u.Quantity):
        fluxunits = u.erg / (u.Angstrom * u.s * u.cm**2)
        flux = flux * fluxunits

    if not isinstance(wave, u.Quantity):
        wave = wave * u.Angstrom

    log.debug('loading specsim desi config')
    config = _specsim_config_for_wave(wave.to('Angstrom').value, dwave_out=dwave_out)

    #- Create simulator
    log.debug('creating specsim desi simulator')
    desi = specsim.simulator.Simulator(config, num_fibers=nspec)

    if obsconditions is None:
        log.warning('Assuming DARK conditions')
        obsconditions = reference_conditions['DARK']
    elif isinstance(obsconditions, str):
        obsconditions = reference_conditions[obsconditions.upper()]

    desi.atmosphere.seeing_fwhm_ref = obsconditions['SEEING'] * u.arcsec
    desi.observation.exposure_time = obsconditions['EXPTIME'] * u.s
    desi.atmosphere.airmass = obsconditions['AIRMASS']
    desi.atmosphere.moon.moon_phase = np.arccos(2*obsconditions['MOONFRAC']-1)/np.pi
    desi.atmosphere.moon.moon_zenith = (90 - obsconditions['MOONALT']) * u.deg
    desi.atmosphere.moon.separation_angle = obsconditions['MOONSEP'] * u.deg

    try:
        desi.observation.exposure_start = astropy.time.Time(obsconditions['MJD'], format='mjd')
        log.info('exposure_start {}'.format(desi.observation.exposure_start.utc.isot))
    except KeyError:
        log.info('MJD not in obsconditions, using DATE-OBS {}'.format(desi.observation.exposure_start.utc.isot))

    for obskey in reference_conditions['DARK'].keys():
        obsval = obsconditions[obskey]
        log.debug('obsconditions {} = {}'.format(obskey, obsval))

    #- Set fiber locations from meta Table or default fiberpos
    fiberpos = desimodel.io.load_fiberpos()
    if meta is not None and len(fiberpos) != len(meta):
        ii = np.in1d(fiberpos['FIBER'], meta['FIBER'])
        fiberpos = fiberpos[ii]

    if meta is None:
        meta = astropy.table.Table()
        meta['X'] = fiberpos['X'][0:nspec]
        meta['Y'] = fiberpos['Y'][0:nspec]
        meta['FIBER'] = fiberpos['FIBER'][0:nspec]

    #- Extract fiber locations from meta Table -> xy[nspec,2]
    assert np.all(meta['FIBER'] == fiberpos['FIBER'][0:nspec])
    if 'XFOCAL_DESIGN' in meta.dtype.names:
        xy = np.vstack([meta['XFOCAL_DESIGN'], meta['YFOCAL_DESIGN']]).T * u.mm
    elif 'X' in meta.dtype.names:
        xy = np.vstack([meta['X'], meta['Y']]).T * u.mm
    else:
        xy = np.vstack([fiberpos['X'], fiberpos['Y']]).T * u.mm

    if 'TARGETID' in meta.dtype.names:
        unassigned = (meta['TARGETID'] == -1)
        if np.any(unassigned):
            #- see https://github.com/astropy/astropy/issues/5961
            #- for the units -> array -> units trick
            xy[unassigned,0] = np.asarray(fiberpos['X'][unassigned], dtype=xy.dtype) * u.mm
            xy[unassigned,1] = np.asarray(fiberpos['Y'][unassigned], dtype=xy.dtype) * u.mm

    #- Determine source types
    #- TODO: source shapes + galsim instead of fixed types + fiberloss table
    source_types = get_source_types(meta)

    log.debug('running simulation')
    desi.instrument.fiberloss_method = 'table'
    desi.simulate(source_fluxes=flux, focal_positions=xy, source_types=source_types)
    return desi

def _specsim_config_for_wave(wave, dwave_out=None):
    '''
    Generate specsim config object for a given wavelength grid

    Args:
        wave: array of linearly spaced wavelengths in Angstroms

    Returns:
        specsim Configuration object with wavelength parameters set to match
        this input wavelength grid
    '''
    import specsim.config

    dwave = round(np.mean(np.diff(wave)), 3)
    assert np.allclose(dwave, np.diff(wave), rtol=1e-6, atol=1e-3)

    config = specsim.config.load_config('desi')
    config.wavelength_grid.min = wave[0]
    config.wavelength_grid.max = wave[-1] + dwave/2.0
    config.wavelength_grid.step = dwave

    if dwave_out is None:
        dwave_out = 1.0

    config.instrument.cameras.b.constants.output_pixel_size = "{:.3f} Angstrom".format(dwave_out)
    config.instrument.cameras.r.constants.output_pixel_size = "{:.3f} Angstrom".format(dwave_out)
    config.instrument.cameras.z.constants.output_pixel_size = "{:.3f} Angstrom".format(dwave_out)

    config.update()
    return config

def get_source_types(fibermap):
    '''
    Return a list of specsim source types based upon fibermap['DESI_TARGET']

    Args:
        fibermap: fibermap Table including DESI_TARGET column

    Returns array of source_types 'sky', 'elg', 'lrg', 'qso', 'star'

    Unassigned fibers fibermap['TARGETID'] == -1 will be treated as 'sky'

    If fibermap.meta['FLAVOR'] = 'arc' or 'flat', returned source types will
    match that flavor, though specsim doesn't use those as source_types

    TODO: specsim/desimodel doesn't have a fiber input loss model for BGS yet,
    so BGS targets get source_type = 'lrg' (!)
    '''
    from desiutil.log import get_logger
    log = get_logger('DEBUG')
    if 'DESI_TARGET' not in fibermap.dtype.names:
        log.warn("DESI_TARGET not in fibermap table; using source_type='star' for everything")
        return np.array(['star',] * len(fibermap))

    source_type = np.zeros(len(fibermap), dtype='U4')
    assert np.all(source_type == '')

    if 'FLAVOR' in fibermap.meta:
        if fibermap.meta['FLAVOR'].lower() == 'arc':
            source_type[:] = 'ARC'
            return source_type
        elif fibermap.meta['FLAVOR'].lower() == 'flat':
            source_type[:] = 'FLAT'
            return source_type

    tm = desitarget.desi_mask
    if 'TARGETID' in fibermap.dtype.names:
        unassigned = fibermap['TARGETID'] == -1
        source_type[unassigned] = 'sky'

    source_type[(fibermap['DESI_TARGET'] & tm.SKY) != 0] = 'sky'
    source_type[(fibermap['DESI_TARGET'] & tm.ELG) != 0] = 'elg'
    source_type[(fibermap['DESI_TARGET'] & tm.LRG) != 0] = 'lrg'
    source_type[(fibermap['DESI_TARGET'] & tm.QSO) != 0] = 'qso'
    starmask = tm.STD_FSTAR | tm.STD_WD | tm.STD_BRIGHT | tm.MWS_ANY
    source_type[(fibermap['DESI_TARGET'] & starmask) != 0] = 'star'

    #- FIXME: no BGS default in specsim yet
    isBGS = (fibermap['DESI_TARGET'] & tm.BGS_ANY) != 0
    if np.any(isBGS):
        nBGS = np.count_nonzero(isBGS)
        log.warning("Setting {} BGS targets to have source_type='lrg'".format(nBGS))
        source_type[isBGS] = 'lrg'

    assert not np.any(source_type == '')

    for name in sorted(np.unique(source_type)):
        n = np.count_nonzero(source_type == name)
        log.debug('{} {} targets'.format(name, n))

    return source_type

#-------------------------------------------------------------------------
#- I/O related routines

def read_fiberassign(tilefile_or_id, indir=None):
    '''
    Returns fiberassignment table for tileid

    Args:
        tilefile_or_id (int or str): tileid (int) or full path to tile file (str)

    Returns:
        fiberassignment Table from HDU 1
    '''
    #- tileid is full path to file instead of int ID or just filename
    if isinstance(tilefile_or_id, str) and os.path.exists(tilefile_or_id):
        return astropy.table.Table.read(tilefile_or_id)

    if indir is None:
        indir = os.path.join(os.environ['DESI_TARGETS'], 'fiberassign')

    if isinstance(tilefile_or_id, (int, np.int32, np.int64)):
        tilefile = os.path.join(indir, 'tile_{:05d}.fits'.format(tilefile_or_id))
    else:
        tilefile = os.path.join(indir, tilefile_or_id)

    return astropy.table.Table.read(tilefile, 'FIBER_ASSIGNMENTS')

#-------------------------------------------------------------------------
#- TODO: fiber area calculation should be in desimodel.focalplane

def fiber_area_arcsec2(x, y):
    '''
    Returns area of fibers at (x,y) in arcsec^2
    '''
    params = desimodel.io.load_desiparams()
    fiber_dia = params['fibers']['diameter_um']
    x = np.asarray(x)
    y = np.asarray(y)
    r = np.sqrt(x**2 + y**2)

    #- Platescales in um/arcsec
    ps = desimodel.io.load_platescale()
    radial_scale = np.interp(r, ps['radius'], ps['radial_platescale'])
    az_scale = np.interp(r, ps['radius'], ps['az_platescale'])

    #- radial and azimuthal fiber radii in arcsec
    rr  = 0.5 * fiber_dia / radial_scale
    raz = 0.5 * fiber_dia / az_scale
    fiber_area = (np.pi * rr * raz)
    return fiber_area

#-------------------------------------------------------------------------
#- MOVE THESE TO desitarget.mocks.io (?)
#-------------------------------------------------------------------------

def get_mock_spectra(fiberassign, mockdir=None):
    '''
    Args:
        fiberassign: table loaded from fiberassign tile file

    Returns (flux, wave, meta) tuple
    '''
    nspec = len(fiberassign)
    flux = None
    meta = None
    wave = None

    issky = (fiberassign['DESI_TARGET'] & desitarget.desi_mask.SKY) != 0
    skyids = fiberassign['TARGETID'][issky]

    for truthfile, targetids in zip(*targets2truthfiles(fiberassign, basedir=mockdir)):

        #- Sky fibers aren't in the truth files
        ok = ~np.in1d(targetids, skyids)

        tmpflux, tmpwave, tmpmeta = read_mock_spectra(truthfile, targetids[ok])

        if flux is None:
            nwave = tmpflux.shape[1]
            flux = np.zeros((nspec, nwave))
            meta = np.zeros(nspec, dtype=tmpmeta.dtype)
            meta['TARGETID'] = -1
            wave = tmpwave.astype('f8')

        ii = np.in1d(fiberassign['TARGETID'], tmpmeta['TARGETID'])
        flux[ii] = tmpflux
        meta[ii] = tmpmeta
        assert np.all(wave == tmpwave)

    #- Set meta['TARGETID'] for sky fibers
    #- TODO: other things to set?
    meta['TARGETID'][issky] = skyids

    set(fiberassign['TARGETID']) == set(meta['TARGETID'])

    assert np.all(fiberassign['TARGETID'] == meta['TARGETID'])

    return flux, wave, astropy.table.Table(meta)

def read_mock_spectra(truthfile, targetids, mockdir=None):
    '''
    Reads mock spectra from a truth file

    Args:
        truthfile (str): full path to a mocks spectra_truth*.fits file
        targetids (array-like): targetids to load from that file
        mockdir: ???

    Returns (flux, wave, truth) tuples:
        flux[nspec, nwave]: flux in 1e-17 erg/s/cm2/Angstrom
        wave[nwave]: wavelengths in Angstroms
        truth[nspec]: metadata truth table
    '''
    with fits.open(truthfile, memmap=False) as fx:
        truth = fx['TRUTH'].data
        wave = fx['WAVE'].data
        flux = fx['FLUX'].data

    #- Trim to just the spectra for these targetids
    ii = np.in1d(truth['TARGETID'], targetids)
    if np.count_nonzero(ii) != len(targetids):
        jj = ~np.in1d(targetids, truth['TARGETID'])
        missingids = targetids[jj]
        raise ValueError('Targets missing from {}: {}'.format(truthfile, missingids))

    flux = flux[ii]
    truth = truth[ii]

    assert set(targetids) == set(truth['TARGETID'])

    #- sort truth to match order of input targetids
    #- Sort tiles to match order of tileids
    i = np.argsort(targetids)
    j = np.argsort(truth['TARGETID'])
    k = np.argsort(i)

    truth = truth[j[k]]
    flux = flux[j[k]]
    assert np.all(truth['TARGETID'] == targetids)

    wave = desispec.io.util.native_endian(wave).astype(np.float64)
    flux = desispec.io.util.native_endian(flux).astype(np.float64)

    return flux, wave, truth

def targets2truthfiles_bricks(targets, basedir):
    '''
    Return list of mock truth files that contain these targets

    Args:
        targets: target Table from targeting or fiber assignment
        basedir: base directory under which files are found

    Returns (truthfiles, targetids):
        truthfiles: list of truth filenames
        targetids: list of lists of targetids in each truthfile

    i.e. targetids[i] is the list of targetids from targets['TARGETID'] that
        are in truthfiles[i]

    Note: uses brickname grouping {basedir}/{subdir}/truth-{brickname}.fits
        where subdir = brickname[0:3]
    '''
    #- TODO: what should be done with assignments without targets?
    targets = targets[targets['TARGETID'] != -1]

    truthfiles = list()
    targetids = list()
    for brickname in sorted(set(targets['BRICKNAME'])):
        filename = 'truth-{}.fits'.format(brickname)
        subdir = brickname[0:3]
        truthfiles.append(os.path.join(basedir, subdir, filename))
        ii = (targets['BRICKNAME'] == brickname)
        targetids.append(targets['TARGETID'][ii])

    return truthfiles, targetids

def targets2truthfiles_healpix(targets, basedir, nside=64):
    '''
    Return list of mock truth files that contain these targets

    Args:
        targets: target Table from targeting or fiber assignment
        basedir: base directory under which files are found

    Returns (truthfiles, targetids):
        truthfiles: list of truth filenames
        targetids: list of lists of targetids in each truthfile

    i.e. targetids[i] is the list of targetids from targets['TARGETID'] that
        are in truthfiles[i]

    Note: uses healpix grouping {basedir}/{subdir}/truth-{nside}-{ipix}.fits
        where subdir = ipix // (nside//2)
    '''
    import healpy
    import desitarget.mock.io as mockio
    assert nside >= 2

    #- TODO: what should be done with assignments without targets?
    targets = targets[targets['TARGETID'] != -1]

    theta = np.radians(90-targets['DEC'])
    phi = np.radians(targets['RA'])
    pixels = healpy.ang2pix(nside, theta, phi, nest=True)

    truthfiles = list()
    targetids = list()
    for ipix in sorted(np.unique(pixels)):
        filename = mockio.findfile('truth', nside, ipix, basedir=basedir)
        truthfiles.append(filename)
        ii = (pixels == ipix)
        targetids.append(targets['TARGETID'][ii])

    return truthfiles, targetids

#- Use brick naming scheme for now
targets2truthfiles = targets2truthfiles_healpix

