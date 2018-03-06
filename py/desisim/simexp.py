from __future__ import absolute_import, division, print_function

import sys
import os.path
import warnings
import datetime, time

import numpy as np

import astropy.table
import astropy.time
from astropy.io import fits
import fitsio

import desitarget
import desitarget.targetmask
import desispec.io
import desispec.io.util
import desimodel.io
from desimodel.focalplane import fiber_area_arcsec2
import desiutil.depend
import desispec.interpolation
import desisim.io
import desisim.specsim

#- Reference observing conditions for each of dark, gray, bright
reference_conditions = dict(DARK=dict(), GRAY=dict(), BRIGHT=dict())
reference_conditions['DARK']['SEEING']  = 1.1
reference_conditions['DARK']['EXPTIME'] = 1000
reference_conditions['DARK']['AIRMASS'] = 1.0
reference_conditions['DARK']['MOONFRAC'] = 0.0
reference_conditions['DARK']['MOONALT'] = -60
reference_conditions['DARK']['MOONSEP'] = 180

reference_conditions['GRAY']['SEEING']  = 1.1
reference_conditions['GRAY']['EXPTIME'] = 1000
reference_conditions['GRAY']['AIRMASS'] = 1.0
reference_conditions['GRAY']['MOONFRAC'] = 0.1
reference_conditions['GRAY']['MOONALT']  = 10
reference_conditions['GRAY']['MOONSEP'] = 60

reference_conditions['BRIGHT']['SEEING']  = 1.1
reference_conditions['BRIGHT']['EXPTIME'] = 300
reference_conditions['BRIGHT']['AIRMASS'] = 1.0
reference_conditions['BRIGHT']['MOONFRAC'] = 0.7
reference_conditions['BRIGHT']['MOONALT']  = 60
reference_conditions['BRIGHT']['MOONSEP'] = 50

for objtype in ('LRG', 'QSO', 'ELG'):
    reference_conditions[objtype] = reference_conditions['DARK']
for objtype in ('MWS', 'BGS'):
    reference_conditions[objtype] = reference_conditions['BRIGHT']

def simarc(arcdata, nspec=5000, nonuniform=False, testslit=False):
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
        * add psfconvolve option
    '''
    wave = arcdata['VACUUM_WAVE']
    phot = arcdata['ELECTRONS']

    if testslit:
        fibermap = astropy.table.Table(testslit_fibermap()[0:nspec])
    else:
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

def simflat(flatfile, nspec=5000, nonuniform=False, exptime=10, testslit=False,
            psfconvolve=True, specsim_config_file="desi"):
    '''
    Simulates a flat lamp calibration exposure

    Args:
        flatfile: filename with flat lamp spectrum data

    Options:
        nspec: (int) number of spectra to simulate
        nonuniform: (bool) include calibration screen non-uniformity
        exptime: (float) exposure time in seconds
        psfconvolve: (bool) passed to simspec.simulator.Simulator camera_output.
            if True, convolve with PSF and include per-camera outputs
        specsim_config_file: (str) path to DESI instrument config file.
            default is desi config in specsim package.

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
    try:
        params = desimodel.io.load_desiparams()
        wavemin = params['ccd']['b']['wavemin']
        wavemax = params['ccd']['z']['wavemax']
    except KeyError:
        wavemin = desimodel.io.load_throughput('b').wavemin
        wavemax = desimodel.io.load_throughput('z').wavemax

    ii = (wavemin <= wave) & (wave <= wavemax)
    wave = wave[ii]
    sbflux = sbflux[ii]

    #- Downsample to 0.2A grid to not blow up memory
    ww = np.arange(wave[0], wave[-1]+0.1, 0.2)
    sbflux = desispec.interpolation.resample_flux(ww, wave, sbflux)
    wave = ww

    if testslit:
        fibermap = astropy.table.Table(testslit_fibermap()[0:nspec])
    else:
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
    config = _specsim_config_for_wave(wave,specsim_config_file=specsim_config_file)
    log.debug('Creating specsim simulator for {} spectra'.format(nspec))
    # sim = specsim.simulator.Simulator(config, num_fibers=nspec)
    sim = desisim.specsim.get_simulator(config, num_fibers=nspec,
        camera_output=psfconvolve)
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

def simscience(targets, fiberassign, obsconditions='DARK', expid=None,
    nspec=None, psfconvolve=True):
    '''
    Simulates a new DESI exposure from surveysim+fiberassign+mock spectra

    Args:
        targets: tuple of (flux[nspec,nwave], wave[nwave], meta[nspec])
        fiberassign: fiber assignments table

    Options:
        obsconditions: observation metadata as
            str: DARK (default) or GRAY or BRIGHT
            dict or row of Table with keys
                SEEING (arcsec), EXPTIME (sec), AIRMASS,
                MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
            Table including EXPID for subselection of which row to use
            filename with obsconditions Table; expid must also be set
        expid: (int) exposure ID
        nspec: (int) number of spectra to simulate
        psfconvolve: (bool) passed to simspec.simulator.Simulator camera_output.
            if True, convolve with PSF and include per-camera outputs

    Returns sim, fibermap, meta
        sim: specsim.simulate.Simulator object
        fibermap: Table
        meta: target metadata truth table

    See obs.new_exposure() for function to generate new random exposure,
    independent from surveysim, fiberassignment, and pre-generated mocks.
    '''
    from desiutil.log import get_logger
    log = get_logger()

    flux, wave, meta = targets

    if nspec is not None:
        fiberassign = fiberassign[0:nspec]

    assert np.all(fiberassign['TARGETID'] == meta['TARGETID'])
    fibermap = fibermeta2fibermap(fiberassign, meta)

    #- Parse multiple options for obsconditions
    if isinstance(obsconditions, str):
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

    sim = simulate_spectra(wave, flux, fibermap=fibermap,
        obsconditions=obsconditions, psfconvolve=psfconvolve)

    return sim, fibermap

def fibermeta2fibermap(fiberassign, meta):
    '''
    Convert a fiberassign + targeting metadata table into a fibermap Table

    A future refactor will standardize the column names of fiber assignment,
    target catalogs, and fibermaps, but in the meantime this is needed.
    '''
    from desitarget.targetmask import desi_mask

    #- Copy column names in common
    fibermap = desispec.io.empty_fibermap(len(fiberassign))
    for c in ['FIBER', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET',
              'BRICKNAME']:
        fibermap[c] = fiberassign[c]

    #- MAG and FILTER; ignore warnings from negative flux
    #- these are deprecated anyway and will be replaced with FLUX_G, FLUX_R,
    #- etc. in the fibermap as well
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        fibermap['FILTER'][:, :5] = \
            ['DECAM_G', 'DECAM_R', 'DECAM_Z', 'WISE_W1', 'WISE_W2']
        fibermap['MAG'][:, 0] = 22.5 - 2.5 * np.log10(meta['FLUX_G'].data)
        fibermap['MAG'][:, 1] = 22.5 - 2.5 * np.log10(meta['FLUX_R'].data)
        fibermap['MAG'][:, 2] = 22.5 - 2.5 * np.log10(meta['FLUX_Z'].data)
        fibermap['MAG'][:, 3] = 22.5 - 2.5 * np.log10(meta['FLUX_W1'].data)
        fibermap['MAG'][:, 4] = 22.5 - 2.5 * np.log10(meta['FLUX_W2'].data)

    #- set OBJTYPE
    #- TODO: what about MWS science targets that are also standard stars?
    stdmask = (desi_mask.STD_FSTAR | desi_mask.STD_WD | desi_mask.STD_BRIGHT)
    isSTD = (fiberassign['DESI_TARGET'] & stdmask) != 0
    isSKY = (fiberassign['DESI_TARGET'] & desi_mask.SKY) != 0
    isSCI = (~isSTD & ~isSKY)
    fibermap['OBJTYPE'][isSTD] = 'STD'
    fibermap['OBJTYPE'][isSKY] = 'SKY'
    fibermap['OBJTYPE'][isSCI] = 'SCIENCE'

    fibermap['LAMBDAREF'] = 5400.0
    fibermap['RA_TARGET'] = fiberassign['RA']
    fibermap['DEC_TARGET'] = fiberassign['DEC']
    fibermap['RA_OBS']   = fiberassign['RA']
    fibermap['DEC_OBS']  = fiberassign['DEC']
    fibermap['X_TARGET'] = fiberassign['XFOCAL_DESIGN']
    fibermap['Y_TARGET'] = fiberassign['YFOCAL_DESIGN']
    fibermap['X_FVCOBS'] = fiberassign['XFOCAL_DESIGN']
    fibermap['Y_FVCOBS'] = fiberassign['YFOCAL_DESIGN']

    #- TODO: POSITIONER -> LOCATION
    #- TODO: TARGETCAT (how should we propagate this info into here?)
    #- TODO: NaNs in fibermap for unassigned positioners targets

    return fibermap

#-------------------------------------------------------------------------
#- specsim related routines

def simulate_spectra(wave, flux, fibermap=None, obsconditions=None, redshift=None,
                     dwave_out=None, seed=None, psfconvolve=True,
                     specsim_config_file = "desi"):
    '''
    Simulates an exposure without reading/writing data files

    Args:
        wave (array): 1D wavelengths in Angstroms
        flux (array): 2D[nspec,nwave] flux in 1e-17 erg/s/cm2/Angstrom
            or astropy Quantity with flux units

    Optional:
        fibermap: table from fiberassign or fibermap; uses X/YFOCAL_DESIGN, TARGETID, DESI_TARGET
        obsconditions: (dict-like) observation metadata including
            SEEING (arcsec), EXPTIME (sec), AIRMASS,
            MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
        redshift : list/array with each index being the redshifts for that target
        seed: (int) random seed
        psfconvolve: (bool) passed to simspec.simulator.Simulator camera_output.
            if True, convolve with PSF and include per-camera outputs
        specsim_config_file: (str) path to DESI instrument config file.
            default is desi config in specsim package.
    TODO: galsim support

    Returns a specsim.simulator.Simulator object
    '''
    import specsim.simulator
    import specsim.config
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from desiutil.log import get_logger
    log = get_logger('DEBUG')

    # Input cosmology to calculate the angular diameter distance of the galaxy's redshift
    from astropy.cosmology import FlatLambdaCDM
    LCDM = FlatLambdaCDM(H0=70, Om0=0.3)
    ang_diam_dist = LCDM.angular_diameter_distance
    
    random_state = np.random.RandomState(seed)

    nspec, nwave = flux.shape

    #- Convert to unit-ful quantities for specsim
    if not isinstance(flux, u.Quantity):
        fluxunits = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm**2)
        flux = flux * fluxunits

    if not isinstance(wave, u.Quantity):
        wave = wave * u.Angstrom

    log.debug('loading specsim desi config {}'.format(specsim_config_file))
    config = _specsim_config_for_wave(wave.to('Angstrom').value, dwave_out=dwave_out, specsim_config_file=specsim_config_file)

    #- Create simulator
    log.debug('creating specsim desi simulator')
    # desi = specsim.simulator.Simulator(config, num_fibers=nspec)
    desi = desisim.specsim.get_simulator(config, num_fibers=nspec,
        camera_output=psfconvolve)

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
    if fibermap is not None and len(fiberpos) != len(fibermap):
        ii = np.in1d(fiberpos['FIBER'], fibermap['FIBER'])
        fiberpos = fiberpos[ii]

    if fibermap is None:
        fibermap = astropy.table.Table()
        fibermap['X'] = fiberpos['X'][0:nspec]
        fibermap['Y'] = fiberpos['Y'][0:nspec]
        fibermap['FIBER'] = fiberpos['FIBER'][0:nspec]
        fibermap['LOCATION'] = fiberpos['LOCATION'][0:nspec]

    #- Extract fiber locations from meta Table -> xy[nspec,2]
    assert np.all(fibermap['FIBER'] == fiberpos['FIBER'][0:nspec])
    if 'XFOCAL_DESIGN' in fibermap.dtype.names:
        xy = np.vstack([fibermap['XFOCAL_DESIGN'], fibermap['YFOCAL_DESIGN']]).T * u.mm
    elif 'X' in fibermap.dtype.names:
        xy = np.vstack([fibermap['X'], fibermap['Y']]).T * u.mm
    else:
        xy = np.vstack([fiberpos['X'], fiberpos['Y']]).T * u.mm

    if 'TARGETID' in fibermap.dtype.names:
        unassigned = (fibermap['TARGETID'] == -1)
        if np.any(unassigned):
            #- see https://github.com/astropy/astropy/issues/5961
            #- for the units -> array -> units trick
            xy[unassigned,0] = np.asarray(fiberpos['X'][unassigned], dtype=xy.dtype) * u.mm
            xy[unassigned,1] = np.asarray(fiberpos['Y'][unassigned], dtype=xy.dtype) * u.mm
        
    #- Determine source types
    #- TODO: source shapes + galsim instead of fixed types + fiberloss table
    source_types = get_source_types(fibermap)
    # source types are sky elg lrg qso bgs star , they 
    # are only used in specsim.fiberloss for the desi.instrument.fiberloss_method="table" method
    
    desi.instrument.fiberloss_method = 'fastsim'

    log.debug('running simulation with {} fiber loss method'.format(desi.instrument.fiberloss_method))
    
    unique_source_types = set(source_types)
    comment_line="source types:"
    for u in set(source_types) :
        comment_line+=" {} {}".format(np.sum(source_types==u),u)
    log.debug(comment_line)
    
    source_fraction=None
    source_half_light_radius=None
    source_minor_major_axis_ratio=None
    source_position_angle=None
    
    if desi.instrument.fiberloss_method == 'fastsim' or desi.instrument.fiberloss_method == 'galsim' :
        # the following parameters are used only with fastsim and galsim methods
        
        elgs=(source_types=="elg")
        lrgs=(source_types=="lrg")
        bgss=(source_types=="bgs")

        if np.sum(lrgs)>0 or np.sum(elgs)>0:
            log.warning("the half light radii are fixed here for LRGs and ELGs (and not magnitude or redshift dependent)")
        if np.sum(bgss)>0 and redshift is None:
            log.warning("the half light radii are fixed here for BGS (as redshifts weren't supplied)")
            
        # BGS parameters based on SDSS main sample, in g-band
        # see analysis from J. Moustakas in
        # https://github.com/desihub/desitarget/blob/master/doc/nb/bgs-morphology-properties.ipynb 
        # B/T (bulge-to-total ratio): 0.48 (0.36 - 0.59).
        # Bulge Sersic n: 2.27 (1.12 - 3.60).
        # log10 (Bulge Half-light radius): 0.11 (-0.077 - 0.307) arcsec
        # log10 (Disk Half-light radius): 0.67 (0.54 - 0.82) arcsec
        # This gives
        # bulge_fraction = 0.48
        # disk_fraction  = 0.52
        # bulge_half_light_radius = 1.3 arcsec
        # disk_half_light_radius  = 4.7 arcsec
        # note we use De Vaucouleurs' law , which correspond to a Sersic index n=4
        
        # source_fraction[:,0] is DISK profile (exponential) fraction
        # source_fraction[:,1] is BULGE profile (devaucouleurs) fraction
        # 1 - np.sum(source_fraction,axis=1) is POINT source profile fraction
        # see specsim.GalsimFiberlossCalculator.create_source routine
        source_fraction=np.zeros((nspec,2)) 
        source_fraction[elgs,0]=1.   # ELG are disk only
        source_fraction[lrgs,1]=1.   # LRG are bulge only
        source_fraction[bgss,0]=0.52 # disk comp in BGS
        source_fraction[bgss,1]=0.48 # bulge comp in BGS       

        # source_half_light_radius[:,0] is the half light radius in arcsec for the DISK profile
        # source_half_light_radius[:,1] is the half light radius in arcsec for the BULGE profile        
        # see specsim.GalsimFiberlossCalculator.create_source routine
        source_half_light_radius=np.zeros((nspec,2))
        source_half_light_radius[elgs,0]=0.45 # ELG are disk only, arcsec
        source_half_light_radius[lrgs,1]=1.   # LRG are bulge only, arcsec

        # 4.7 is angular size of z=0.1 disk, and 1.3 is angular size of z=0.1 bulge
        bgs_disk_z01 = 4.7  # in arcsec
        bgs_bulge_z01 = 1.3 # in arcsec
        
        # Convert to angular size of the objects in this sample with given redshifts
        if redshift is None:
            angscales = np.ones(np.sum(bgss))
        else:
            bgs_redshifts = redshift[bgss]
            # Avoid infinities
            if np.any(bgs_redshifts <= 0.):
                bgs_redshifts[bgs_redshifts <= 0.] = 0.0001
            angscales = ( ang_diam_dist(0.1) / ang_diam_dist(bgs_redshifts) ).value
        source_half_light_radius[bgss,0]= bgs_disk_z01 * angscales # disk comp in BGS, arcsec
        source_half_light_radius[bgss,1]= bgs_bulge_z01 * angscales  # bulge comp in BGS, arcsec
        
        if desi.instrument.fiberloss_method == 'galsim' :
            # the following parameters are used only with galsim method
        
            # source_minor_major_axis_ratio[:,0] is the axis ratio for the DISK profile
            # source_minor_major_axis_ratio[:,1] is the axis ratio for the BULGE profile
            # see specsim.GalsimFiberlossCalculator.create_source routine
            source_minor_major_axis_ratio=np.zeros((nspec,2)) 
            source_minor_major_axis_ratio[elgs,0]=0.7 
            source_minor_major_axis_ratio[lrgs,1]=0.7
            source_minor_major_axis_ratio[bgss,1]=0.7
            
            # the source position angle is in degrees
            # see specsim.GalsimFiberlossCalculator.create_source routine
            source_position_angle = np.zeros((nspec,2))
            random_angles = 360.*random_state.uniform(size=nspec)
            source_position_angle[elgs,0]=random_angles[elgs]
            source_position_angle[lrgs,1]=random_angles[lrgs]
            source_position_angle[bgss,1]=random_angles[bgss]

    #- Work around randomness in specsim quickfiberloss calculations
    #- while not impacting global random state.
    #- See https://github.com/desihub/specsim/issues/83
    randstate = np.random.get_state()
    np.random.seed(seed)
    desi.simulate(source_fluxes=flux, focal_positions=xy, source_types=source_types,
                  source_fraction=source_fraction,
                  source_half_light_radius=source_half_light_radius,
                  source_minor_major_axis_ratio=source_minor_major_axis_ratio,
                  source_position_angle=source_position_angle)
    np.random.set_state(randstate)

    return desi

def _specsim_config_for_wave(wave, dwave_out=None, specsim_config_file = "desi"):
    '''
    Generate specsim config object for a given wavelength grid

    Args:
        wave: array of linearly spaced wavelengths in Angstroms
    
    Options:
        specsim_config_file: (str) path to DESI instrument config file.
            default is desi config in specsim package.
    Returns:
        specsim Configuration object with wavelength parameters set to match
        this input wavelength grid
    '''
    import specsim.config

    dwave = round(np.mean(np.diff(wave)), 3)
    assert np.allclose(dwave, np.diff(wave), rtol=1e-6, atol=1e-3)

    config = specsim.config.load_config(specsim_config_file)
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
        log.warning("DESI_TARGET not in fibermap table; using source_type='star' for everything")
        return np.array(['star',] * len(fibermap))

    source_type = np.zeros(len(fibermap), dtype='U4')
    assert np.all(source_type == '')

    tm = desitarget.targetmask.desi_mask
    if 'TARGETID' in fibermap.dtype.names:
        unassigned = fibermap['TARGETID'] == -1
        source_type[unassigned] = 'sky'

    source_type[(fibermap['OBJTYPE'] == 'FLAT')] = 'FLAT'
    source_type[(fibermap['OBJTYPE'] == 'ARC')] = 'ARC'
    source_type[(fibermap['DESI_TARGET'] & tm.SKY) != 0] = 'sky'
    source_type[(fibermap['DESI_TARGET'] & tm.ELG) != 0] = 'elg'
    source_type[(fibermap['DESI_TARGET'] & tm.LRG) != 0] = 'lrg'
    source_type[(fibermap['DESI_TARGET'] & tm.QSO) != 0] = 'qso'
    source_type[(fibermap['DESI_TARGET'] & tm.BGS_ANY) != 0] = 'bgs'
    starmask = tm.STD_FSTAR | tm.STD_WD | tm.STD_BRIGHT | tm.MWS_ANY
    source_type[(fibermap['DESI_TARGET'] & starmask) != 0] = 'star'

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
#- Move this to desispec.io?

def testslit_fibermap():
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

    issky = (fiberassign['DESI_TARGET'] & desitarget.targetmask.desi_mask.SKY) != 0
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
    if len(targetids) != len(np.unique(targetids)):
        from desiutil.log import get_logger
        log = get_logger()
        log.error("Requested TARGETIDs for {} are not unique".format(
            os.path.basename(truthfile)))

    #- astropy.io.fits doesn't return a real ndarray, causing problems
    #- with the reordering downstream so use fitsio instead
    # with fits.open(truthfile, memmap=False) as fx:
    #     truth = fx['TRUTH'].data
    #     wave = fx['WAVE'].data
    #     flux = fx['FLUX'].data
    with fitsio.FITS(truthfile) as fx:
        truth = fx['TRUTH'].read()
        wave = fx['WAVE'].read()
        flux = fx['FLUX'].read()

    missing = np.in1d(targetids, truth['TARGETID'], invert=True)
    if np.any(missing):
        missingids = targetids[missing]
        raise ValueError('Targets missing from {}: {}'.format(truthfile, missingids))

    #- Trim to just the spectra for these targetids
    ii = np.in1d(truth['TARGETID'], targetids)
    flux = flux[ii]
    truth = truth[ii]

    assert set(targetids) == set(truth['TARGETID'])

    #- sort truth to match order of input targetids
    if len(targetids) == len(truth['TARGETID']):
        i = np.argsort(targetids)
        j = np.argsort(truth['TARGETID'])
        k = np.argsort(i)
        reordered_truth = truth[j[k]]
        reordered_flux = flux[j[k]]
    else:
        #- Slower, but works even with repeated TARGETIDs
        ii = np.argsort(truth['TARGETID'])
        sorted_truthids = truth['TARGETID'][ii]
        reordered_flux = np.empty(shape=(len(targetids), flux.shape[1]), dtype=flux.dtype)
        reordered_truth = np.empty(shape=(len(targetids),), dtype=truth.dtype)
        for j, tx in enumerate(targetids):
            k = np.searchsorted(sorted_truthids, tx)
            reordered_flux[j] = flux[ii[k]]
            reordered_truth[j] = truth[ii[k]]

    assert np.all(reordered_truth['TARGETID'] == targetids)

    wave = desispec.io.util.native_endian(wave).astype(np.float64)
    reordered_flux = desispec.io.util.native_endian(reordered_flux).astype(np.float64)

    return reordered_flux, wave, reordered_truth

def targets2truthfiles(targets, basedir, nside=64):
    '''
    Return list of mock truth files that contain these targets

    Args:
        targets: table with TARGETID column, e.g. from fiber assignment
        basedir: base directory under which files are found

    Returns (truthfiles, targetids):
        truthfiles: list of truth filenames
        targetids: list of lists of targetids in each truthfile

    i.e. targetids[i] is the list of targetids from targets['TARGETID'] that
        are in truthfiles[i]
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
        targetids.append(np.asarray(targets['TARGETID'][ii]))

    return truthfiles, targetids

