"""
desisim.scripts.quicktransients
===============================
"""

import os
import numpy as np
import healpy as hp
from datetime import datetime

from desisim.templates import BGS, ELG, LRG
from desisim.transients import transients

from desitarget.mock.mockmaker import BGSMaker, ELGMaker, LRGMaker
from desitarget.cuts import isBGS_colors, isELG_colors, isLRG_colors

from desisim.simexp import reference_conditions
from desisim.transients import transients
from desisim.scripts.quickspectra import sim_spectra

from desispec.io import read_spectra, write_spectra
from desispec.coaddition import coadd_cameras

from desiutil.log import get_logger, DEBUG

from astropy.table import Table, hstack, vstack

import argparse


def _set_wave(wavemin=None, wavemax=None, dw=0.8):
    """Set default wavelength grid for simulations.

    Parameters
    ----------
    wavemin : float or None
        Minimum wavelength, in Angstroms.
    wavemax : float or None
        Maximum wavelength, in Angstroms.
    dw : float
        Bin size.

    Returns
    -------
    wave : ndarray
        Grid of wavelength values.
    """
    from desimodel.io import load_throughput

    if wavemin is None:
        wavemin = load_throughput('b').wavemin - 10.0
    if wavemax is None:
        wavemax = load_throughput('z').wavemax + 10.0

    return np.arange(round(wavemin, 1), wavemax, dw)


def _get_healpixels_in_footprint(nside=64):
    """Obtain a list of HEALPix pixels in the DESI footprint.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (in form nside=2**k, k=[1,2,3,...]).

    Returns
    -------
    healpixels : ndarray
        List of HEALPix pixels within the DESI footprint.
    """
    from desimodel import footprint
    from desimodel.io import load_tiles

    # Load DESI tiles.
    tile_tab = load_tiles()

    npix = hp.nside2npix(nside)
    pix_ids = np.arange(npix)
    ra, dec = hp.pix2ang(nside, pix_ids, lonlat=True)

    # Get a list of pixel IDs inside the DESI footprint.
    in_desi = footprint.is_point_in_desi(tile_tab, ra, dec)
    healpixels = pix_ids[in_desi]

    return healpixels


def write_templates(filename, flux, wave, target, truth, objtruth):
    """Write galaxy templates to a FITS file.
    Parameters
    ----------
    filename : str
        Path to output file.
    flux : ndarray
        Array of flux data for template spectra.
    wave : ndarray
        Array of wavelengths.
    target : Table
        Target information.
    truth : Table
        Template simulation truth.
    objtruth : Table
        Object-specific truth data.
    """
    import astropy.units as u
    from astropy.io import fits

    hx = fits.HDUList()

    # Write the wavelength table.
    hdu_wave = fits.PrimaryHDU(wave)
    hdu_wave.header['EXTNAME'] = 'WAVE'
    hdu_wave.header['BUNIT'] = 'Angstrom'
    hdu_wave.header['AIRORVAC'] = ('vac', 'Vacuum wavelengths')
    hx.append(hdu_wave)

    # Write the flux table.
    fluxunits = 1e-17 * u.erg / (u.s * u.cm**2 * u.Angstrom)
    hdu_flux = fits.ImageHDU(flux)
    hdu_flux.header['EXTNAME'] = 'FLUX'
    hdu_flux.header['BUNIT'] = str(fluxunits)
    hx.append(hdu_flux)

    # Write targets table.
    hdu_targets = fits.table_to_hdu(target)
    hdu_targets.header['EXTNAME'] = 'TARGETS'
    hx.append(hdu_targets)

    # Write truth table.
    hdu_truth = fits.table_to_hdu(truth)
    hdu_truth.header['EXTNAME'] = 'TRUTH'
    hx.append(hdu_truth)

    # Write objtruth table.
    hdu_objtruth = fits.table_to_hdu(objtruth)
    hdu_objtruth.header['EXTNAME'] = 'OBJTRUTH'
    hx.append(hdu_objtruth)

    hx.writeto(filename, overwrite=True)


def parse(options=None):
    """Parse command line options.
    """
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Fast galaxy + transient simulator')

    # Set up observing conditions.
    cond = parser.add_argument_group('Observing conditions')
    for c, v in reference_conditions['DARK'].items():
        cond.add_argument('--{}'.format(c.lower()), type=float,
                          default=v)

    # Set up simulation.
    sims = parser.add_argument_group('Simulation settings')
    sims.add_argument('--nside', type=int, default=64,
                      help='HEALPix NSIDE')
    sims.add_argument('--nspec', type=int, default=100,
                      help='Spectra per healpixel')
    sims.add_argument('--nsim', type=int, default=10,
                      help='Number of simulations')
    sims.add_argument('--seed', type=int, default=None,
                      help='RNG seed')
    sims.add_argument('--host', choices=['bgs','elg','lrg'], default='bgs',
                      help='Host galaxy type')
    sims.add_argument('--outdir', default='',
                      help='Absolute path to simulation output')
    sims.add_argument('--coadd_cameras', action='store_true',
                      help='If true, coadd cameras in generated spectra')

    # Set up transient model as a simulation setting. None==no transient.
    tran = parser.add_argument_group('Transient settings')
    tran.add_argument('--magrange', nargs=2, type=float, default=[0.,2.5],
                      help='Transient-host mag range; e.g., 0 5')
    tran.add_argument('--epochrange', nargs=2, type=float, default=[-10.,10.],
                      help='Epoch range w.r.t. t0 in days; e.g., -10 10')
    modelnames = []
    mdict = transients.get_type_dict()
    for t, models in mdict.items():
        for m in models:
            modelnames.append(m)
    tran.add_argument('--transient', default=None,
                      help='None or one of these models: {}'.format(transients.get_type_dict()))

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):
    
    log = get_logger()

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    thedate = datetime.now().strftime('%Y-%m-%d')

    # Generate transient model if one is specified.
    trans_model = None
    if args.transient is not None:
        trans_model = transients.get_model(args.transient)

    # Generate list of HEALPix pixels to randomly sample the mocks.
    rng = np.random.RandomState(args.seed)
    nside = args.nside
    healpixels = _get_healpixels_in_footprint(nside=args.nside)
    npix = np.minimum(10*args.nsim, len(healpixels))
    pixels = rng.choice(healpixels, size=npix, replace=False)
    ipix = iter(pixels)

    # Set up the template generator.
    fluxratio_range = 10**(-np.sort(args.magrange)[::-1] / 2.5)
    epoch_range = np.sort(args.epochrange)

    if args.host == 'bgs':
        maker = BGSMaker(seed=args.seed)
        tmpl_maker = BGS
    elif args.host == 'elg':
        maker = ELGMaker(seed=args.seed)
        tmpl_maker = ELG
    elif args.host == 'lrg':
        maker = LRGMaker(seed=args.seed)
        tmpl_maker = LRG
    else:
        raise ValueError('Unusable host type {}'.format(args.host))

    maker.template_maker = tmpl_maker(transient=trans_model,
                                      tr_fluxratio=fluxratio_range,
                                      tr_epoch=epoch_range)

    for j in range(args.nsim):
        # Loop until finding a non-empty healpixel with mock galaxies.
        tdata = []
        while len(tdata) == 0:
            pixel = next(ipix)
            tdata = maker.read(healpixels=pixel, nside=args.nside)

        # Generate spectral templates and write them to truth files.
        # Keep producing templates until we have enough to pass brightness cuts.
        wave = None
        flux, targ, truth, objtr = [], [], [], []
        
        ntosim = np.min([args.nspec, len(tdata['RA'])])
        ngood = 0

        while ngood < args.nspec:
            idx = rng.choice(len(tdata['RA']), ntosim)
            tflux, twave, ttarg, ttruth, tobj = maker.make_spectra(tdata, indx=idx)
            g, r, z, w1, w2 = [ttruth['FLUX_{}'.format(_)] for _ in ['G','R','Z','W1','W2']]
            rfib = ttarg['FIBERFLUX_R']
#            print(g, r, z, w1, w2, rfib)

            # Apply color cuts.
            is_bright = isBGS_colors(rfib, g, r, z, w1, w2, targtype='bright')
            is_faint  = isBGS_colors(rfib, g, r, z, w1, w2, targtype='faint')
            is_wise   = isBGS_colors(rfib, g, r, z, w1, w2, targtype='wise')

            keep = np.logical_or.reduce([is_bright, is_faint, is_wise])

            _ngood = np.count_nonzero(keep)
            if _ngood > 0:
                ngood += _ngood
                flux.append(tflux[keep, :])
                targ.append(ttarg[keep])
                truth.append(ttruth[keep])
                objtr.append(tobj[keep])

        wave = maker.wave
        flux = np.vstack(flux)[:args.nspec, :]
        targ = vstack(targ)[:args.nspec]
        truth = vstack(truth)[:args.nspec]
        objtr = vstack(objtr)[:args.nspec]

        # Set up and verify the TARGETID across all truth tables.
        n = len(truth)
        new_id = 10000*pixel + 100*j + np.arange(1, n+1)
        targ['OBJID'][:] = new_id
        truth['TARGETID'][:] = new_id
        objtr['TARGETID'][:] = new_id

        assert(len(truth) == args.nspec)
        assert(np.all(targ['OBJID'] == truth['TARGETID']))
        assert(len(targ) == len(np.unique(targ['OBJID'])))
        assert(len(truth) == len(np.unique(truth['TARGETID'])))
        assert(len(objtr) == len(np.unique(objtr['TARGETID'])))

        truthfile = os.path.join(args.outdir,
                     '{}_{}_{:04d}s_{:03d}_truth.fits'.format(args.host, thedate, int(args.exptime), j+1))
        write_templates(truthfile, flux, wave, targ, truth, objtr)
        log.info('Wrote {}'.format(truthfile))

        # Get observing conditions and generate spectra.
        obs = dict(AIRMASS=args.airmass, EXPTIME=args.exptime,
                   MOONALT=args.moonalt, MOONFRAC=args.moonfrac,
                   MOONSEP=args.moonsep, SEEING=args.seeing)

        fcols = dict(BRICKID=targ['BRICKID'],
                     BRICK_OBJID=targ['OBJID'],
                     FLUX_G=targ['FLUX_G'],
                     FLUX_R=targ['FLUX_R'],
                     FLUX_Z=targ['FLUX_Z'],
                     FLUX_W1=targ['FLUX_W1'],
                     FLUX_W2=targ['FLUX_W2'],
                     FLUX_IVAR_G=targ['FLUX_IVAR_G'],
                     FLUX_IVAR_R=targ['FLUX_IVAR_R'],
                     FLUX_IVAR_Z=targ['FLUX_IVAR_Z'],
                     FLUX_IVAR_W1=targ['FLUX_IVAR_W1'],
                     FLUX_IVAR_W2=targ['FLUX_IVAR_W2'],
                     FIBERFLUX_G=targ['FIBERFLUX_G'],
                     FIBERFLUX_R=targ['FIBERFLUX_R'],
                     FIBERFLUX_Z=targ['FIBERFLUX_Z'],
                     FIBERTOTFLUX_G=targ['FIBERTOTFLUX_G'],
                     FIBERTOTFLUX_R=targ['FIBERTOTFLUX_R'],
                     FIBERTOTFLUX_Z=targ['FIBERTOTFLUX_Z'],
                     MW_TRANSMISSION_G=targ['MW_TRANSMISSION_G'],
                     MW_TRANSMISSION_R=targ['MW_TRANSMISSION_R'],
                     MW_TRANSMISSION_Z=targ['MW_TRANSMISSION_Z'],
                     EBV=targ['EBV'])

        specfile = os.path.join(args.outdir,
                    '{}_{}_{:04d}s_{:03d}_spect.fits'.format(args.host, thedate, int(args.exptime), j+1))

        # redshifts = truth['TRUEZ'] if args.host=='bgs' else None
        redshifts = None

        sim_spectra(wave, flux, args.host, specfile,
                    sourcetype=args.host,
                    obsconditions=obs, meta=obs, fibermap_columns=fcols,
                    targetid=truth['TARGETID'], redshift=redshifts,
                    ra=targ['RA'], dec=targ['DEC'],
                    seed=args.seed, expid=j)

        if args.coadd_cameras:
            coaddfile = specfile.replace('spect', 'coadd')
            spectra = read_spectra(specfile)
            spectra = coadd_cameras(spectra)

            write_spectra(coaddfile, spectra)
            log.info('Wrote {}'.format(coaddfile))

