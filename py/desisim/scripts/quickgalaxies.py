"""
desisim.scripts.quickgalaxies
=============================
"""
from __future__ import absolute_import, division, print_function

import healpy as hp
import numpy as np
import os

from datetime import datetime

from abc import abstractmethod, ABCMeta
from argparse import Action, ArgumentParser

from astropy.table import Table, vstack

from desisim.templates import BGS
from desisim.scripts.quickspectra import sim_spectra
from desitarget.mock.mockmaker import BGSMaker
from desitarget.cuts import isBGS_colors
from desiutil.log import get_logger, DEBUG

from yaml import load

import matplotlib.pyplot as plt


class SetDefaultFromFile(Action, metaclass=ABCMeta):
    """Abstract interface class to set command-line arguments from a file."""

    def __call__(self, parser, namespace, values, option_string=None):
        config = self._get_config_from_file(values)
        for key, value in config.items():
            setattr(namespace, key, value)

    @abstractmethod
    def _get_config_from_file(self, filename):
        raise NotImplementedError


class SetDefaultFromYAMLFile(SetDefaultFromFile):
    """Concrete class that sets command-line arguments from a YAML file."""

    def _get_config_from_file(self, filename):
        """Implementation of configuration reader.

        Parameters
        ----------
        filename : string
            Name of configuration file to read.

        Returns
        -------
        config : dictionary
            Configuration dictionary.
        """

        with open(filename, 'r') as f:
            config = load(f)
        return config


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


def _default_wave(wavemin=None, wavemax=None, dw=0.2):
    """Generate a default wavelength vector for the output spectra."""
    from desimodel.io import load_throughput

    if wavemin is None:
        wavemin = load_throughput('b').wavemin - 10.0
    if wavemax is None:
        wavemax = load_throughput('z').wavemax + 10.0

    return np.arange(round(wavemin, 1), wavemax, dw)


def bgs_write_simdata(sim, overwrite=False):
    """Create a metadata table with simulation inputs.

    Parameters
    ----------
    sim : dict
        Simulation parameters from command line.
    overwrite : bool
        Overwrite simulation data file.

    Returns
    -------
    simdata : Table
        Data table written to disk.
    """
    from desispec.io.util import makepath
    from desispec.io.util import write_bintable

    simdatafile = os.path.join(sim.simdir, 
                               'bgs_{}_simdata.fits'.format(sim.simid))
    makepath(simdatafile)

    cols = [
        ('SEED', 'S20'),
        ('NSPEC', 'i4'),
        ('EXPTIME', 'f4'),
        ('AIRMASS', 'f4'),
        ('SEEING', 'f4'),
        ('MOONFRAC', 'f4'),
        ('MOONSEP', 'f4'),
        ('MOONALT', 'f4')]

    simdata = Table(np.zeros(sim.nsim, dtype=cols))
    simdata['EXPTIME'].unit = 's'
    simdata['SEEING'].unit = 'arcsec'
    simdata['MOONSEP'].unit = 'deg'
    simdata['MOONALT'].unit = 'deg'

    simdata['SEED'] = sim.seed
    simdata['NSPEC'] = sim.nspec
    simdata['AIRMASS'] = sim.airmass
    simdata['SEEING'] = sim.seeing
    simdata['MOONALT'] = sim.moonalt
    simdata['MOONSEP'] = sim.moonsep
    simdata['MOONFRAC'] = sim.moonfrac
    simdata['EXPTIME'] = sim.exptime

    if overwrite or not os.path.isfile(simdatafile):
        print('Writing {}'.format(simdatafile))
        write_bintable(simdatafile, simdata, extname='SIMDATA', clobber=True)

    return simdata


def simdata2obsconditions(sim):
    """Pack simdata observation conditions into a dictionary.

    Parameters
    ----------
    simdata : Table
        Simulation data table.

    Returns
    -------
    obs : dict
        Observation conditions dictionary.
    """
    obs = dict(AIRMASS=sim.airmass,
               EXPTIME=sim.exptime,
               MOONALT=sim.moonalt,
               MOONFRAC=sim.moonfrac,
               MOONSEP=sim.moonsep,
               SEEING=sim.seeing)
    return obs


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

    print('Writing {}'.format(filename))
    hx.writeto(filename, overwrite=True)


def parse(options=None):
    """Parse command-line options.
    """
    parser = ArgumentParser(description='Fast galaxy simulator')
    parser.add_argument('--config', action=SetDefaultFromYAMLFile)
    #
    # Observational conditions.
    #
    cond = parser.add_argument_group('Observing conditions')
    cond.add_argument('--airmass', dest='airmass', type=float, default=1.,
                      help='Airmass [1..40].')
    cond.add_argument('--exptime', dest='exptime', type=int, default=300,
                      help='Exposure time [s].')
    cond.add_argument('--seeing', dest='seeing', type=float, default=1.1,
                      help='Seeing [arcsec].')
    cond.add_argument('--moonalt', dest='moonalt', type=float, default=-60.,
                      help='Moon altitude [deg].')
    cond.add_argument('--moonfrac', dest='moonfrac', type=float, default=0.,
                      help='Illuminated moon fraction [0..1].')
    cond.add_argument('--moonsep', dest='moonsep', type=float, default=180.,
                      help='Moon separation angle [deg].')
    #
    # Galaxy simulation settings.
    #
    mcset = parser.add_argument_group('Simulation settings')
    mcset.add_argument('--nside', dest='nside', type=int, default=64,
                       help='HEALPix NSIDE parameter.')
    mcset.add_argument('--nspec', dest='nspec', type=int, default=100,
                       help='Number of spectra per HEALPix pixel.')
    mcset.add_argument('--nsim', dest='nsim', type=int, default=10,
                       help='Number of simulations (HEALPix pixels).')
    mcset.add_argument('--seed', dest='seed', type=int, default=None,
                       help='Random number seed')
    mcset.add_argument('--addsnia', dest='addsnia', action='store_true', default=False,
                       help='Add SNe Ia to host spectra.')
    mcset.add_argument('--addsniip', dest='addsniip', action='store_true', default=False,
                       help='Add SNe IIp to host spectra.')
    mcset.add_argument('--snrmin', dest='snrmin', type=float, default=0.01,
                       help='SN/host minimum flux ratio.')
    mcset.add_argument('--snrmax', dest='snrmax', type=float, default=1.00,
                       help='SN/host maximum flux ratio.')
    #
    # Output settings.
    #
    output = parser.add_argument_group('Output settings')
    output.add_argument('--simid', dest='simid',
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='ID/name for simulations.')
    output.add_argument('--simdir', dest='simdir', default='',
                        help='Simulation output directory absolute path.')

    # Parse command line options.
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):

    log = get_logger()

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # Save simulation output.
    rng = np.random.RandomState(args.seed)
    simdata = bgs_write_simdata(args)
    obs = simdata2obsconditions(args)

    # Generate list of HEALPix pixels to randomly sample from the mocks.
    healpixels = _get_healpixels_in_footprint(nside=args.nside)
    npix = np.minimum(10*args.nsim, len(healpixels))
    pixels = rng.choice(healpixels, size=npix, replace=False)
    ipix = iter(pixels)

    # Set up the template generator.
    maker = BGSMaker(seed=args.seed)
    maker.template_maker = BGS(add_SNeIa=args.addsnia,add_SNeIIp=args.addsniip, wave=_default_wave())

    for j in range(args.nsim):

        # Loop until finding a non-empty healpixel (one with mock galaxies).
        tdata = []
        while len(tdata) == 0:
            pixel = next(ipix)
            tdata = maker.read(healpixels=pixel, nside=args.nside)

        # Add SN generation options.
        if args.addsnia or args.addsniip:
            tdata['SNE_FLUXRATIORANGE'] = (args.snrmin, args.snrmax)
            tdata['SNE_FILTER'] = 'decam2014-r'

        # Generate nspec spectral templates and write them to "truth" files.
        wave = None
        flux, targ, truth, obj = [], [], [], []

        # Generate templates until we have enough to pass brightness cuts.
        ntosim = np.min((args.nspec, len(tdata['RA'])))
        ngood = 0
        while ngood < args.nspec:
            idx = rng.choice(len(tdata['RA']), ntosim)
            tflux, twave, ttarg, ttruth, tobj = \
                maker.make_spectra(tdata, indx=idx)

            # Apply color cuts.
            is_bright = isBGS_colors(gflux=ttruth['FLUX_G'],
                                     rflux=ttruth['FLUX_R'],
                                     zflux=ttruth['FLUX_Z'],
                                     w1flux=ttruth['FLUX_W1'],
                                     w2flux=ttruth['FLUX_W2'],
                                     targtype='bright')

            is_faint = isBGS_colors(gflux=ttruth['FLUX_G'],
                                    rflux=ttruth['FLUX_R'],
                                    zflux=ttruth['FLUX_Z'],
                                    w1flux=ttruth['FLUX_W1'],
                                    w2flux=ttruth['FLUX_W2'],
                                    targtype='faint')

            is_wise  = isBGS_colors(gflux=ttruth['FLUX_G'],
                                    rflux=ttruth['FLUX_R'],
                                    zflux=ttruth['FLUX_Z'],
                                    w1flux=ttruth['FLUX_W1'],
                                    w2flux=ttruth['FLUX_W2'],
                                    targtype='wise')
            
            keep = np.logical_or(np.logical_or(is_bright, is_faint), is_wise)

            _ngood = np.count_nonzero(keep)
            if _ngood > 0:
                ngood += _ngood
                flux.append(tflux[keep, :])
                targ.append(ttarg[keep])
                truth.append(ttruth[keep])
                obj.append(tobj[keep])

        wave = maker.wave
        flux = np.vstack(flux)[:args.nspec, :]
        targ = vstack(targ)[:args.nspec]
        truth = vstack(truth)[:args.nspec]
        obj = vstack(obj)[:args.nspec]

        if args.addsnia or args.addsniip:
            # TARGETID in truth table is split in two; deal with it here.
            truth['TARGETID'] = truth['TARGETID_1']

        # Set up and verify the TARGETID across all truth tables.
        n = len(truth)
        new_id = 10000000*pixel + 100000*j + np.arange(1, n+1)

        truth['TARGETID'][:] = new_id
        targ['TARGETID'][:] = new_id
        obj['TARGETID'][:] = new_id

        assert(len(truth) == args.nspec)
        assert(np.all(targ['TARGETID'] == truth['TARGETID']))
        assert(len(truth) == len(np.unique(truth['TARGETID'])))
        assert(len(targ) == len(np.unique(targ['TARGETID'])))
        assert(len(obj) == len(np.unique(obj['TARGETID'])))

        truthfile = os.path.join(args.simdir,
                                 'bgs_{}_{:03}_truth.fits'.format(args.simid, j))
        write_templates(truthfile, flux, wave, targ, truth, obj)

        # Generate simulated spectra, given observing conditions.
        specfile = os.path.join(args.simdir,
                                'bgs_{}_{:03}_spectra.fits'.format(args.simid, j))
        sim_spectra(wave, flux, 'bgs', specfile, obsconditions=obs,
                    sourcetype='bgs', targetid=truth['TARGETID'],
                    redshift=truth['TRUEZ'], seed=args.seed, expid=j)
