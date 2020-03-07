"""
desisim.scripts.quicktransients
===============================
"""

import os
import numpy as np
import healpy as hp

from desisim.templates import BGS
from desisim.transients import transients

from desitarget.mock.mockmaker import BGSMaker
from desitarget.cuts import isBGS_colors

from desisim.simexp import reference_conditions

from desiutil.log import get_logger, DEBUG

from argparse import ArgumentParser


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


def parse(options=None):
    """Parse command line options.
    """
    parser = ArgumentParser(description='Fast galaxy + transient simulator')

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

    # Set up transient model as a simulation setting. None==no transient.
    modelnames = []
    mdict = transients.get_type_dict()
    for t, models in mdict.items():
        for m in models:
            modelnames.append(m)
    sims.add_argument('--transient', default=None,
                      help='Models: {}'.format(transients.get_type_dict()))
    sims.add_argument('--deltamag', nargs=2, type=float, default=[0.,5.],
                      help='Transient-host mag range; e.g., 0, 5')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def main(args=None):
    
    log = get_logger()

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    # Generate list of HEALPix pixels to randomly sample the mocks.
    rng = np.random.RandomState(args.seed)
    nside = args.nside
    healpixels = _get_healpixels_in_footprint(nside=args.nside)
    npix = np.minimum(10*args.nsim, len(healpixels))
    pixels = rng.choice(healpixels, size=npix, replace=False)
    ipix = iter(pixels)

    # Set up the template generator.
    maker = BGSMaker(seed=args.seed)
#    maker.template_maker = BGS(wave=_default_wave())

    for j in range(args.nsim):
        # Loop until finding a non-empty healpixel with mock galaxies.
        tdata = []
        while len(tdata) == 0:
            pixel = next(ipix)
            tdata = maker.read(healpixels=pixel, nside=args.nside)

        # Generate spectral templates and write them to truth files.
        # Keep producing templates until we have enough to pass brightness cuts.
        wave = None
        flux, targ, truth, obj = [], [], [], []
        
        ntosim = np.min([args.nspec, len(tdata['RA'])])
        ngood = 0

        while ngood < args.nspec:
            idx = rng.choice(len(tdata['RA']), ntosim)
            tflux, twave, ttarg, ttruth, tobj = maker.make_spectra(tdata, indx=idx)

            g, r, z, w1, w2 = [ttruth['FLUX_{}'.format(_)] for _ in ['G','R','Z','W1','W2']]
            rfib = ttarg['FIBERFLUX_R']
            print(g, r, z, w1, w2, rfib)

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
                obj.append(tobj[keep])
