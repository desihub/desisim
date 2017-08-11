"""
desisim.scripts.brightsims
==========================

Generate a canonical set of bright-time simulations.
"""

from __future__ import division, print_function

import os
import sys
import numpy as np
import argparse
from argparse import Namespace

from astropy.io import fits
from astropy.table import Table

import desisim.scripts.quickgen as quickbrick
from desiutil.log import get_logger, DEBUG
from desispec.io.util import write_bintable, makepath

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Generate a canonical set of bright-time simulations.')

    parser.add_argument('-b', '--brickname', type=str, help='unique output brickname suffix',
                        default='brightsims', metavar='')

    parser.add_argument('--objtype', type=str,  help='BGS, MWS, or BRIGHT_MIX', default='BRIGHT_MIX', metavar='')
    parser.add_argument('--nexp', type=int,  help='number of bricks to simulate', default=10, metavar='')
    parser.add_argument('--nspec', type=int,  help='number of spectra (per brick) to simulate', default=100, metavar='')

    parser.add_argument('-s', '--seed', type=int,  help='random seed', default=None, metavar='')
    parser.add_argument('-o', '--outdir', type=str,  help='output directory for simulation parameters',
                        default='.', metavar='')
    parser.add_argument('--brickdir', type=str,  help='top-level output bricks directory',
                        default='./bricks', metavar='')
    parser.add_argument('-v', '--verbose', action='store_true', help='toggle on verbose output')

    parser.add_argument('--exptime-range', type=float, default=(300, 300), nargs=2, metavar='',
                        help='minimum and maximum exposure time (s)')
    parser.add_argument('--airmass-range', type=float, default=(1.25, 1.25), nargs=2, metavar='',
                        help='minimum and maximum airmass')

    parser.add_argument('--moon-phase-range', type=float, default=(0.0, 1.0), nargs=2, metavar='',
                        help='minimum and maximum lunar phase (0=full, 1=new')
    parser.add_argument('--moon-angle-range', type=float, default=(0, 150), nargs=2, metavar='',
                        help='minimum and maximum lunar separation angle (0-180 deg')
    parser.add_argument('--moon-zenith-range', type=float, default=(0, 60), nargs=2, metavar='',
                        help='minimum and maximum lunar zenith angle (0-90 deg')

    bgs_parser = parser.add_argument_group('options for BGS objects')
    bgs_parser.add_argument('--zrange-bgs', type=float, default=(0.01, 0.4), nargs=2, metavar='',
                            help='minimum and maximum redshift')
    bgs_parser.add_argument('--rmagrange-bgs', type=float, default=(15.0, 19.5), nargs=2, metavar='',
                            help='Minimum and maximum r-band (AB) magnitude range')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args):

    # Set up the logger.
    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    objtype = args.objtype.upper()

    log.debug('Using OBJTYPE {}'.format(objtype))
    log.debug('Simulating {:g} bricks each with {:g} spectra'.format(args.nexp, args.nspec))

    # Draw priors uniformly given the input ranges.
    rand = np.random.RandomState(args.seed)
    exptime = rand.uniform(args.exptime_range[0], args.exptime_range[1], args.nexp)
    airmass = rand.uniform(args.airmass_range[0], args.airmass_range[1], args.nexp)
    moonphase = rand.uniform(args.moon_phase_range[0], args.moon_phase_range[1], args.nexp)
    moonangle = rand.uniform(args.moon_angle_range[0], args.moon_angle_range[1], args.nexp)
    moonzenith = rand.uniform(args.moon_zenith_range[0], args.moon_zenith_range[1], args.nexp)
    maglimits = args.rmagrange-bgs
    zrange = args.zrange-bgs
    # Build a metadata table with the simulation inputs.
    metafile = makepath(os.path.join(args.outdir, '{}-input.fits'.format(args.brickname)))
    metacols = [
        ('BRICKNAME', 'S20'),
        ('SEED', 'S20'),
        ('EXPTIME', 'f4'),
        ('AIRMASS', 'f4'),
        ('MOONPHASE', 'f4'),
        ('MOONANGLE', 'f4'),
        ('MOONZENITH', 'f4')]
    meta = Table(np.zeros(args.nexp, dtype=metacols))
    meta['EXPTIME'].unit = 's'
    meta['MOONANGLE'].unit = 'deg'
    meta['MOONZENITH'].unit = 'deg'

    meta['BRICKNAME'] = ['{}-{:03d}'.format(args.nexp, ii) for ii in range(args.nexp)]
    meta['EXPTIME'] = exptime
    meta['AIRMASS'] = airmass
    meta['MOONPHASE'] = moonphase
    meta['MOONANGLE'] = moonangle
    meta['MOONZENITH'] = moonzenith

    log.debug('Writing {}'.format(metafile))
    write_bintable(metafile, meta, extname='METADATA', clobber=True)

    # Generate each brick in turn.
    for ii in range(args.nexp):
        thisbrick = meta['BRICKNAME'][ii]
        log.debug('Building brick {}'.format(thisbrick))

        brickargs = ['--brickname', thisbrick,
                     '--objtype', args.objtype,
                     '--nspec', '{}'.format(args.nspec),
                     '--outdir', os.path.join(args.brickdir, thisbrick),
                     '--outdir-truth', os.path.join(args.brickdir, thisbrick),
                     '--exptime', '{}'.format(exptime[ii]),
                     '--airmass', '{}'.format(airmass[ii]),
                     '--moon-phase', '{}'.format(moonphase[ii]),
                     '--moon-angle', '{}'.format(moonangle[ii]),
                     '--moon-zenith', '{}'.format(moonzenith[ii]),
                     '--zrange-bgs', '{}'.format(args.zrange_bgs[0]), '{}'.format(args.zrange_bgs[1]),
                     '--rmagrange-bgs', '{}'.format(args.rmagrange_bgs[0]), '{}'.format(args.rmagrange_bgs[1])]
        if args.seed is not None:
            brickargs.append('--seed')
            brickargs.append('{}'.format(args.seed))

        quickargs = quickbrick.parse(brickargs)
        if args.verbose:
            quickargs.verbose = True
        quickbrick.main(quickargs)
