from __future__ import absolute_import, division, print_function
import sys, os
import argparse

import numpy as np
import astropy.table
import astropy.time
import astropy.units as u

import desisim.simexp
import desisim.obs
import desisim.io
import desisim.util
from desiutil.log import get_logger

def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #- Required
    parser.add_argument('--program', type=str, required=False, default='qso',
        help="Program name, e.g. dark, bright, gray.")

    parser.add_argument('--zrange', type=str, required=False, default='2.6:3.6',
        help="Quasar redshift range distribution (uniformly sampled).")
    parser.add_argument('--colorcut', action='store_true', help="Apply a colorcut, which would randomize continua.")
    parser.add_argument(
        '--dwave-out', type=float, default=0.2,
        help="Output wavelength step size in A.")
    # parser.add_argument('--add-lyaforest', action='store_true',
    #     help="Adds Lya forest to QSO spectra AFTER magnitude normalization.")
    # parser.add_argument('--dla')
    # parser.add_argument('--bal')

    #- Optional observing conditions to override program defaults
    parser.add_argument('--seeing', type=float, default=None, help="Seeing FWHM [arcsec]")
    parser.add_argument('--airmass', type=float, default=None, help="Airmass")
    parser.add_argument('--exptime', type=float, default=None, help="Exposure time [sec]")
    parser.add_argument('--moonfrac', type=float, default=None, help="Moon illumination fraction; 1=full")
    parser.add_argument('--moonalt', type=float, default=None, help="Moon altitude [degrees]")
    parser.add_argument('--moonsep', type=float, default=None, help="Moon separation to tile [degrees]")

    #- Optional
    parser.add_argument('--expid', type=int, default=None, help="exposure ID")
    parser.add_argument('--night', type=int, default=None, help="YEARMMDD of observation")
    parser.add_argument('--tileid', type=int, default=None, help="Tile ID")
    parser.add_argument('--outdir', type=str, help="output directory")
    parser.add_argument('--nspec', type=int, default=5000, help="number of spectra to include")
    parser.add_argument('--clobber', action='store_true', help="overwrite any pre-existing output files")
    parser.add_argument('--seed', type=int, default=None, help="Random number seed")
    parser.add_argument('--nproc', type=int, default=None, help="Number of multiprocessing processes")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args):

    log = get_logger()

    #- Generate obsconditions with args.program, then override as needed
    args.program = args.program.upper()
    if args.program in ['ARC', 'FLAT']:
        obsconditions = None
    else:
        obsconditions = desisim.simexp.reference_conditions[args.program]
        if args.airmass is not None:
            obsconditions['AIRMASS'] = args.airmass
        if args.seeing is not None:
            obsconditions['SEEING'] = args.seeing
        if args.exptime is not None:
            obsconditions['EXPTIME'] = args.exptime
        if args.moonfrac is not None:
            obsconditions['MOONFRAC'] = args.moonfrac
        if args.moonalt is not None:
            obsconditions['MOONALT'] = args.moonalt
        if args.moonsep is not None:
            obsconditions['MOONSEP'] = args.moonsep

    specify_targets = dict({})
    zrange = tuple([float(z) for z in args.zrange.split(':')])
    specify_targets['QSO'] = {
        # 'lyaforest':args.add_lyaforest,
        'zrange':zrange,
        'nocolorcuts': not args.colorcut,
        # 'add_forest_after_magnorm': True
        }

    sim, fibermap, meta, obs, objmeta = desisim.obs.new_exposure(args.program,
        specify_targets=specify_targets, nspec=args.nspec, night=args.night, expid=args.expid, 
        tileid=args.tileid, nproc=args.nproc, seed=args.seed, 
        obsconditions=obsconditions, outdir=args.outdir, dwave_out=args.dwave_out)
