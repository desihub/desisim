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

from astropy.table import Table, Column, vstack
import astropy.units as u
from astropy.io import fits

from specsim.simulator import Simulator
from desimodel.io import load_desiparams
from desispec.io import fitsheader, empty_fibermap, Brick
from desispec.resolution import Resolution
from desispec.log import get_logger, DEBUG
from desisim.targets import sample_objtype
from desisim.obs import get_night
import desisim.templates

def parse(options=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Quickly generate brick files.')

    # Mandatory input
    parser.add_argument('-p', '--prefix', type=str, help='unique output brickname suffix (required input)', metavar='')

    # Simulation options
    parser.add_argument('--objtype', type=str,  help='ELG, LRG, QSO, BGS, MWS, WD, DARK_MIX, or BRIGHT_MIX', default='DARK_MIX', metavar='')
    parser.add_argument('-b', '--nbrick', type=int,  help='number of spectra to simulate', default=100, metavar='')
    parser.add_argument('-n', '--nspec', type=int,  help='number of spectra to simulate', default=100, metavar='')

    parser.add_argument('-s', '--seed', type=int,  help='random seed', default=None, metavar='')
    parser.add_argument('-o', '--outdir', type=str,  help='output directory', default='.', metavar='')
    parser.add_argument('-v', '--verbose', action='store_true', help='toggle on verbose output')

    parser.add_argument('--exptime-range', type=float, default=(300300), nargs=2, metavar='', 
                        help='minimum and maximum exposure time (s)')
    parser.add_argument('--airmass-range', type=float, default=(1.25,1.25), nargs=2, metavar='', 
                        help='minimum and maximum airmass')
    parser.add_argument('--moon-phase-range', type=float, default=(0.0,1.0), nargs=2, metavar='', 
                        help='minimum and maximum lunar phase (0=full, 1=new')
    parser.add_argument('--moon-angle-range', type=float, default=(0,90), nargs=2, metavar='', 
                        help='minimum and maximum lunar separation angle (0-90 deg')
    parser.add_argument('--moon-zenith-range', type=float, default=(0,90), nargs=2, metavar='', 
                        help='minimum and maximum lunar zenith angle (0-90 deg')
 
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


    known_objtype = ('ELG', 'LRG', 'QSO', 'BGS', 'MWS', 'WD', 'DARK_MIX', 'BRIGHT_MIX')
    if args.objtype.upper() not in known_objtype:
        log.critical('Unknown OBJTYPE {}'.format(args.objtype))
        return -1
        
    rand = np.random.RandomState(args.seed)

    # Initialize the quick simulator object and its optional parameters.
    log.debug('Initializing specsim Simulator with configuration file {}'.format(args.config))
    desiparams = load_desiparams()
    qsim = Simulator(args.config)

    objtype = args.objtype.upper()
    log.debug('Using OBJTYPE {}'.format(objtype))
    if objtype == 'BGS' or objtype == 'MWS' or objtype == 'BRIGHT_MIX':
        qsim.instrument.exposure_time = desiparams['exptime_bright'] * u.s
        qsim.atmosphere.moon.moon_zenith = args.moon_zenith * u.deg
        qsim.atmosphere.moon.separation_angle = args.moon_angle * u.deg
        qsim.atmosphere.moon.moon_phase = args.moon_phase
    else:
        qsim.instrument.exposure_time = desiparams['exptime_dark'] * u.s
