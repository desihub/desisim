"""
desisim.scripts.pixsim
======================

This is a module.
"""
from __future__ import absolute_import, division, print_function

import os,sys
import os.path
import shutil

import random
from time import asctime

import numpy as np

import desimodel.io
from desiutil.log import get_logger
import desispec.io
from desispec.parallel import stdouterr_redirected

from ..pixsim import simulate_exposure
from .. import io

log = get_logger()

def expand_args(args):
    '''expand camera string into list of cameras
    '''
    if args.simspec is None:
        if args.night is None or args.expid is None:
            msg = 'Must set --simspec or both --night and --expid'
            log.error(msg)
            raise ValueError(msg)
        args.simspec = io.findfile('simspec', args.night, args.expid)

    #- expand camera list
    if args.cameras is not None:
        args.cameras = args.cameras.split(',')

    #- write to same directory as simspec
    if args.rawfile is None:
        rawfile = os.path.basename(desispec.io.findfile('raw', args.night, args.expid))
        args.rawfile = os.path.join(os.path.dirname(args.simspec), rawfile)

    if args.simpixfile is None:
        args.simpixfile = io.findfile(
            'simpix', night=args.night, expid=args.expid,
            outdir=os.path.dirname(os.path.abspath(args.rawfile)))


#-------------------------------------------------------------------------
#- Parse options
def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Generates simulated DESI pixel-level raw data',
        )

    #- Inputs
    parser.add_argument("--simspec", type=str, help="input simspec file")
    parser.add_argument("--psf", type=str, help="PSF filename")
    parser.add_argument("--cosmics", action="store_true", help="Add cosmics")
    # parser.add_argument("--cosmics_dir", type=str, 
    #     help="Input directory with cosmics templates")
    # parser.add_argument("--cosmics_file", type=str, 
    #     help="Input file with cosmics templates")

    #- Outputs
    parser.add_argument("--rawfile", type=str, help="output raw data file")
    parser.add_argument("--simpixfile", type=str, 
        help="output truth image file")

    #- Alternately derive inputs/outputs from night, expid, and cameras
    parser.add_argument("--night", type=str, help="YEARMMDD")
    parser.add_argument("--expid", type=int, help="exposure id")
    parser.add_argument("--cameras", type=str, help="cameras, e.g. b0,r5,z9")

    parser.add_argument("--ccd_npix_x", type=int, 
        help="for testing; number of x (columns) to include in output", 
        default=None)
    parser.add_argument("--ccd_npix_y", type=int, 
        help="for testing; number of y (rows) to include in output", 
        default=None)

    parser.add_argument("--verbose", action="store_true", 
        help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", 
        help="Overwrite existing raw and simpix files")
    parser.add_argument("--seed", type=int, help="random number seed")

    parser.add_argument("--ncpu", type=int, 
        help="Number of cpu cores per thread to use", default=0)
    parser.add_argument("--wavemin", type=float, 
        help="Minimum wavelength to simulate")
    parser.add_argument("--wavemax", type=float, 
        help="Maximum wavelength to simulate")
    parser.add_argument("--nspec", type=int, 
        help="Number of spectra to simulate per camera")

    if options is None:
        args = parser.parse_args()
    else:
        options = [str(x) for x in options]
        args = parser.parse_args(options)

    expand_args(args)
    return args

def main(args, comm=None):
    if args.verbose:
        import logging
        log.setLevel(logging.DEBUG)

    if comm is None or comm.rank == 0:
        log.info('Starting pixsim at {}'.format(asctime()))
        if args.overwrite and os.path.exists(args.rawfile):
           log.debug('Removing {}'.format(args.rawfile))
           os.remove(args.rawfile)

    simulate_exposure(args.simspec, args.rawfile, cameras=args.cameras,
        simpixfile=args.simpixfile, addcosmics=args.cosmics,
        nspec=args.nspec, wavemin=args.wavemin, wavemax=args.wavemax,
        comm=comm)

