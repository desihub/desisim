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
    #if args.simspec is None:
    #    if args.night is None or args.expid is None:
    #        msg = 'Must set --simspec or both --night and --expid'
    #        log.error(msg)
    #        raise ValueError(msg)
    #    args.simspec = io.findfile('simspec', args.night, args.expid)

    #- expand camera list
    if args.cameras is not None:
        args.cameras = args.cameras.split(',')

    #- write to same directory as simspec
    #if args.rawfile is None:
    #    rawfile = os.path.basename(desispec.io.findfile('raw', args.night, args.expid))
    #    args.rawfile = os.path.join(os.path.dirname(args.simspec), rawfile)

    #if args.simpixfile is None:
    #    outdir = os.path.dirname(os.path.abspath(args.rawfile))
    #    args.simpixfile = io.findfile(
    #        'simpix', night=args.night, expid=args.expid, outdir=outdir)

    if args.keywords is not None :
        res={}
        for kv in args.keywords.split(",") :
            t=kv.split("=")
            if len(t)==2 :
                k=t[0]
                v=t[1]
                if isinstance(v,str) :
                    try :
                        v=int(v)
                        typed=True
                    except :
                        pass
                if isinstance(v,str) :
                    try :
                        v=float(v)
                    except :
                        pass
                res[k]=v
        args.keywords=res


#-------------------------------------------------------------------------
#- Parse options
def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Generates simulated DESI pixel-level raw data',
        )

    #- Inputs
    parser.add_argument("--simspec", type=str, help="input simspec file", required=True)
    parser.add_argument("--psf", type=str, help="PSF filename, optional", default=None)
    parser.add_argument("--cosmics", action="store_true", help="Add cosmics")

    #- Outputs
    parser.add_argument("--rawfile", type=str, help="output raw data file", required=True)
    parser.add_argument("--simpixfile", type=str, required=False, default=None,
                        help="optional output truth image file")
    parser.add_argument("--outfibermap", type=str, required=False, default=None,
                        help="optional output fibermap")
    parser.add_argument("--cameras", type=str, help="cameras, e.g. b0,r5,z9")
    parser.add_argument("--keywords", type=str, default=None, help="optional additional keywords in header of rawfile of the form 'key1=val1,key2=val2,...")

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

    #- Not yet supported so don't pretend it is
    ### parser.add_argument("--seed", type=int, help="random number seed")

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
                          comm=comm,keywords=args.keywords, outfibermap=args.outfibermap)
