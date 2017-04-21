from __future__ import absolute_import, division, print_function
import sys, os
import argparse
import datetime
import time
import warnings

import numpy as np
import astropy.table
import astropy.time
from astropy.io import fits
import astropy.units as u

import desimodel.io
import desisim.newexp
import desisim.io
import desispec.io
import desiutil.depend

def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    if 'DESI_ROOT' in os.environ:
        _default_flatfile = os.path.join(os.getenv('DESI_ROOT'),
            'spectro', 'templates', 'calib', 'v0.3', 'flat-3100K-quartz-iodine.fits')
    else:
        _default_flatfile = None

    #- Required
    parser.add_argument('--expid', type=int, help="exposure ID")
    parser.add_argument('--night', type=str, help="YEARMMDD")

    #- Optional
    parser.add_argument('--flatfile', type=str, help="input flatlamp calib spec file",
        default=_default_flatfile)
    parser.add_argument('--simspec', type=str, help="output simspec file")
    parser.add_argument('--fibermap', type=str, help="output fibermap file")
    parser.add_argument('--nspec', type=int, default=5000, help="number of spectra to include")
    parser.add_argument('--nonuniform', action='store_true', help="Include calibration screen non-uniformity")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.simspec is None:
        args.simspec = desisim.io.findfile('simspec', args.night, args.expid)

    if args.fibermap is None:
        #- put in same directory as simspec by default
        filedir = os.path.dirname(os.path.abspath(args.simspec))
        filename = os.path.basename(desispec.io.findfile('fibermap', args.night, args.expid))
        args.fibermap = os.path.join(filedir, filename)

    return args

def main(args=None):
    '''
    TODO: document
    
    Note: this bypasses specsim since we don't have an arclamp model in
    surface brightness units; we only have electrons on the CCD
    '''
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)
    
    sim, fibermap = \
        desisim.newexp.newflat(args.flatfile, nspec=args.nspec, nonuniform=args.nonuniform)

    fibermap.write(args.fibermap)

    header = fits.Header()
    desiutil.depend.add_dependencies(header)
    header['EXPID'] = args.expid
    header['NIGHT'] = args.night
    header['FLAVOR'] = 'flat'
    header['DOSVER'] = 'SIM'

    #- TODO: DATE-OBS on night instead of now
    tx = astropy.time.Time(datetime.datetime(*time.gmtime()[0:6]))
    header['DATE-OBS'] = tx.utc.isot

    desisim.io.write_simspec(sim, None, args.expid, args.night,
        filename=args.simspec, header=header)
    
    
    
