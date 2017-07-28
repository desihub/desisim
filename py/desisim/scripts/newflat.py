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
import desisim.simexp
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
    parser.add_argument('--outdir', type=str, help="output directory")
    parser.add_argument('--nspec', type=int, default=5000, help="number of spectra to include")
    parser.add_argument('--nonuniform', action='store_true', help="Include calibration screen non-uniformity")
    parser.add_argument('--clobber', action='store_true', help="overwrite any pre-existing output files")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.simspec is None:
        args.simspec = desisim.io.findfile('simspec', args.night, args.expid,
                                           outdir=args.outdir)

    if args.fibermap is None:
        #- put in same directory as simspec by default
        filedir = os.path.dirname(os.path.abspath(args.simspec))
        filename = os.path.basename(desispec.io.findfile('fibermap', args.night, args.expid))
        args.fibermap = os.path.join(filedir, filename)

    return args

def main(args=None):
    '''
    Generates a new flat exposure; see newflat --help for usage options
    '''
    import desiutil.log
    log = desiutil.log.get_logger()

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    sim, fibermap = \
        desisim.simexp.simflat(args.flatfile, nspec=args.nspec, nonuniform=args.nonuniform)

    log.info('Writing {}'.format(args.fibermap))
    fibermap.meta['NIGHT'] = args.night
    fibermap.meta['EXPID'] = args.expid
    fibermap.write(args.fibermap, overwrite=args.clobber)

    header = fits.Header()
    desiutil.depend.add_dependencies(header)
    header['EXPID'] = args.expid
    header['NIGHT'] = args.night
    header['FLAVOR'] = 'flat'
    header['DOSVER'] = 'SIM'

    #- Set calibrations as happening at 15:00 AZ local time = 22:00 UTC
    year = int(args.night[0:4])
    month = int(args.night[4:6])
    day = int(args.night[6:8])
    tx = astropy.time.Time(datetime.datetime(year, month, day, 22, 0, 0))
    header['DATE-OBS'] = tx.utc.isot

    #- metadata truth and obs dictionary are None
    desisim.io.write_simspec(sim, None, fibermap, None, args.expid, args.night,
        filename=args.simspec, header=header, overwrite=args.clobber)



