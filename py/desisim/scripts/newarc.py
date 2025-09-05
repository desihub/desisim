import sys, os
import argparse
import datetime
import time
import warnings

import numpy as np
import astropy.table
# See pixsim.py
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
        _default_arcfile = os.path.join(os.getenv('DESI_ROOT'),
            'spectro', 'templates', 'calib', 'v0.4', 'arc-lines-average-in-vacuum-from-winlight-20170118.fits')
    else:
        _default_arcfile = None

    #- Required
    parser.add_argument('--expid', type=int, help="exposure ID")
    parser.add_argument('--night', type=str, help="YEARMMDD")

    #- Optional
    parser.add_argument('--arcfile', type=str, help="input arc calib spec file",
        default=_default_arcfile)
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
    TODO: document

    Note: this bypasses specsim since we don't have an arclamp model in
    surface brightness units; we only have electrons on the CCD
    '''
    import desiutil.log
    log = desiutil.log.get_logger()

    from desiutil.iers import freeze_iers
    freeze_iers()

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    log.info('reading arc data from {}'.format(args.arcfile))
    arcdata = astropy.table.Table.read(args.arcfile)

    wave, phot, fibermap = \
        desisim.simexp.simarc(arcdata, nspec=args.nspec, nonuniform=args.nonuniform)

    log.info('Writing {}'.format(args.fibermap))
    fibermap.meta['NIGHT'] = args.night
    fibermap.meta['EXPID'] = args.expid
    fibermap.meta['EXTNAME'] = 'FIBERMAP'
    fibermap.write(args.fibermap, overwrite=args.clobber)

    #- TODO: explain bypassing desisim.io.write_simspec
    header = fits.Header()
    desiutil.depend.add_dependencies(header)
    header['EXPID'] = args.expid
    header['NIGHT'] = args.night
    header['FLAVOR'] = 'arc'
    header['DOSVER'] = 'SIM'
    header['EXPTIME'] = 5       #- TODO: add exptime support

    #- TODO: DATE-OBS on night instead of now
    tx = astropy.time.Time(datetime.datetime(*time.gmtime()[0:6]))
    header['DATE-OBS'] = tx.utc.isot

    desisim.io.write_simspec_arc(args.simspec, wave, phot, header, fibermap, overwrite=args.clobber)
