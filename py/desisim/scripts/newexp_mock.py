from __future__ import absolute_import, division, print_function
import sys, os
import argparse

import numpy as np
import astropy.table
import astropy.time
import astropy.units as u

from desisim.simexp import simscience, get_mock_spectra
import desisim.io
import desisim.util
from desiutil.log import get_logger

def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #- Required
    parser.add_argument('--fiberassign', type=str, required=True,
                        help="input fiberassign directory or tile file")
    parser.add_argument('--mockdir', type=str, required=True,
                        help="directory with mock targets and truth")
    parser.add_argument('--obslist', type=str, required=True,
                        help="input surveysim obslist file")
    parser.add_argument('--expid', type=int, required=True, help="exposure ID")

    #- Optional
    parser.add_argument('--nside', help='healpixel organization scheme of the mock spectra', type=int, default=64)
    parser.add_argument('--outdir', type=str, help="output directory")
    parser.add_argument('--nspec', type=int, default=None, help="number of spectra to include")
    parser.add_argument('--clobber', action='store_true', help="overwrite any pre-existing output files")

    log = get_logger()
    if options is None:
        args = parser.parse_args()
        log.info(' '.join(sys.argv))
    else:
        args = parser.parse_args(options)
        log.info('newexp-mock '+' '.join(options))

    return args

def main(args=None):

    log = get_logger()
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if args.obslist.endswith('.ecsv'):
        obslist = astropy.table.Table.read(args.obslist, format='ascii.ecsv')
    else:
        obslist = astropy.table.Table.read(args.obslist)

    obs = obslist[obslist['EXPID'] == args.expid][0]
    tileid = obs['TILEID']
    night = obs['NIGHT']
    program = obs['PROGRAM']

    if os.path.isdir(args.fiberassign):
        #- TODO: move file location logic to desispec / desitarget / fiberassign
        args.fiberassign = os.path.join(args.fiberassign, 'tile-{:06d}.fits'.format(tileid))
        if not os.path.exists(args.fiberassign):
            #- try previous name
            args.fiberassign = os.path.join(os.path.dirname(args.fiberassign), 'tile_{:06d}.fits'.format(tileid))

    fiberassign = astropy.table.Table.read(args.fiberassign, 'FIBERASSIGN')

    if args.outdir is None:
        args.outdir = desisim.io.simdir(night=night, expid=args.expid, mkdir=True)

    if args.nspec is None:
        args.nspec = len(fiberassign)
    elif args.nspec <= len(fiberassign):
        fiberassign = fiberassign[0:args.nspec]
    else:
        log.error('args.nspec {} > len(fiberassign) {}'.format(
            args.nspec, len(fiberassign)))
        sys.exit(1)

    log.info('Simulating night {} expid {} tile {}'.format(night, args.expid, tileid))
    try:
        flux, wave, meta, objmeta = get_mock_spectra(fiberassign, mockdir=args.mockdir, nside=args.nside)
    except Exception as err:
        log.fatal('Failed expid {} fiberassign {} tile {}'.format(
            args.expid, args.fiberassign, tileid))
        raise err

    sim, fibermap = simscience((flux, wave, meta), fiberassign,
        obsconditions=obs, psfconvolve=False)

    #- TODO: header keyword code is replicated from obs.new_exposure()
    telera, teledec = desisim.io.get_tile_radec(tileid)
    header = dict(
        NIGHT = (night, 'Night of observation YEARMMDD'),
        EXPID = (args.expid, 'DESI exposure ID'),
        TILEID = (tileid, 'DESI tile ID'),
        PROGRAM = (program, 'program [dark, bright, ...]'),
        FLAVOR = ('science', 'Flavor [arc, flat, science, zero, ...]'),
        TELRA = (telera, 'Telescope pointing RA [degrees]'),
        TELDEC = (teledec, 'Telescope pointing dec [degrees]'),
        AIRMASS = (obs['AIRMASS'], 'Airmass at middle of exposure'),
        EXPTIME = (obs['EXPTIME'], 'Exposure time [sec]'),
        SEEING = (obs['SEEING'], 'Seeing FWHM [arcsec]'),
        MOONFRAC = (obs['MOONFRAC'], 'Moon illumination fraction 0-1; 1=full'),
        MOONALT  = (obs['MOONALT'], 'Moon altitude [degrees]'),
        MOONSEP  = (obs['MOONSEP'], 'Moon:tile separation angle [degrees]'),
        )
    header['DATE-OBS'] = (sim.observation.exposure_start.isot, 'Start of exposure')

    #- Write fibermap to $DESI_SPECTRO_SIM/$PIXPROD not $DESI_SPECTRO_DATA
    fibermap.meta.update(header)
    fibermap.meta['EXTNAME'] = 'FIBERMAP'
    fibermap.write(desisim.io.findfile('simfibermap', night, args.expid,
        outdir=args.outdir), overwrite=args.clobber)

    desisim.io.write_simspec(sim, meta, fibermap, obs, args.expid, night, header=header,
                             objmeta=objmeta, outdir=args.outdir, overwrite=args.clobber)
