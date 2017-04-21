from __future__ import absolute_import, division, print_function
import sys, os
import argparse

import numpy as np
import astropy.table
import astropy.time

from desisim.newexp import newexp
import desisim.io

def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #- Required
    parser.add_argument('--fiberassign', type=str, help="input fiberassign tile file")
    parser.add_argument('--obslist', type=str, help="input surveysim obslist file")
    parser.add_argument('--mockdir', type=str, help="directory with mock targets and truth")
    parser.add_argument('--obsnum', type=int, default=None, help="index in obslist file to use")

    #- Optional
    parser.add_argument('--expid', type=int, default=None, help="exposure ID")
    parser.add_argument('--outdir', type=str, help="output directory")
    parser.add_argument('--nspec', type=int, default=None, help="number of spectra to include")

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    if args.expid is None:
        args.expid = args.obsnum

    return args

def main(args=None):

    if args is None:
        args = parse()

    fiberassign = astropy.table.Table.read(args.fiberassign, 'FIBER_ASSIGNMENTS')

    if args.obslist.endswith('.ecsv'):
        obslist = astropy.table.Table.read(args.obslist, format='ascii.ecsv')
    else:
        obslist = astropy.table.Table.read(args.obslist)

    obs = obslist[args.obsnum]

    dateobs = astropy.time.Time(obs['MJD'], format='mjd')
    night = dateobs.utc.isot[0:10].replace('-', '')     #- YEARMMDD string

    if args.outdir is None:
        args.outdir = desisim.io.simdir(night=night, mkdir=True)

    if args.nspec is None:
        args.nspec = len(fiberassign)

    sim, fibermap, meta = newexp(fiberassign, args.mockdir, obsconditions=obs, nspec=args.nspec)

    fibermap.meta['NIGHT'] = night
    fibermap.meta['EXPID'] = args.expid
    fibermap.write(desisim.io.findfile('simfibermap', night, args.expid,
        outdir=args.outdir))
    header = dict(FLAVOR='science')
    desisim.io.write_simspec(sim, meta, args.expid, night, header=header,
        outdir=args.outdir)
