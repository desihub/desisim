#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Generate S/N plots as a function of object type for the
current production
"""

import argparse

def parse(options=None):


    parser = argparse.ArgumentParser(description="Generate S/N QA for a production [v1.0]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--reduxdir', type = str, default = None, metavar = 'PATH',
    #                    help = 'Override default path ($DESI_SPECTRO_REDUX) to processed data.')
    #parser.add_argument('--rawdir', type = str, default = None, metavar = 'PATH',
    #                    help = 'Override default path ($DESI_SPECTRO_DATA) to processed data.')
    parser.add_argument('--qafig_root', type=str, default=None, help = 'Root name (and path) of QA figure files')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args):
    import os.path
    import numpy as np
    import pdb

    import desispec.io
    from desiutil.log import get_logger
    from desisim.spec_qa.s2n import load_s2n_values, obj_s2n_wave, obj_s2n_z


    log = get_logger()

    prod = os.getenv('SPECPROD')
    if prod is None:
        log.fatal("Must set $SPECPROD")

    if args.qafig_root is not None:
        qafig_root = args.qafig_root
    else:
        qafig_root = prod
    # Grab nights
    nights = desispec.io.get_nights()

    # Loop on channel
    for channel in ['b','r', 'z']:
        if channel == 'b':
            wv_bins = np.arange(3570., 5950., 20.)
        elif channel == 'z':
            wv_bins = np.arange(7000., 11000., 20.)
            z_bins = np.linspace(1.0, 1.6, 100) # z camera
        else:
            pdb.set_trace()
        # Loop on OBJTYPE
        for objtype in ['ELG', 'LRG', 'QSO']:
            if objtype == 'ELG':
                flux_bins = np.linspace(19., 24., 6)
                oii_bins = np.array([1., 6., 10., 30., 100., 1000.])
            elif objtype == 'LRG':
                flux_bins = np.linspace(16., 22., 6)
            elif objtype == 'QSO':
                flux_bins = np.linspace(15., 24., 6)
            # Load
            s2n_values = load_s2n_values(objtype, nights, channel)#, sub_exposures=exposures)
            # Plot
            outfile = qafig_root+'_s2n_{:s}_{:s}.png'.format(objtype, channel)
            obj_s2n_wave(s2n_values, wv_bins, flux_bins, objtype, outfile=outfile)
            # S/N vs. z for ELG
            if (channel == 'z') & (objtype=='ELG'):
                outfile = qafig_root+'_s2n_{:s}_{:s}_redshift.png'.format(objtype,channel)
                obj_s2n_z(s2n_values, z_bins, oii_bins, objtype, outfile=outfile)


