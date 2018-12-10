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
from desisim.spec_qa import __qa_version__

def parse(options=None):


    parser = argparse.ArgumentParser(description="Generate S/N QA for a production [v{:s}]".format(__qa_version__), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--rawdir', type = str, default = None, metavar = 'PATH',
    #                    help = 'Override default path ($DESI_SPECTRO_DATA) to processed data.')
    parser.add_argument('--qaprod_dir', type=str, default=None, help = 'Path to where QA figure files are generated.  Default is qaprod_dir')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(namespace=options)
    return args

def main(args):
    import os.path
    import numpy as np

    import desispec.io
    from desiutil.log import get_logger
    from desisim.spec_qa.s2n import load_s2n_values, obj_s2n_wave, obj_s2n_z
    from desisim.spec_qa.s2n import load_all_s2n_values
    from desisim.spec_qa.s2n import parse_s2n_values

    # Initialize
    if args.qaprod_dir is not None:
        qaprod_dir = args.qaprod_dir
    else:
        qaprod_dir = desispec.io.meta.qaprod_root()
    # Generate the path
    # Grab nights
    nights = desispec.io.get_nights()


    # Load all s2n (once)
    all_s2n_values = []
    channels = ['b', 'r', 'z']
    for channel in channels:
        print("Loading S/N for channel {}".format(channel))
        all_s2n_values.append(load_all_s2n_values(nights, channel))

    # Loop on channel
    for ss, channel in enumerate(channels):
        if channel == 'b':
            wv_bins = np.arange(3570., 5700., 20.)
        elif channel == 'r':
            wv_bins = np.arange(5750., 7600., 20.)
        elif channel == 'z':
            wv_bins = np.arange(7500., 9800., 20.)
            z_bins = np.linspace(1.0, 1.6, 100) # z camera
        else:
            raise IOError("Bad channel value: {}".format(channel))
        # Loop on OBJTYPE
        for objtype in ['ELG', 'LRG', 'QSO']:
            if objtype == 'ELG':
                flux_bins = np.linspace(19., 24., 6)
                oii_bins = np.array([1., 6., 10., 30., 100., 1000.])
            elif objtype == 'LRG':
                flux_bins = np.linspace(16., 22., 6)
            elif objtype == 'QSO':
                flux_bins = np.linspace(15., 24., 6)
            # Parse
            fdict = all_s2n_values[ss]
            s2n_dict = parse_s2n_values(objtype, fdict)
            # Plot
            outfile = qaprod_dir+'/QA_s2n_{:s}_{:s}.png'.format(objtype, channel)
            desispec.io.util.makepath(outfile)
            obj_s2n_wave(s2n_dict, wv_bins, flux_bins, objtype, outfile=outfile)
            # S/N vs. z for ELG
            if (channel == 'z') & (objtype=='ELG'):
                outfile = qaprod_dir+'/QA_s2n_{:s}_{:s}_redshift.png'.format(objtype,channel)
                desispec.io.util.makepath(outfile)
                obj_s2n_z(s2n_dict, z_bins, oii_bins, objtype, outfile=outfile)


