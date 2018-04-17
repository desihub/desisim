#!/usr/bin/env python
#
# See top-level LICENSE.rst file for Copyright information
#
# -*- coding: utf-8 -*-
"""
Read fibermaps and zbest files to generate QA related to redshifts
 and compare against the 'true' values
"""

import argparse
from desisim.spec_qa import __qa_version__

def parse(options=None):


    parser = argparse.ArgumentParser(description="Generate QA on redshift for a production [v{:s}]".format(__qa_version__), formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--load_simz_table', type = str, default = None, required=False,
                        help = 'Load an existing simz Table to remake figures')
    #parser.add_argument('--reduxdir', type = str, default = None, metavar = 'PATH',
    #                    help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--rawdir', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--yaml_file', type = str, default = None, required=False,
                        help = 'YAML file for debugging (primarily).')
    parser.add_argument('--write_simz_table', type=str, default=None, help = 'Write simz to this filename')
    parser.add_argument('--qafig_path', type=str, default=None, help = 'Path to where QA figure files are generated.  Default is specprod_dir+/QA')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args):
    import os.path
    import sys
    import yaml
    import numpy as np
    import pdb
    import matplotlib
    matplotlib.use('agg')

    import desispec.io
    from desiutil.log import get_logger

    from desisim.spec_qa import redshifts as dsqa_z
    from desiutil.io import yamlify
    import desiutil.depend

    log = get_logger()

    # Initialize
    specprod_dir = desispec.io.meta.specprod_root()

    if args.qafig_path is not None:
        qafig_path = args.qafig_path
    else:
        qafig_path = desispec.io.meta.qaprod_root()


    if args.load_simz_table is not None:
        from astropy.table import Table
        log.info("Loading simz info from {:s}".format(args.load_simz_table))
        simz_tab = Table.read(args.load_simz_table)
    else:
        # Grab list of fibermap files
        fibermap_files = []
        zbest_files = []
        nights = desispec.io.get_nights()
        for night in nights:
            for exposure in desispec.io.get_exposures(night, raw=True, rawdata_dir=args.rawdir):
                # Ignore exposures with no fibermap, assuming they are calibration data.
                fibermap_path = desispec.io.findfile(filetype='fibermap', night=night,
                                                     expid=exposure, rawdata_dir=args.rawdir)
                if not os.path.exists(fibermap_path):
                    log.debug('Skipping exposure %08d with no fibermap.' % exposure)
                    continue
                # Load data
                fibermap_data = desispec.io.read_fibermap(fibermap_path)
                # Skip calib
                if fibermap_data['OBJTYPE'][0] in ['FLAT','ARC','BIAS']:
                    continue
                elif fibermap_data['OBJTYPE'][0] in ['SKY','STD','SCIENCE','BGS','MWS_STAR','ELG', 'LRG', 'QSO']:
                    pass
                else:
                    pdb.set_trace()
                # Append fibermap file
                fibermap_files.append(fibermap_path)
                # Slurp the zbest_files
                zbest_files += dsqa_z.find_zbest_files(fibermap_data)

        # Cut down zbest_files to unique ones
        zbest_files = list(set(zbest_files))

        if len(zbest_files) == 0:
            log.fatal('No zbest files found')
            sys.exit(1)

        # Write? Table
        simz_tab, zbtab = dsqa_z.load_z(fibermap_files, zbest_files=zbest_files)
        dsqa_z.match_truth_z(simz_tab, zbtab)
        if args.write_simz_table is not None:
            simz_tab.write(args.write_simz_table, overwrite=True)

    # Meta data
    meta = dict(
        DESISIM = desiutil.depend.getdep(simz_tab.meta, 'desisim'),
        SPECPROD = os.getenv('SPECPROD', 'unknown'),
        PIXPROD = os.getenv('PIXPROD', 'unknown'),
        )
    # Include specter version if it was used to generate input files
    # (it isn't used for specsim inputs so that dependency may not exist)
    try:
        meta['SPECTER'] = desiutil.depend.getdep(simz_tab.meta, 'specter')
    except KeyError:
        pass
    
    # Run stats
    log.info("Running stats..")
    summ_dict = dsqa_z.summ_stats(simz_tab)
    if args.yaml_file is not None:
        log.info("Generating yaml file of stats: {:s}".format(args.yaml_file))
        # yamlify
        # Write yaml
        desispec.io.util.makepath(args.yaml_file)
        with open(args.yaml_file, 'w') as outfile:
            outfile.write(yaml.dump(yamlify(meta), default_flow_style=False))
            outfile.write(yaml.dump(yamlify(summ_dict), default_flow_style=False))

    log.info("Generating QA files")
    # Summary for dz of all types
    outfile = qafig_path+'/QA_dzsumm.png'
    desispec.io.util.makepath(outfile)
    dsqa_z.dz_summ(simz_tab, outfile=outfile)
    # Summary of individual types
    #outfile = args.qafig_root+'_summ_fig.png'
    #dsqa_z.summ_fig(simz_tab, summ_dict, meta, outfile=outfile)
    for objtype in ['BGS', 'MWS', 'ELG','LRG', 'QSO_T', 'QSO_L']:
        outfile = qafig_path+'/QA_zfind_{:s}.png'.format(objtype)
        desispec.io.util.makepath(outfile)
        dsqa_z.obj_fig(simz_tab, objtype, summ_dict, outfile=outfile)

if __name__ == '__main__':
    main()
