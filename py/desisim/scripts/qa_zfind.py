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

def parse(options=None):


    parser = argparse.ArgumentParser(description="Generate QA on redshift for a production [v1.1]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', action = 'store_true',
        help = 'Provide verbose reporting of progress.')
    parser.add_argument('--night', type=str, default = None, metavar = 'YYYYMMDD',
        help = 'Night to process in the format YYYYMMDD')
    parser.add_argument('--reduxdir', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--rawdir', type = str, default = None, metavar = 'PATH',
                        help = 'Override default path ($DESI_SPECTRO_REDUX/$SPECPROD) to processed data.')
    parser.add_argument('--qafile', type = str, default = None, required=False,
                        help = 'path of QA file.')
    parser.add_argument('--qafig', type=str, default=None, help = 'path of QA figure file')

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
    from matplotlib.backends.backend_pdf import PdfPages

    import desispec.io
    from desiutil.log import get_logger

    from desisim.spec_qa import redshifts as dsqa_z
    from desiutil.io import yamlify
    import desiutil.depend
    from desimodel.footprint import radec2pix

    log = get_logger()

    #if args.night is None:
    #    log.critical('Missing required night argument.')
    #    return -1

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
            '''
            # Search for zbest files
            fibermap_data = desispec.io.read_fibermap(fibermap_path)
            flavor = fibermap_data.meta['FLAVOR']
            if flavor.lower() in ('arc', 'flat', 'bias'):
                log.debug('Skipping calibration {} exposure {:08d}'.format(flavor, exposure))
                continue

            brick_names = set(fibermap_data['BRICKNAME'])
            import pdb; pdb.set_trace()
            for brick in brick_names:
                zbest_path=desispec.io.findfile('zbest', groupname=brick, specprod_dir=args.reduxdir)
                if os.path.exists(zbest_path):
                    log.debug('Found {}'.format(os.path.basename(zbest_path)))
                    zbest_files.append(zbest_path)
                else:
                    log.warn('Missing {}'.format(os.path.basename(zbest_path)))
                    #pdb.set_trace()
            '''
            # Load data
            fibermap_data = desispec.io.read_fibermap(fibermap_path)
            # Skip calib
            if fibermap_data['OBJTYPE'][0] in ['FLAT','ARC','BIAS']:
                continue
            elif fibermap_data['OBJTYPE'][0] in ['SKY','STD','SCIENCE']:
                pass
            else:
                pdb.set_trace()
            # Append fibermap file
            fibermap_files.append(fibermap_path)
            # Search for zbest files with healpy
            ra_targ = fibermap_data['RA_TARGET'].data
            dec_targ = fibermap_data['DEC_TARGET'].data
            # Getting some NAN in RA/DEC
            good = np.isfinite(ra_targ) & np.isfinite(dec_targ)
            pixels = radec2pix(64, ra_targ[good], dec_targ[good])
            uni_pixels = np.unique(pixels)
            for uni_pix in uni_pixels:
                zbest_files.append(desispec.io.findfile('zbest', groupname=uni_pix, nside=64))

    # Cut down zbest_files to unique ones
    zbest_files = list(set(zbest_files))

    if len(zbest_files) == 0:
        log.fatal('No zbest files found')
        sys.exit(1)

    # Load Table
    simz_tab = dsqa_z.load_z(fibermap_files, zbest_files)
    #from astropy.table import Table
    #simz_tab = Table.read('tmp_simz.fits')

    # Meta data
    meta = dict(
        DESISIM = desiutil.depend.getdep(simz_tab.meta, 'desisim'),
        SPECTER = desiutil.depend.getdep(simz_tab.meta, 'specter'),
        SPECPROD = os.getenv('SPECPROD', 'unknown'),
        PIXPROD = os.getenv('PIXPROD', 'unknown'),
        )
    
    # Run stats
    summ_dict = dsqa_z.summ_stats(simz_tab)
    if args.qafile is not None:
        log.info("Generating yaml file: {:s}".format(args.qafile))
        # yamlify
        # Write yaml
        with open(args.qafile, 'w') as outfile:
            outfile.write(yaml.dump(yamlify(meta), default_flow_style=False))
            outfile.write(yaml.dump(yamlify(summ_dict), default_flow_style=False))

    if args.qafig is not None:
        log.info("Generating PDF file: {:s}".format(args.qafig))
        pp = PdfPages(args.qafig)
        # Summary for dz of all types
        dsqa_z.dz_summ(simz_tab, pp=pp)
        # Summary of individual types
        dsqa_z.summ_fig(simz_tab, summ_dict, meta, pp=pp)
        for objtype in ['ELG','LRG', 'QSO_T', 'QSO_L']:
            dsqa_z.obj_fig(simz_tab, objtype, summ_dict, pp=pp)
        log.info("closing QA figure file")
        # All done
        pp.close()

if __name__ == '__main__':
    main()
