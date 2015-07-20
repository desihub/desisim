#!/usr/bin/env python

"""
Simulate noiseless, infinite-resolution spectral templates for DESI.
"""
from __future__ import division, print_function

import os
import sys
import argparse

from desispec.io.util import makepath
from desispec.log import get_logger

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Simulate noiseless, infinite-resolution spectral templates for DESI.')
                                         

    parser.add_argument('--nmodel', type=long, default=None, metavar='', 
                        help='number of model (template) spectra to generate (required input)')
    # Optional inputs.
    parser.add_argument('--objtype', type=str, default='ELG', metavar='', 
                        help='object type (ELG, LRG, QSO, BGS, STD, or STAR)') 
    parser.add_argument('--minwave', type=float, default=3600, metavar='', 
                        help='minimum output wavelength range [Angstrom]')
    parser.add_argument('--maxwave', type=float, default=10000, metavar='', 
                        help='maximum output wavelength range [Angstrom]')
    parser.add_argument('--cdelt', type=float, default=2, metavar='', 
                        help='dispersion of output wavelength array [Angstrom/pixel]')
    # Boolean keywords.
    parser.add_argument('--nocolorcuts', action='store_true', 
                        help="""do not apply color cuts to select objects (only used for
                        object types ELG, LRG, and QSO)""")
    parser.add_argument('--notemplates', action='store_true', 
                        help="""do not generate templates
    (but do generate the diagnostic plots if OUTFILE exists)""")
    parser.add_argument('--noplot', action='store_true', 
                        help='do not generate diagnostic (QA) plots')
    # Output filenames.
    parser.add_argument('--outfile', type=str, default='OBJTYPE-templates.fits', metavar='', 
                        help='output FITS file name')
    parser.add_argument('--qafile', type=str, default='OBJTYPE-templates.pdf', metavar='', 
                        help='output file name for the diagnostic (QA) plots')
    
    # Objtype-specific defaults.
    elg_parser = parser.add_argument_group('options for objtype=ELG')
    elg_parser.add_argument('--zrange-elg', type=float, default=(0.6,1.6), nargs=2, metavar='', 
                            dest='zrange_elg', help='minimum and maximum redshift')
    elg_parser.add_argument('--rmagrange-elg', type=float, default=(21.0,23.5), nargs=2, metavar='',
                            dest='rmagrange_elg', help='Minimum and maximum r-band (AB) magnitude range')
    elg_parser.add_argument('--minoiiflux', type=float, default='1E-17', metavar='',
                            help='Minimum integrated [OII] 3727 flux')
    
    lrg_parser = parser.add_argument_group('options for objtype=LRG')
    lrg_parser.add_argument('--zrange-lrg', type=float, default=(0.5,1.1), nargs=2, metavar='', 
                            help='minimum and maximum redshift')
    lrg_parser.add_argument('--zmagrange-lrg', type=float, default=(19.5,20.6), nargs=2, metavar='',
                            help='Minimum and maximum z-band (AB) magnitude range')
    
    bgs_parser = parser.add_argument_group('options for objtype=BGS')
    bgs_parser.add_argument('--zrange-bgs', type=float, default=(0.01,0.4), nargs=2, metavar='', 
                            help='minimum and maximum redshift')
    bgs_parser.add_argument('--rmagrange-bgs', type=float, default=(15.0,19.0), nargs=2, metavar='',
                            help='Minimum and maximum r-band (AB) magnitude range')

    args = parser.parse_args()
    if args.nmodel is None:
        parser.print_help()
        sys.exit(1)

    log = get_logger()
    objtype = args.objtype.upper()
    
    # Check that the right environment variables are set.
    envOK = True
    for envvar in ['DESI_'+objtype+'_TEMPLATES']:
        if envvar not in os.environ:
            log.error('Missing ${} environment variable'.format(envvar))
            envOK = False
    if not envOK:
        sys.exit(1)
    
    # Set default output file name.
    if args.outfile:
        outfile = args.outfile
        if outfile == 'OBJTYPE-templates.fits':
            outfile = objtype.lower()+'-templates.fits'
    else: 
        outfile = objtype.lower()+'-templates.fits'
    
    log.info('Building {} {} templates.'.format(args.nmodel, objtype))
    cmd = " ".join(sys.argv)

    # Call the right Class depending on the object type.
    if not args.notemplates:
        if objtype == 'ELG':
            from desisim.templates import ELG
            elg = ELG(nmodel=args.nmodel,minwave=args.minwave,maxwave=args.maxwave,
                      cdelt=args.cdelt)
            elg.make_templates(zrange=args.zrange_elg,rmagrange=args.rmagrange_elg,
                               minoiiflux=args.minoiiflux,outfile=outfile,
                               comments=cmd)
        elif objtype == 'LRG':
            log.warning('{} objtype not yet supported!'.format(objtype))
            sys.exit(1)
        elif objtype == 'QSO':
            log.warning('{} objtype not yet supported!'.format(objtype))
            sys.exit(1)
        elif objtype == 'BGS':
            log.warning('{} objtype not yet supported!'.format(objtype))
            sys.exit(1)
        elif objtype == 'STD':
            log.warning('{} objtype not yet supported!'.format(objtype))
            sys.exit(1)
        elif objtype == 'STAR':
            log.warning('{} objtype not yet supported!'.format(objtype))
            sys.exit(1)
        else:
            log.warning('Object type {} not recognized'.format(objtype))
            sys.exit(1)
    
    # Generate diagnostic QAplots.
    if not args.noplot:
        import matplotlib.pyplot as plt
    
        if args.qafile:
            qafile = args.qafile
            if qafile == 'OBJTYPE-templates.pdf':
                qafile = objtype.lower()+'-templates.pdf'
            else: 
                qafile = objtype.lower()+'-templates.pdf'

        rzmin = 0.3
        slope1 = 1.0
        slope2 = -1.0
        int1 = -0.2
        int2 = 1.2

        plt.plot([1.0,2.0,3.0],[1.0,2.0,3.0],'rs')
        plt.savefig(qafile)
        
        #indx = np.where((rz>=rzmin)&(gr<=np.polyval([slope1,int1],rz))&
        #                (gr<=np.polyval([slope2,int2],rz)))[0]
    

if __name__ == '__main__':
    main()
