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

    parser = argparse.ArgumentParser(description = 'Simulate noiseless, infinite-resolution spectral templates for DESI.')
    #formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     
    # Mandatory inputs.
    parser.add_argument('--nmodel', type=long, default=None, metavar='', 
                        help='number of model (template) spectra to generate (required input)')
    # Optional inputs.
    parser.add_argument('--objtype', type=str, default='ELG', metavar='', 
                        help='object type (ELG, LRG, QSO, BGS, STD, or STAR)') 
    parser.add_argument('--minwave', type=float, default=3600, metavar='', 
                        help='minimum output wavelength range (default 3600) [Angstrom]')
    parser.add_argument('--maxwave', type=float, default=10000, metavar='', 
                        help='maximum output wavelength range (default 10000) [Angstrom]')
    parser.add_argument('--cdelt', type=float, default=2, metavar='', 
                        help='dispersion of output wavelength array (default 2) [Angstrom/pixel]')
    parser.add_argument('--zrange', type=float, nargs=2, metavar='', 
                        help='minimum and maximum redshift (default depends on OBJTYPE)')
    parser.add_argument('--rmagrange', type=float, nargs=2, metavar='',
                        help='Minimum and maximum r-band (AB) magnitude range (default depends on OBJTYPE)')
    # Boolean keywords.
    parser.add_argument('--nocolorcuts', action='store_true', 
                        help="""do not apply color cuts to select objects (only used for
                        object types ELG, LRG, and QSO)""")
    parser.add_argument('--notemplates', action='store_true', 
                        help="""do not generate templates
    (but do generate the diagnostic plots if OUTFILE exists)""")
    parser.add_argument('--noqaplots', action='store_true', 
                        help='do not generate diagnostic (QA) plots')
    # Output filenames.
    parser.add_argument('--outfile', type=str, metavar='', 
                        help='output FITS file name (default OBJTYPE-templates.fits')
    parser.add_argument('--qafile', type=str, metavar='', 
                        help='output file name for the diagnostic (QA) plots (default OBJTYPE-templates.pdf)')
    
    # Objtype-specific defaults.
    elg_parser = parser.add_argument_group('options for objtype=ELG (default zrange=[0.6,1.6])')
    #elg_parser.add_argument('--rmagrange', type=float, nargs=2, metavar='',
    #                        help='Minimum and maximum r-band (AB) magnitude range')
    elg_parser.add_argument('--minoiiflux', type=float, default='1E-17', metavar='',
                            help='Minimum integrated [OII] 3727 flux')
    
    lrg_parser = parser.add_argument_group('options for objtype=LRG (default zrange=[0.5,1.1])')
    lrg_parser.add_argument('--zmagrange', type=float, default=[19.5,20.6], nargs=2, metavar='',
                            help='Minimum and maximum z-band (AB) magnitude range')
    
    bgs_parser = parser.add_argument_group('options for objtype=BGS',
                                           description="""default zrange=[0.01,0.4]\n
                                          default rmagrange=[15,19]""")
    #bgs_parser.add_argument('--rmagrange', type=float, nargs=2, metavar='',
    #                        help='Minimum and maximum r-band (AB) magnitude range')

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
    else: 
        outfile = objtype.lower()+'-templates.fits'
    
    log.info('Building {} {} templates.'.format(args.nmodel, objtype))
    
    # Call the right Class depending on the object type.
    if not args.notemplates:
        if objtype == 'ELG':
            from desisim.templates import ELG
            default_zrange = (0.6,1.6)
            default_rmagrange = (21.0,23.5)
            elg = ELG(nmodel=args.nmodel)
            elg.make_templates(zrange=default_zrange,rmagrange=default_rmagrange,
                               minoiiflux=args.minoiiflux,outfile=outfile)
        elif objtype == 'LRG':
            default_zrange = (0.5,1.1)
            print('{} objtype not yet supported!'.format(objtype))
        elif objtype == 'QSO':
            print('{} objtype not yet supported!'.format(objtype))
        elif objtype == 'BGS':
            default_zrange = (0.01,0.4)
            default_rmagrange = (15.0,19.0)
            print('{} objtype not yet supported!'.format(objtype))
        elif objtype == 'STD':
            print('{} objtype not yet supported!'.format(objtype))
        elif objtype == 'STAR':
            print('{} objtype not yet supported!'.format(objtype))
        else:
            log.warning('Object type {} not recognized'.format(objtype))
            sys.exit(1)
    
    # Generate diagnostic QAplots.
    if not args.noqaplots:
        import matplotlib.pyplot as plt
    
        rzmin = 0.3
        slope1 = 1.0
        slope2 = -1.0
        int1 = -0.2
        int2 = 1.2
    
        #indx = np.where((rz>=rzmin)&(gr<=np.polyval([slope1,int1],rz))&
        #                (gr<=np.polyval([slope2,int2],rz)))[0]
    

if __name__ == '__main__':
    main()
