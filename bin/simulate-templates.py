#!/usr/bin/env python

"""
Simulate noiseless, infinite-resolution spectral templates for DESI.

ToDo:
 * Include a minimum [OII] flux cut
 * Allow the user to modify the grz color cuts.
 * Allow the priors on the emission-line parameters to be varied.
 * Make the random seed an optional input so the templates are reproducible.
 * Should I worry about the emission-line strengths when synthesizing grz?
"""
from __future__ import division, print_function

import os
import sys
import argparse

from desispec.io.util import makepath
from desispec.log import get_logger

# Parse the simulation parameters from the command line and choose a
# reasonable set of default values.
parser = argparse.ArgumentParser(
    usage = '%(prog)s [options]',
    description = 'Simulate noiseless, infinite-resolution spectral templates for DESI.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Mandatory inputs.
parser.add_argument('--nmodel', type=long, default=50, metavar='', 
                    help='number of model (template) spectra to generate')

# Optional inputs.
parser.add_argument('--objtype', type=str, default='ELG', metavar='', 
                    help='object type (ELG, LRG, QSO, BGS, STD, or STAR)') 
parser.add_argument('--minwave', type=float, default=3600, metavar='', 
                    help='minimum output wavelength range (Angstrom)')
parser.add_argument('--maxwave', type=float, default=10000, metavar='', 
                    help='maximum output wavelength range (Angstrom)')
parser.add_argument('--cdelt', type=float, default=2, metavar='', 
                    help='dispersion of the output wavelength array (Angstrom/pixel)')
# Output filenames.
parser.add_argument('--outfile', type=str, default='{OBJTYPE}-templates.fits', metavar='', 
                    help='output FITS file name')
parser.add_argument('--qafile', type=str, default='{OBJTYPE}-templates.pdf', metavar='', 
                    help='output file name')
# Boolean keywords.
parser.add_argument('--notemplates', action='store_false', help="""do not generate templates (but do generate
the diagnostic plots if OUTFILE already exists)""")
parser.add_argument('--noplots', action='store_false', help='do not generate diagnostic (QA) plots')


# Objtype-specific defaults.
elg_parser = parser.add_argument_group('options for objtype=ELG:')
elg_parser.add_argument('--zrange', type=float, default=[0.6,1.6], nargs=2, metavar='', 
                      help='minimum and maximum redshift')
elg_parser.add_argument('--rmagrange', type=float, default=[21.0,23.5], nargs=2, metavar='',
                      help='Minimum and maximum r-band (AB) magnitude range')

lrg_parser = parser.add_argument_group('options for objtype=LRG:')
#lrg_parser.add_argument('--zrange', type=float, default='0.5 1.1', nargs=2, metavar='', 
#                      help='minimum and maximum redshift')
lrg_parser.add_argument('--zmagrange', type=float, default=[19.5,20.6], nargs=2, metavar='',
                      help='Minimum and maximum z-band (AB) magnitude range')

log = get_logger()

args = parser.parse_args()
objtype = args.objtype.upper()

# Check that the right environment variables are set.
envOK = True
for envvar in ['DESI_'+objtype+'_TEMPLATES']:
    if envvar not in os.environ:
        print('Missing ${} environment variable'.format(envvar))
        envOK = False
if not envOK:
    sys.exit(1)

# Set default output file name.
if args.outfile:
    outfile = args.outfile
else: 
    outfile = objtype.lower()+'-templates.fits'

# Call the right Class depending on the object type.
if not args.notemplates:
    if objtype == 'ELG':
        from desisim.templates import ELG
    
        elg = ELG(nmodel=args.nmodel)
        elg.make_templates(zrange=args.zrange,rmagrange=args.rmagrange,
                           outfile=outfile)
	elif objtype == 'LRG':
	    print('{} objtype yet supported!'.format(objtype))
	elif objtype == 'QSO':
	    print('{} objtype yet supported!'.format(objtype))
	elif objtype == 'STD':
	    print('{} objtype yet supported!'.format(objtype))
	elif objtype == 'STAR':
	    print('{} objtype yet supported!'.format(objtype))

# Optionally generate diagnostic plots.
if not args.noplots:
    import matplotlib.pyplot as plt
    print('Hey!')
