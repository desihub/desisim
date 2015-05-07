#!/usr/bin/env python

"""
Generate simulated galaxy templates.

Give examples...
"""

from __future__ import division, print_function

import os
import sys
import numpy as np

from optparse import OptionParser

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

from desispec.log import get_logger

from desisim.templates import EMSpectrum
from desisim.templates import read_templates
from desisim.filter import filter
from imaginglss.analysis import cuts
from desispec.interpolation import resample_flux
import desispec.io.util

# parse the simulation parameters from the command line and choose a
# reasonable set of default values
parser = OptionParser(
    usage = '%prog [options]',
    description = 'This is a neat piece of code.',
    epilog = 'Not sure what this does!')

parser.add_option('--nmodel', default=5, type=long,
                  help='Number of model (template) spectra to generate [%default]')
parser.add_option('--objtype', default='elg', 
                  help='Object type to simulate (elg, lrg, bgs, star) [%default]')
parser.add_option('--oiiihbeta_range', default=(-0.5,0.0), type=float, nargs=2, 
                  help='Logarithmic minimum and maximum [OIII]/Hbeta ratio [%default]')
parser.add_option('--oiiratio_meansig', default=(0.73,0.05), type=float, nargs=2, 
                  help='Mean and sigma of [OII] 3729/3726 doublet ratio [%default]')
parser.add_option('--linesigma_meansig', default=(75.0,20.0), type=float, nargs=2, 
                  help='Mean and sigma of emission-line width [%default km/s]')
parser.add_option('--redshift_range', default=(0.6,1.6), type=float, nargs=2, 
                  help='Minimum and maximum redshift [%default]')
parser.add_option('--rmag_range', default=(21.0,23.5), type=float, nargs=2, 
                  help='Minimum and maximum magnitude range [%default]')

opts, args = parser.parse_args()
log = get_logger()

objtype = opts.objtype

# check environment
envOK = True
for envvar in ['DESI_'+objtype.upper()+'_TEMPLATES']:
    if envvar not in os.environ:
        print('Hi!')
        print('Missing ${} environment variable'.format(envvar))
        envOK = False
if not envOK:
    sys.exit(1)

# Initialize the grz filter profiles. Need an exception if this fails!
gfilt = filter(filtername='decam_g.txt')
rfilt = filter(filtername='decam_r.txt')
zfilt = filter(filtername='decam_z.txt')

# Read the rest-frame continuum basis spectra and K-corrections.  Need an
# exception if this failed!
baseflux, basewave, basemeta, basekcorr = read_templates(objtype=objtype,continuum=True,kcorrections=True)
#baseflux, basewave, basemeta = read_templates(objtype=objtype,continuum=True)
nbase = len(basemeta)

# Split the desired number of models into chunks because not all models will
# satisfy the grz color and [OII] flux cuts.

# ToDo:
# * Include a minimum [OII] flux cut
# * Allow the user to modify the grz color cuts.

nchunk = min(opts.nmodel,500)

nobj = 0
while nobj<opts.nmodel:
    # Choose a random subset of the templates and only keep the ones that
    # satisfy the grz color and [OII] flux cuts.
    these = np.random.randint(0,nbase-1,nchunk)

    redshift = np.random.uniform(opts.redshift_range[0],opts.redshift_range[1],nchunk)
    rmag = np.random.uniform(opts.rmag_range[0],opts.rmag_range[1],nchunk)

    d4000 = basemeta['D4000'][these]
    ewoii = 10.0**np.polyval([1.1074,-4.7338,5.6585],d4000) # rest-frame, Angstrom

    # The only way to not have to loop here would be to pre-compute the
    # K-corrections on a uniform redshift grid.
    for ii, iobj in enumerate(these):
        zfactor = 1.0+redshift[ii]
        wave = basewave*zfactor
        flux = baseflux[iobj,:]/zfactor
        
        rflux = 10.0**(-0.4*(rmag[ii]-22.5)) # nanomaggies
        rnorm = rflux/rfilt.get_maggies(wave,flux)
        gflux = gfilt.get_maggies(wave,flux)*rnorm
        zflux = zfilt.get_maggies(wave,flux)*rnorm
        oiiflux = basemeta['OII_CONTINUUM'][iobj]*ewoii[ii]*rnorm*10**(-0.4*22.5) # erg/s/cm2

        grzmask = cuts.Fluxes.ELG(gflux=gflux,rflux=rflux,zflux=zflux)
        oiimask = [1] # generalize this

        # Build the final template if the grz color and [OII] flux cuts are
        # satisfied.
        if all(grzmask) and all(oiimask):
            print(iobj)
            nobj = nobj+1

            plt.clf()
            plt.loglog(wave,flux,'r')
            plt.show()


## draw random values assuming that these quantities are uncorrelated
#oiiihbeta = np.random.uniform(opts.oiiihbeta_range[0],opts.oiiihbeta_range[1],opts.nmodel)
#oiiratio = np.random.normal(opts.oiiratio_meansig[0],opts.oiiratio_meansig[1],opts.nmodel)
#linesigma = np.random.normal(opts.linesigma_meansig[0],opts.linesigma_meansig[1],opts.nmodel)
#
## pack simulation parameters into a binary table
## ...
#
### build a default spectrum in order to initialize the
### emission-line data and output wavelength array
#
#for ii in range(opts.nmodel):
#
#    # initialize the simulation parameters for this spectrum
#    print(ii, oiiihbeta[ii], linesigma[ii], oiiratio[ii])
#    EM = EMSpectrum(linesigma=linesigma[ii], oiiratio=oiiratio[ii],
#                oiiihbeta=oiiihbeta[ii])
#
#    # build the emission-line spectrum
#    emflux = EM.emlines()
#    
#    plt.clf()
#    log10wave = EM.wavelength()
#    plt.plot(10**log10wave,emflux,'r')
#    #plt.xlim([3600,4100])
#    plt.show()

#baseflux, basewave, basemeta, basekcorr = read_templates(objtype=objtype,
#                                                         continuum=True,
#                                                         kcorrections=True)
#    # grz color cuts
#    gr = -2.5*np.log10(interp1d(basekcorr['REDSHIFT'][0],basekcorr['GMAGGIES'][0,these,:])(redshift)/
#                       interp1d(basekcorr['REDSHIFT'][0],basekcorr['RMAGGIES'][0,these,:])(redshift))
#    rz = -2.5*np.log10(interp1d(basekcorr['REDSHIFT'][0],basekcorr['RMAGGIES'][0,these,:])(redshift)/
#                       interp1d(basekcorr['REDSHIFT'][0],basekcorr['ZMAGGIES'][0,these,:])(redshift))
#    gflux = 10.0**(-0.4*(rmag + gr))
#    zflux = 10.0**(-0.4*(rmag - rz))
