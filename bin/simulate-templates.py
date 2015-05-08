#!/usr/bin/env python

"""
Generate simulated galaxy templates.

Give examples...


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
import numpy as np

from optparse import OptionParser

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

from desispec.log import get_logger

from desisim.templates import (EMSpectrum, read_templates)
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

parser.add_option('--nmodel', default=2, type=long,
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

# Check that the right environment variables are set.
envOK = True
for envvar in ['DESI_'+objtype.upper()+'_TEMPLATES']:
    if envvar not in os.environ:
        print('Missing ${} environment variable'.format(envvar))
        envOK = False
if not envOK:
    sys.exit(1)

# Initialize the filter profiles. Need an exception if reading these fails!
gfilt = filter(filtername='decam_g.txt')
rfilt = filter(filtername='decam_r.txt')
zfilt = filter(filtername='decam_z.txt')
w1filt = filter(filtername='wise_w1.txt')

# Read the rest-frame continuum basis spectra.  Need an exception if this
# fails!
baseflux, basewave, basemeta = read_templates(objtype=objtype,continuum=True)
nbase = len(basemeta)

# Split the desired number of models into manageable chunks in case NMODEL is a
# very large number.
nchunk = min(opts.nmodel,10)

nobj = 0
while nobj<opts.nmodel:
    # Choose a random subset of the templates and only keep the ones that
    # satisfy the grz color and [OII] flux cuts.
    these = np.random.randint(0,nbase-1,nchunk)

    # Assign uniform redshift and r-magnitude distributions.
    redshift = np.random.uniform(opts.redshift_range[0],opts.redshift_range[1],nchunk)
    rmag = np.random.uniform(opts.rmag_range[0],opts.rmag_range[1],nchunk)

    # Assume the emission-line priors are uncorrelated.
    oiiihbeta = np.random.uniform(opts.oiiihbeta_range[0],opts.oiiihbeta_range[1],nchunk)
    oiiratio = np.random.normal(opts.oiiratio_meansig[0],opts.oiiratio_meansig[1],nchunk)
    linesigma = np.random.normal(opts.linesigma_meansig[0],opts.linesigma_meansig[1],nchunk)

    d4000 = basemeta['D4000'][these]
    ewoii = 10.0**(np.polyval([1.1074,-4.7338,5.6585],d4000)+ # rest-frame, Angstrom
                   np.random.normal(0.0,0.3)) 

    # The only way to not have to loop here would be to pre-compute the
    # K-corrections on a uniform redshift grid.
    for ii, iobj in enumerate(these):
        zfactor = 1.0+redshift[ii]
        wave = basewave*zfactor

        # Normalize so the spectrum has the right r-band magnitude.
        rnorm = 10.0**(-0.4*rmag[ii])/rfilt.get_maggies(wave,baseflux[iobj,:])
        flux = baseflux[iobj,:]*rnorm # [erg/s/cm2/A, observed]

        rflux = 10.0**(-0.4*(rmag[ii]-22.5))                # nanomaggies
        gflux = gfilt.get_maggies(wave,flux)*10**(0.4*22.5) # nanomaggies
        zflux = zfilt.get_maggies(wave,flux)*10**(0.4*22.5) # nanomaggies

        oiiflux = basemeta['OII_CONTINUUM'][iobj]*ewoii[ii]*rnorm # [erg/s/cm2]

        grzmask = cuts.Fluxes.ELG(gflux=gflux,rflux=rflux,zflux=zflux)
        oiimask = [1] # generalize this
        print(rflux,gflux,zflux,oiiflux,grzmask)

        # Build the final template if the grz color and [OII] flux cuts are
        # satisfied.
        if all(grzmask) and all(oiimask):
            print(nobj, redshift[ii], rmag[ii], d4000[ii], ewoii[ii], oiiflux)
            EM = EMSpectrum(linesigma=linesigma[ii], oiiratio=oiiratio[ii],
                            oiiihbeta=oiiihbeta[ii], oiiflux=oiiflux)
            emwave1 = 10.0**EM.wavelength()
            emflux1 = EM.emlines()
            emflux = resample_flux(wave/zfactor,emwave1,emflux1) # [erg/s/cm2/A, observed]

            outflux = flux + emflux

            # Synthesize photometry.

            plt.clf()
            plt.plot(wave[0::10],outflux[0::10],'r')
            #plt.plot(wave,emflux,'g')
            #plt.plot(emwave1,emflux1,'b')
            plt.xlim([5000,12000])
            plt.ylim([0,outflux.max()])
            plt.show()

            nobj = nobj+1

# Write out or return?  Write these out with 0.2 A binning.
# Add the version number to the metadata header
# Create the EM spectrum just once
# vacuum wavelengths!
