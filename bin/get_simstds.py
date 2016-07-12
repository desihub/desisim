#!/usr/bin/env python

"""
Cheat by getting standard star info directly from input simulations
rather than fitting them from the data.

Usage: get_simstds.py night expid [options]

Options:
  -h, --help            show this help message and exit
  -s SPECTROID, --spectroid=SPECTROID
                        spectrograph ID [0-9, default 0]
  -o OUTFILE, --outfile=OUTFILE
                        output filename
"""

import sys
import os
import os.path
from astropy.io import fits
import numpy as np

def get_simstds(simspecfile, spectroid=0):
    """
    Extract the true standard star flux from the input simspec files.
    
    Args:
        night : string YEARMMDD
        expid : int or string exposure ID
        spectroid : optional spectrograph ID [default 0]
        
    Returns tuple of:
        stdfiber : 1D array of fiberid [0-499] of standard stars
        wave : 1D array of wavelength sampling
        flux : 2D array [nstd, nwave] flux in ergs/s/cm^2/A
    """
    #- Read inputs
    ii = slice(500*spectroid, 500*(spectroid+1))
    hdr = fits.getheader(simspecfile, 0)
    flux = fits.getdata(simspecfile, 'FLUX')[ii]
    meta = fits.getdata(simspecfile, 'METADATA')[ii]
    wave = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(hdr['NAXIS1'])

    #- Which ones are standard stars?
    stdfiber = np.where(meta['OBJTYPE'] == 'STD')[0]

    return stdfiber, wave, flux[stdfiber]

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage = "%prog [night expid] [options]")
    parser.add_option("-s", "--spectroid", type=int,  default=0, help="spectrograph ID [0-9, default 0]")
    parser.add_option("-i", "--inspec", type=str, help="input simspec filename")
    parser.add_option("-o", "--outfile", type=str, help="output filename")

    opts, args = parser.parse_args()

    if opts.inspec is None:
        night = args[0]
        expid = int(args[1])

        #- Make sure we have what we need as input
        assert 'DESI_SPECTRO_SIM' in os.environ
        assert 'PIXPROD' in os.environ
        if isinstance(expid, str):
            expid = int(expid)
        
        #- Where are the input files?
        simpath = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'), night)
        opts.inspec = '{}/simspec-{:08d}.fits'.format(simpath, expid)

    stdfiber, wave, flux = get_simstds(opts.inspec, opts.spectroid)
    
    if opts.outfile is None:
        assert 'DESI_SPECTRO_REDUX' in os.environ
        assert 'SPECPROD' in os.environ

        outdir = os.path.join(os.getenv('DESI_SPECTRO_REDUX'), os.getenv('SPECPROD'),
            'exposures', night, '{:08d}'.format(expid))
        opts.outfile = '{}/stdflux-sp{}-{:08d}.fits'.format(outdir, opts.spectroid, expid)
        print opts.outfile
    
    fits.writeto(opts.outfile, flux, clobber=True)
    fits.append(opts.outfile, wave)
    fits.append(opts.outfile, stdfiber)
    



