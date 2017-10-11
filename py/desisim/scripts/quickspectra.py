from __future__ import absolute_import, division, print_function
import sys, os
import argparse
import time

import numpy as np
import astropy.table
import astropy.time
import astropy.units as u
import astropy.io.fits as pyfits

import desisim.simexp
import desisim.obs
import desisim.io
import desisim.util
from desiutil.log import get_logger
import desispec.io
import desimodel.io
import desitarget
from desispec.spectra import Spectra,spectra_dtype
from desispec.resolution import Resolution
import matplotlib.pyplot as plt

def sim_spectra(wave, flux, program, spectra_filename, obsconditions=None, sourcetype=None, expid=0, seed=0):
    """
    Simulate spectra from an input set of wavelength and flux and writes a FITS file in the Spectra format that can
    be used as input to the redshift fitter.

    Args:
        wave : 1D np.array of wavelength in Angstrom (in vacuum) in observer frame (i.e. redshifted)
        flux : 1D or 2D np.array. 1D array must have same size as wave, 2D array must have shape[1]=wave.size
               flux has to be in units of 10^-17 ergs/s/cm2/A
        spectra_filename : path to output FITS file in the Spectra format
    
    Optional:
        obsconditions : dictionnary of observation conditions with SEEING EXPTIME AIRMASS MOONFRAC MOONALT MOONSEP
        sourcetype : list of string, allowed values are (sky,elg,lrg,qso,bgs,star), type of sources, used for fiber aperture loss , default is star
        expid : this expid number will be saved in the Spectra fibermap
        seed : random seed       
    """
    log = get_logger()
    
    if len(flux.shape)==1 :
        flux=flux.reshape((1,flux.size))
    nspec=flux.shape[0]
    
    log.info("Starting simulation of {} spectra".format(nspec))
    
    if sourcetype is None :        
        sourcetype = np.array(["star" for i in range(nspec)])
    log.debug("sourcetype = {}".format(sourcetype))
    
    tileid  = 0
    telera  = 0
    teledec = 0    
    dateobs = time.gmtime()
    night   = desisim.obs.get_night(utc=dateobs)
    program = program.lower()
        
       
    frame_fibermap = desispec.io.fibermap.empty_fibermap(nspec)    
    frame_fibermap.meta["FLAVOR"]="custom"
    frame_fibermap.meta["NIGHT"]=night
    frame_fibermap.meta["EXPID"]=expid
    
    # add DESI_TARGET 
    tm = desitarget.desi_mask    
    frame_fibermap['DESI_TARGET'][sourcetype=="star"]=tm.STD_FSTAR
    frame_fibermap['DESI_TARGET'][sourcetype=="lrg"]=tm.LRG
    frame_fibermap['DESI_TARGET'][sourcetype=="elg"]=tm.ELG
    frame_fibermap['DESI_TARGET'][sourcetype=="qso"]=tm.QSO
    frame_fibermap['DESI_TARGET'][sourcetype=="sky"]=tm.SKY
    frame_fibermap['DESI_TARGET'][sourcetype=="bgs"]=tm.BGS_ANY
    
    # add dummy TARGETID
    frame_fibermap['TARGETID']=np.arange(nspec).astype(int)
         
    # spectra fibermap has two extra fields : night and expid
    spectra_fibermap = np.zeros(shape=(nspec,), dtype=spectra_dtype())
    for s in range(nspec):
        for tp in frame_fibermap.dtype.fields:
            spectra_fibermap[s][tp] = frame_fibermap[s][tp]
    spectra_fibermap[:]['EXPID'] = expid # needed by spectra
    spectra_fibermap[:]['NIGHT'] = night # needed by spectra
    
    
    if obsconditions is None:
        if program in ['dark', 'lrg', 'qso']:
            obsconditions = desisim.simexp.reference_conditions['DARK']
        elif program in ['elg', 'gray', 'grey']:
            obsconditions = desisim.simexp.reference_conditions['GRAY']
        elif program in ['mws', 'bgs', 'bright']:
            obsconditions = desisim.simexp.reference_conditions['BRIGHT']
        else:
            raise ValueError('unknown program {}'.format(program))
    elif isinstance(obsconditions, str):
        try:
            obsconditions = desisim.simexp.reference_conditions[obsconditions.upper()]
        except KeyError:
            raise ValueError('obsconditions {} not in {}'.format(
                obsconditions.upper(),
                list(desisim.simexp.reference_conditions.keys())))
    
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    ii = (wavemin <= wave) & (wave <= wavemax)

    flux_unit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 )
    
    wave = wave[ii]*u.Angstrom
    flux = flux[:,ii]*flux_unit

    random_state = np.random.RandomState(seed)
    
    sim = desisim.simexp.simulate_spectra(wave, flux, fibermap=frame_fibermap, obsconditions=obsconditions)
    sim.generate_random_noise(random_state)
    
    scale=1e17
    specdata = None

    resolution={}
    for camera in sim.instrument.cameras:
        R = Resolution(camera.get_output_resolution_matrix())
        resolution[camera.name] = np.tile(R.to_fits_array(), [nspec, 1, 1])
        
    for table in sim.camera_output :
        
        wave = table['wavelength'].astype(float)
        flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
        ivar = table['flux_inverse_variance'].T.astype(float)
        
        band  = table.meta['name'].strip()[0]
        
        flux = flux * scale
        ivar = ivar / scale**2
        mask  = np.zeros(flux.shape).astype(int)
        
        spec = Spectra([band], {band : wave}, {band : flux}, {band : ivar}, 
                       resolution_data={band : resolution[band]}, 
                       mask={band : mask}, 
                       fibermap=spectra_fibermap, 
                       meta=None,
                       single=True)
        
        if specdata is None :
            specdata = spec
        else :
            specdata.update(spec)
    
    desispec.io.write_spectra(spectra_filename, specdata)        
    log.info('Wrote '+spectra_filename)
    

def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   description="""Fast simulation of spectra into the final DESI format (Spectra class) that can be directly used as
                                   an input to the redshift fitter (redrock). The input file is an ASCII file with first column the wavelength in A (in vacuum, redshifted), the other columns are treated as spectral flux densities in units of 10^-17 ergs/s/cm2/A.""")

    #- Required
    parser.add_argument('-i','--input', type=str, required=True, help="Input spectra, ASCII or fits")
    parser.add_argument('-o','--out-spectra', type=str, required=True, help="Output spectra")
    #- Optional 
    parser.add_argument('--repeat', type=int, default=1, help="Duplicate the input spectra to have several random realizations")
    
    #- Optional observing conditions to override program defaults
    parser.add_argument('--program', type=str, default="DARK", help="Program (DARK, GRAY or BRIGHT)")
    parser.add_argument('--seeing', type=float, default=None, help="Seeing FWHM [arcsec]")
    parser.add_argument('--airmass', type=float, default=None, help="Airmass")
    parser.add_argument('--exptime', type=float, default=None, help="Exposure time [sec]")
    parser.add_argument('--moonfrac', type=float, default=None, help="Moon illumination fraction; 1=full")
    parser.add_argument('--moonalt', type=float, default=None, help="Moon altitude [degrees]")
    parser.add_argument('--moonsep', type=float, default=None, help="Moon separation to tile [degrees]")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--source-type', type=str, default=None, help="Source type (for fiber loss), among sky,elg,lrg,qso,bgs,star")
    
    
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):

    log = get_logger()
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if args.source_type is not None :
        allowed=["sky","elg","lrg","qso","bgs","star"]
        if not args.source_type in allowed :
            log.error("source type has to be among {}".format(allowed))
            sys.exit(12)
        
    exptime = args.exptime
    if exptime is None :
        exptime = 1000. # sec
    
    #- Generate obsconditions with args.program, then override as needed
    obsconditions = desisim.simexp.reference_conditions[args.program]
    if args.airmass is not None:
        obsconditions['AIRMASS'] = args.airmass
    if args.seeing is not None:
        obsconditions['SEEING'] = args.seeing
    if exptime is not None:
        obsconditions['EXPTIME'] = exptime
    if args.moonfrac is not None:
        obsconditions['MOONFRAC'] = args.moonfrac
    if args.moonalt is not None:
        obsconditions['MOONALT'] = args.moonalt
    if args.moonsep is not None:
        obsconditions['MOONSEP'] = args.moonsep
    
    # ascii version
    isfits=False
    hdulist=None
    try :
        hdulist=pyfits.open(args.input)
        isfits=True
    except (IOError,OSError) :
        pass 
    
    if isfits :
        log.info("Reading an input FITS file")
        if not "WAVELENGTH" in hdulist :
            log.error("need an HDU with EXTNAME='WAVELENGTH' with a 1D array/image of wavelength in A in vacuum")
            sys.exit(12)
        if not "FLUX" in hdulist :
            log.error("need an HDU with EXTNAME='FLUX' with a 1D or 2D array/image of flux in units of 10^-17 ergs/s/cm2/A")
            sys.exit(12)
        input_wave = hdulist["WAVELENGTH"].data
        input_flux = hdulist["FLUX"].data
        if input_wave.size != input_flux.shape[1] :
            log.error("WAVELENGTH size {} != FLUX shape[1] = {} (NAXIS1 in fits)")
        hdulist.close()
    else : 
        # read is ASCII
        try :
            tmp = np.loadtxt(args.input).T
        except (ValueError,TypeError) :
            log.error("could not read ASCII file, need at least two columns, separated by ' ', the first one for wavelength in A in vacuum, the other ones for flux in units of 10^-17 ergs/s/cm2/A, one column per spectrum.")
            log.error("error message : {}".format(sys.exc_info()))
            sys.exit(12)
        
        if tmp.shape[0]<2 :
            log.error("need at least two columns in ASCII file (one for wavelength in A in vacuum, one for flux in units of 10^-17 ergs/s/cm2/A")
            sys.exit(12)
        
        input_wave = tmp[0]
        input_flux = tmp[1:]
    
    if args.repeat>1 :
        input_flux = np.tile(input_flux, (args.repeat,1 ))
        log.info("input flux shape (after repeat) = {}".format(input_flux.shape))
    else :
        log.info("input flux shape = {}".format(input_flux.shape))
    
    sourcetype=args.source_type
    if sourcetype is not None and len(input_flux.shape)>1 :
        nspec=input_flux.shape[0]
        sourcetype=np.array([sourcetype for i in range(nspec)])
    
    sim_spectra(input_wave, input_flux, args.program, obsconditions=obsconditions,spectra_filename=args.out_spectra,seed=args.seed,sourcetype=sourcetype)
    
