from __future__ import absolute_import, division, print_function
import sys, os
import argparse
import time

import numpy as np
import astropy.table
import astropy.time
import astropy.units as u
import astropy.io.fits as pyfits

import desisim.specsim
import desisim.simexp
import desisim.obs
import desisim.io
import desisim.util
from desiutil.log import get_logger
import desispec.io
import desispec.io.util
import desimodel.io
import desitarget
from desispec.spectra import Spectra
from desispec.resolution import Resolution

def sim_spectra(wave, flux, program, spectra_filename, obsconditions=None,
                sourcetype=None, targetid=None, redshift=None, expid=0, seed=0, skyerr=0.0, ra=None, dec=None, meta=None, fibermap_columns=None, fullsim=False,use_poisson=True, specsim_config_file="desi"):
    """
    Simulate spectra from an input set of wavelength and flux and writes a FITS file in the Spectra format that can
    be used as input to the redshift fitter.

    Args:
        wave : 1D np.array of wavelength in Angstrom (in vacuum) in observer frame (i.e. redshifted)
        flux : 1D or 2D np.array. 1D array must have same size as wave, 2D array must have shape[1]=wave.size
               flux has to be in units of 10^-17 ergs/s/cm2/A
        spectra_filename : path to output FITS file in the Spectra format
        program : dark, lrg, qso, gray, grey, elg, bright, mws, bgs
            ignored if obsconditions is not None
    
    Optional:
        obsconditions : dictionnary of observation conditions with SEEING EXPTIME AIRMASS MOONFRAC MOONALT MOONSEP
        sourcetype : list of string, allowed values are (sky,elg,lrg,qso,bgs,star), type of sources, used for fiber aperture loss , default is star
        targetid : list of targetids for each target. default of None has them generated as str(range(nspec))
        redshift : list/array with each index being the redshifts for that target
        expid : this expid number will be saved in the Spectra fibermap
        seed : random seed
        skyerr : fractional sky subtraction error
        ra : numpy array with targets RA (deg)
        dec : numpy array with targets Dec (deg)
        meta : dictionnary, saved in primary fits header of the spectra file 
        fibermap_columns : add these columns to the fibermap
        fullsim : if True, write full simulation data in extra file per camera
        use_poisson : if False, do not use numpy.random.poisson to simulate the Poisson noise. This is useful to get reproducible random realizations.
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
    tm = desitarget.targetmask.desi_mask
    frame_fibermap['DESI_TARGET'][sourcetype=="star"]=tm.STD_FAINT
    frame_fibermap['DESI_TARGET'][sourcetype=="lrg"]=tm.LRG
    frame_fibermap['DESI_TARGET'][sourcetype=="elg"]=tm.ELG
    frame_fibermap['DESI_TARGET'][sourcetype=="qso"]=tm.QSO
    frame_fibermap['DESI_TARGET'][sourcetype=="sky"]=tm.SKY
    frame_fibermap['DESI_TARGET'][sourcetype=="bgs"]=tm.BGS_ANY
    
    
    if fibermap_columns is not None :
        for k in fibermap_columns.keys() :
            frame_fibermap[k] = fibermap_columns[k]
        
    if targetid is None:
        targetid = np.arange(nspec).astype(int)
        
    # add TARGETID
    frame_fibermap['TARGETID'] = targetid
         
    # spectra fibermap has two extra fields : night and expid
    # This would be cleaner if desispec would provide the spectra equivalent
    # of desispec.io.empty_fibermap()
    spectra_fibermap = desispec.io.empty_fibermap(nspec)
    spectra_fibermap = desispec.io.util.add_columns(spectra_fibermap,
                       ['NIGHT', 'EXPID', 'TILEID'],
                       [np.int32(night), np.int32(expid), np.int32(tileid)],
                       )

    for s in range(nspec):
        for tp in frame_fibermap.dtype.fields:
            spectra_fibermap[s][tp] = frame_fibermap[s][tp]
 
    if ra is not None :
        spectra_fibermap["TARGET_RA"] = ra
        spectra_fibermap["FIBER_RA"]    = ra
    if dec is not None :
        spectra_fibermap["TARGET_DEC"] = dec
        spectra_fibermap["FIBER_DEC"]    = dec
            
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
    try:
        params = desimodel.io.load_desiparams()
        wavemin = params['ccd']['b']['wavemin']
        wavemax = params['ccd']['z']['wavemax']
    except KeyError:
        wavemin = desimodel.io.load_throughput('b').wavemin
        wavemax = desimodel.io.load_throughput('z').wavemax

    if wave[0] > wavemin:
        log.warning('Minimum input wavelength {}>{}; padding with zeros'.format(
                wave[0], wavemin))
        dwave = wave[1] - wave[0]
        npad = int((wave[0] - wavemin)/dwave + 1)
        wavepad = np.arange(npad) * dwave
        wavepad += wave[0] - dwave - wavepad[-1]
        fluxpad = np.zeros((flux.shape[0], len(wavepad)), dtype=flux.dtype)
        wave = np.concatenate([wavepad, wave])
        flux = np.hstack([fluxpad, flux])
        assert flux.shape[1] == len(wave)
        assert np.allclose(dwave, np.diff(wave))
        assert wave[0] <= wavemin

    if wave[-1] < wavemax:
        log.warning('Maximum input wavelength {}<{}; padding with zeros'.format(
                wave[-1], wavemax))
        dwave = wave[-1] - wave[-2]
        npad = int( (wavemax - wave[-1])/dwave + 1 )
        wavepad = wave[-1] + dwave + np.arange(npad)*dwave
        fluxpad = np.zeros((flux.shape[0], len(wavepad)), dtype=flux.dtype)
        wave = np.concatenate([wave, wavepad])
        flux = np.hstack([flux, fluxpad])
        assert flux.shape[1] == len(wave)
        assert np.allclose(dwave, np.diff(wave))
        assert wavemax <= wave[-1]

    ii = (wavemin <= wave) & (wave <= wavemax)

    flux_unit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 )
    
    wave = wave[ii]*u.Angstrom
    flux = flux[:,ii]*flux_unit

    sim = desisim.simexp.simulate_spectra(wave, flux, fibermap=frame_fibermap,
        obsconditions=obsconditions, redshift=redshift, seed=seed,
        psfconvolve=True, specsim_config_file=specsim_config_file)

    random_state = np.random.RandomState(seed)
    #sim.generate_random_noise(random_state,use_poisson=use_poisson)
    sim.generate_random_noise(random_state)
    
    scale=1e17
    specdata = None

    resolution={}
    for camera in sim.instrument.cameras:
        R = Resolution(camera.get_output_resolution_matrix())
        resolution[camera.name] = np.tile(R.to_fits_array(), [nspec, 1, 1])

    skyscale = skyerr * random_state.normal(size=sim.num_fibers)

    if fullsim :
        for table in sim.camera_output :
            band  = table.meta['name'].strip()[0]
            table_filename=spectra_filename.replace(".fits","-fullsim-{}.fits".format(band))
            table.write(table_filename,format="fits",overwrite=True)
            print("wrote",table_filename)

    for table in sim.camera_output :
        
        wave = table['wavelength'].astype(float)
        flux = (table['observed_flux']+table['random_noise_electrons']*table['flux_calibration']).T.astype(float)
        if np.any(skyscale):
            flux += ((table['num_sky_electrons']*skyscale)*table['flux_calibration']).T.astype(float)

        ivar = table['flux_inverse_variance'].T.astype(float)
        
        band  = table.meta['name'].strip()[0]
        
        flux = flux * scale
        ivar = ivar / scale**2
        mask  = np.zeros(flux.shape).astype(int)
        
        spec = Spectra([band], {band : wave}, {band : flux}, {band : ivar}, 
                       resolution_data={band : resolution[band]}, 
                       mask={band : mask}, 
                       fibermap=spectra_fibermap, 
                       meta=meta,
                       single=True)
        
        if specdata is None :
            specdata = spec
        else :
            specdata.update(spec)
    
    desispec.io.write_spectra(spectra_filename, specdata)        
    log.info('Wrote '+spectra_filename)
    
    # need to clear the simulation buffers that keeps growing otherwise
    # because of a different number of fibers each time ...
    desisim.specsim._simulators.clear()
    desisim.specsim._simdefaults.clear()


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
    parser.add_argument('--skyerr', type=float, default=0.0, help="Fractional sky subtraction error")
    parser.add_argument('--fullsim',action='store_true',help="write full simulation data in extra file per camera, for debugging")

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
    obsconditions = desisim.simexp.reference_conditions[args.program.upper()]
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
        if 'WAVELENGTH' in hdulist:
            input_wave = hdulist["WAVELENGTH"].data
        elif "WAVE" in hdulist:
            input_wave = hdulist["WAVE"].data
        else:
            log.error("need an HDU with EXTNAME='WAVELENGTH' with a 1D array/image of wavelength in A in vacuum")
            sys.exit(12)
        if not "FLUX" in hdulist :
            log.error("need an HDU with EXTNAME='FLUX' with a 1D or 2D array/image of flux in units of 10^-17 ergs/s/cm2/A")
            sys.exit(12)
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
    
    sim_spectra(input_wave, input_flux, args.program, obsconditions=obsconditions,
        spectra_filename=args.out_spectra,seed=args.seed,sourcetype=sourcetype,
        skyerr=args.skyerr,fullsim=args.fullsim)
    
