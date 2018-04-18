from __future__ import absolute_import, division, print_function
import sys, os
import argparse
import time

import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits
import multiprocessing

from desiutil.log import get_logger
from desispec.io.util import write_bintable
from desispec.io.fibermap import read_fibermap
from desisim.simexp import reference_conditions
from desisim.templates import SIMQSO
from desisim.scripts.quickspectra import sim_spectra
from desisim.lya_spectra import read_lya_skewers , apply_lya_transmission
from desispec.interpolation import resample_flux
from desimodel.io import load_pixweight
from desimodel import footprint

import matplotlib.pyplot as plt

def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   description="""Fast simulation of QSO Lya spectra into the final DESI format (Spectra class) that can be directly used as
                                   an input to the redshift fitter (redrock) or correlation function code (picca). The input file is a Lya transmission skewer fits file which format is described in https://desi.lbl.gov/trac/wiki/LymanAlphaWG/LyaSpecSim.""")

    #- Required
    parser.add_argument('-i','--infile', type=str, nargs= "*", required=True, help="Input skewer healpix fits file(s)")
    parser.add_argument('-o','--outfile', type=str, required=False, help="Output spectra (only used if single input file)")
    parser.add_argument('--outdir', type=str, default=".", required=False, help="Output directory")
    
    #- Optional observing conditions to override program defaults
    parser.add_argument('--program', type=str, default="DARK", help="Program (DARK, GRAY or BRIGHT)")
    parser.add_argument('--seeing', type=float, default=None, help="Seeing FWHM [arcsec]")
    parser.add_argument('--airmass', type=float, default=None, help="Airmass")
    parser.add_argument('--exptime', type=float, default=None, help="Exposure time [sec]")
    parser.add_argument('--moonfrac', type=float, default=None, help="Moon illumination fraction; 1=full")
    parser.add_argument('--moonalt', type=float, default=None, help="Moon altitude [degrees]")
    parser.add_argument('--moonsep', type=float, default=None, help="Moon separation to tile [degrees]")
    parser.add_argument('--seed', type=int, default=None, required = False, help="Random seed")
    parser.add_argument('--skyerr', type=float, default=0.0, help="Fractional sky subtraction error")
    parser.add_argument('--norm-filter', type=str, default="decam2014-g", help="Broadband filter for normalization")
    parser.add_argument('--nmax', type=int, default=None, help="Max number of QSO per input file, for debugging")
    parser.add_argument('--downsampling', type=float, default=None,help="fractional random down-sampling (value between 0 and 1)")
    parser.add_argument('--zmin', type=float, default=0,help="Min redshift")
    parser.add_argument('--zmax', type=float, default=10,help="Max redshift")
    parser.add_argument('--wmin', type=float, default=3500,help="Min wavelength (obs. frame)")
    parser.add_argument('--wmax', type=float, default=10000,help="Max wavelength (obs. frame)")
    parser.add_argument('--dwave', type=float, default=0.2,help="Internal wavelength step (don't change this)")
    parser.add_argument('--nproc', type=int, default=1,help="number of processors to run faster")
    parser.add_argument('--zbest', action = "store_true" ,help="add a zbest file per spectrum with the truth")
    parser.add_argument('--overwrite', action = "store_true" ,help="rerun if spectra exists (default is skip)")
    parser.add_argument('--target-selection', action = "store_true" ,help="apply QSO target selection cuts to the simulated quasars")
    parser.add_argument('--desi-footprint', action = "store_true" ,help="select QSOs in DESI footprint")
    
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def simulate_one_healpix(ifilename,args,model,obsconditions,decam_and_wise_filters,footprint_healpix_weight,footprint_healpix_nside,seed) :
    
    log = get_logger()

    # set seed now
    # we need a seed per healpix because
    # the spectra simulator REQUIRES a seed
    np.random.seed(seed)
    
    healpix=0
    nside=0
    vals = os.path.basename(ifilename).split(".")[0].split("-")
    if len(vals)<3 :
        log.error("Cannot guess nside and healpix from filename {}".format(ifilename))
        raise ValueError("Cannot guess nside and healpix from filename {}".format(ifilename))
    try :
        healpix=int(vals[-1])
        nside=int(vals[-2])
    except ValueError:
        raise ValueError("Cannot guess nside and healpix from filename {}".format(ifilename))
    
    zbest_filename = None
    if args.outfile :
        ofilename = args.outfile
    else :
        ofilename = os.path.join(args.outdir,"{}/{}/spectra-{}-{}.fits".format(healpix//100,healpix,nside,healpix))
    pixdir = os.path.dirname(ofilename)
        
    if not args.overwrite :
        # check whether output exists or not
        
        if args.zbest :
            zbest_filename = os.path.join(pixdir,"zbest-{}-{}.fits".format(nside,healpix))
            if os.path.isfile(ofilename) and os.path.isfile(zbest_filename) :
                log.info("skip existing {} and {}".format(ofilename,zbest_filename))
                return
        else : # only test spectra file
            if os.path.isfile(ofilename) :
                log.info("skip existing {}".format(ofilename))
                return

    log.info("Read skewers in {}, random seed = {}".format(ifilename,seed))
    trans_wave, transmission, metadata = read_lya_skewers(ifilename)
    ok = np.where(( metadata['Z'] >= args.zmin ) & (metadata['Z'] <= args.zmax ))[0]
    transmission = transmission[ok]
    metadata = metadata[:][ok]
    
    # create quasars
    
    if args.desi_footprint :
        footprint_healpix = footprint.radec2pix(footprint_healpix_nside, metadata["RA"], metadata["DEC"])
        selection = np.where(footprint_healpix_weight[footprint_healpix]>0.99)[0]
        log.info("Select QSOs in DESI footprint {} -> {}".format(transmission.shape[0],selection.size))
        if selection.size == 0 :
            log.warning("No intersection with DESI footprint")
            return
        transmission = transmission[selection]
        metadata = metadata[:][selection]
    

    nqso=transmission.shape[0]
    if args.downsampling is not None :
        if args.downsampling <= 0 or  args.downsampling > 1 :
           log.error("Down sampling fraction={} must be between 0 and 1".format(args.downsampling))
           raise ValueError("Down sampling fraction={} must be between 0 and 1".format(args.downsampling))
        indices = np.where(np.random.uniform(size=nqso)<args.downsampling)[0]
        if indices.size == 0 : 
            log.warning("Down sampling from {} to 0 (by chance I presume)".format(nqso))
            return
        transmission = transmission[indices]
        metadata = metadata[:][indices]
        nqso = transmission.shape[0]

    if args.nmax is not None :
        if args.nmax < nqso :
            log.info("Limit number of QSOs from {} to nmax={} (random subsample)".format(nqso,args.nmax))
            # take a random subsample
            indices = (np.random.uniform(size=args.nmax)*nqso).astype(int)
            transmission = transmission[indices]
            metadata = metadata[:][indices]
            nqso = args.nmax

    log.info("Simulate {} QSOs".format(nqso))
    tmp_qso_flux, tmp_qso_wave, meta = model.make_templates(
        nmodel=nqso, redshift=metadata['Z'], 
        lyaforest=False, nocolorcuts=True, noresample=True, seed = seed)

    log.info("Resample to transmission wavelength grid")
    # because we don't want to alter the transmission field with resampling here
    qso_flux=np.zeros((tmp_qso_flux.shape[0],trans_wave.size))
    for q in range(tmp_qso_flux.shape[0]) :
        qso_flux[q]=np.interp(trans_wave,tmp_qso_wave,tmp_qso_flux[q])
    tmp_qso_flux = qso_flux
    tmp_qso_wave = trans_wave

    log.info("Apply lya")
    tmp_qso_flux = apply_lya_transmission(tmp_qso_wave,tmp_qso_flux,trans_wave,transmission)

    if args.target_selection :
        log.info("Compute QSO magnitudes for target selection")
        maggies = decam_and_wise_filters.get_ab_maggies(
            1e-17 * tmp_qso_flux, tmp_qso_wave.copy(), mask_invalid=True)
        for band, filt in zip( ('FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'),
                               ('decam2014-g', 'decam2014-r', 'decam2014-z',
                                'wise2010-W1', 'wise2010-W2') ):
            meta[band] = np.ma.getdata(1e9 * maggies[filt]) # nanomaggies
        isqso = isQSO_colors(gflux=meta['FLUX_G'], rflux=meta['FLUX_R'], zflux=meta['FLUX_Z'],
                           w1flux=meta['FLUX_W1'], w2flux=meta['FLUX_W2'])
        log.info("Target selection: {}/{} QSOs selected".format(np.sum(isqso),nqso))
        selection=np.where(isqso)[0]
        if selection.size==0 : return
        tmp_qso_flux = tmp_qso_flux[selection]
        metadata     = metadata[:][selection]
        meta         = meta[:][selection]
        nqso         = selection.size

    log.info("Resample to a linear wavelength grid (needed by DESI sim.)")
    # we need a linear grid. for this resampling we take care of integrating in bins
    # we do not do a simple interpolation
    qso_wave=np.linspace(args.wmin,args.wmax,int((args.wmax-args.wmin)/args.dwave)+1)
    qso_flux=np.zeros((tmp_qso_flux.shape[0],qso_wave.size))
    for q in range(tmp_qso_flux.shape[0]) :
        qso_flux[q]=resample_flux(qso_wave,tmp_qso_wave,tmp_qso_flux[q])

    log.info("Simulate DESI observation and write output file")
    pixdir = os.path.dirname(ofilename)
    if len(pixdir)>0 :
        if not os.path.isdir(pixdir) :
            log.info("Creating dir {}".format(pixdir))
            os.makedirs(pixdir)
    
    if "MOCKID" in metadata.dtype.names :
        #log.warning("Using MOCKID as TARGETID")
        targetid=np.array(metadata["MOCKID"]).astype(int)
    elif "ID" in metadata.dtype.names :
        log.warning("Using ID as TARGETID")
        targetid=np.array(metadata["ID"]).astype(int)
    else :
        log.warning("No TARGETID")
        targetid=None

    log.warning("Assuming the healpix scheme is 'NESTED'")
    meta={"HPXNSIDE":nside,"HPXPIXEL":healpix, "HPXNEST":True}
    
    sim_spectra(qso_wave,qso_flux, args.program, obsconditions=obsconditions,spectra_filename=ofilename,sourcetype="qso", skyerr=args.skyerr,ra=metadata["RA"],dec=metadata["DEC"],targetid=targetid,meta=meta,seed=seed)

    if args.zbest :
        log.info("Read fibermap")
        fibermap = read_fibermap(ofilename)

        

        log.info("Writing a zbest file {}".format(zbest_filename))
        columns = [
            ('CHI2', 'f8'),
            ('COEFF', 'f8' , (4,)),
            ('Z', 'f8'),
            ('ZERR', 'f8'),
            ('ZWARN', 'i8'),
            ('SPECTYPE', (str,96)),
            ('SUBTYPE', (str,16)),
            ('TARGETID', 'i8'),
            ('DELTACHI2', 'f8'),
            ('BRICKNAME', (str,8))]
        zbest = Table(np.zeros(nqso, dtype=columns))
        zbest["CHI2"][:]      = 0.
        zbest["Z"]            = metadata['Z']
        zbest["ZERR"][:]      = 0.
        zbest["ZWARN"][:]     = 0
        zbest["SPECTYPE"][:]  = "QSO"
        zbest["SUBTYPE"][:]   = ""
        zbest["TARGETID"]     = fibermap["TARGETID"]
        zbest["DELTACHI2"][:] = 25.

        hzbest = pyfits.convenience.table_to_hdu(zbest); hzbest.name="ZBEST"
        hfmap  = pyfits.convenience.table_to_hdu(fibermap);  hfmap.name="FIBERMAP"

        hdulist =pyfits.HDUList([pyfits.PrimaryHDU(),hzbest,hfmap])
        hdulist.writeto(zbest_filename, clobber=True)
        hdulist.close() # see if this helps with memory issue




def _func(arg) :
    """ Used for multiprocessing.Pool """
    return simulate_one_healpix(**arg)

def main(args=None):

    log = get_logger()
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)
    
    if args.outfile is not None and len(args.infile)>1 :
        log.error("Cannot specify single output file with multiple inputs, use --outdir option instead")
        return 1
        
    if not os.path.isdir(args.outdir) :
        log.info("Creating dir {}".format(args.outdir))
        os.makedirs(args.outdir)
    
    exptime = args.exptime
    if exptime is None :
        exptime = 1000. # sec
    
    #- Generate obsconditions with args.program, then override as needed
    obsconditions = reference_conditions[args.program.upper()]
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
    
    log.info("Load SIMQSO model")
    model=SIMQSO(normfilter=args.norm_filter,nproc=1)
    
    decam_and_wise_filters = None
    if args.target_selection :
        log.info("Load DeCAM and WISE filters for target selection sim.")
        
        from speclite import filters
        from desitarget.cuts import isQSO_colors

        decam_and_wise_filters = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                         'wise2010-W1', 'wise2010-W2')
        
    footprint_healpix_weight = None
    footprint_healpix_nside  = None
    if args.desi_footprint :
        if not 'DESIMODEL' in os.environ :
            log.error("To apply DESI footprint, I need the DESIMODEL variable to find the file $DESIMODEL/data/footprint/desi-healpix-weights.fits")
            sys.exit(1)
        footprint_filename=os.path.join(os.environ['DESIMODEL'],'data','footprint','desi-healpix-weights.fits')
        if not os.path.isfile(footprint_filename): 
            log.error("Cannot find $DESIMODEL/data/footprint/desi-healpix-weights.fits")
            sys.exit(1)
        pixmap=pyfits.open(footprint_filename)[0].data
        footprint_healpix_nside=256 # same resolution as original map so we don't loose anything
        footprint_healpix_weight = load_pixweight(footprint_healpix_nside, pixmap=pixmap)
        
    if args.seed is not None :
        np.random.seed(args.seed)
    
    # seeds for each healpix are themselves random numbers
    seeds = np.random.randint(2**32, size=len(args.infile))
    
    if args.nproc > 1 :
        func_args = [ {"ifilename":filename , \
                       "args":args, "model":model , \
                       "obsconditions":obsconditions , \
                       "decam_and_wise_filters": decam_and_wise_filters , \
                       "footprint_healpix_weight": footprint_healpix_weight , \
                       "footprint_healpix_nside": footprint_healpix_nside , \
                       "seed":seeds[i]
                   } for i,filename in enumerate(args.infile) ]
        pool = multiprocessing.Pool(args.nproc)
        pool.map(_func, func_args)
    else :
        for i,ifilename in enumerate(args.infile) : 
            simulate_one_healpix(ifilename,args,model,obsconditions,decam_and_wise_filters,footprint_healpix_weight,footprint_healpix_nside,seed=seeds[i])
    
        
