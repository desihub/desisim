from __future__ import absolute_import, division, print_function

import sys, os
import argparse
import time

import numpy as np
from astropy.table import Table,Column
import astropy.io.fits as pyfits
import multiprocessing

from desiutil.log import get_logger
from desispec.io.util import write_bintable
from desispec.io.fibermap import read_fibermap
from desisim.simexp import reference_conditions
from desisim.templates import SIMQSO
from desisim.scripts.quickspectra import sim_spectra
from desisim.lya_spectra import read_lya_skewers , apply_lya_transmission, apply_metals_transmission
from desisim.dla import dla_spec,insert_dlas
from desisim.bal import BAL
from desispec.interpolation import resample_flux
from desimodel.io import load_pixweight
from desimodel import footprint
from speclite import filters
from desitarget.cuts import isQSO_colors

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
    parser.add_argument('--mags', action = "store_true" ,help="compute and write the QSO mags in the fibermap")
    parser.add_argument('--desi-footprint', action = "store_true" ,help="select QSOs in DESI footprint")
    parser.add_argument('--metals', type=str, default=None, required=False, help = "list of metal to get the transmission from, if 'all' runs on all metals", nargs='*')

    #- Optional arguments to include dla

    parser.add_argument('--dla',type=str,required=False, help="Add DLA to simulated spectra either randonmly (--dla random) or from transmision file (--dla file)")
    parser.add_argument('--balprob',type=float,required=False, help="To add BAL features with the specified probability (e.g --balprob 0.5). Expect a number between 0 and 1 ")    
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
    

    # read the header of the tranmission file to find the healpix pixel number, nside
    # and if we are lucky the scheme.
    # if this fails, try to guess it from the filename (for backward compatibility)
    healpix=-1
    nside=-1
    hpxnest=True
    
    hdulist=pyfits.open(ifilename)
    if "METADATA" in hdulist :
        head=hdulist["METADATA"].header
        for k in ["HPXPIXEL","PIXNUM"] :
            if k in head :
                healpix=int(head[k])
                log.info("healpix={}={}".format(k,healpix))
                break
        for k in ["HPXNSIDE","NSIDE"] :
            if k in head :
                nside=int(head[k])
                log.info("nside={}={}".format(k,nside))
                break
        for k in ["HPXNEST","NESTED","SCHEME"] :
            if k in head :
                if k == "SCHEME" : 
                    hpxnest=(head[k]=="NEST")
                else :
                    hpxnest=bool(head[k])
                log.info("hpxnest from {} = {}".format(k,hpxnest))
                break
    if healpix >= 0 and nside < 0 :
        log.error("Read healpix in header but not nside.")
        raise ValueError("Read healpix in header but not nside.")
    
    if healpix < 0 : 
        vals = os.path.basename(ifilename).split(".")[0].split("-")
        if len(vals)<3 :
            log.error("Cannot guess nside and healpix from filename {}".format(ifilename))
            raise ValueError("Cannot guess nside and healpix from filename {}".format(ifilename))
        try :
            healpix=int(vals[-1])
            nside=int(vals[-2])
        except ValueError:
            raise ValueError("Cannot guess nside and healpix from filename {}".format(ifilename))
        log.warning("Guessed healpix and nside from filename, assuming the healpix scheme is 'NESTED'")
    

    zbest_filename = None
    if args.outfile :
        ofilename = args.outfile
    else :
        ofilename = os.path.join(args.outdir,"{}/{}/spectra-{}-{}.fits".format(healpix//100,healpix,nside,healpix))
    pixdir = os.path.dirname(ofilename)
    
    if args.zbest :
        zbest_filename = os.path.join(pixdir,"zbest-{}-{}.fits".format(nside,healpix))
 
    if not args.overwrite :
        # check whether output exists or not
        if args.zbest :
            if os.path.isfile(ofilename) and os.path.isfile(zbest_filename) :
                log.info("skip existing {} and {}".format(ofilename,zbest_filename))
                return
        else : # only test spectra file
            if os.path.isfile(ofilename) :
                log.info("skip existing {}".format(ofilename))
                return

    log.info("Read skewers in {}, random seed = {}".format(ifilename,seed))

 ##ALMA: It reads only the skewers only if there are no DLAs or if they are added randomly. 
    if(not args.dla or args.dla=='random'):
       trans_wave, transmission, metadata = read_lya_skewers(ifilename)
       ok = np.where(( metadata['Z'] >= args.zmin ) & (metadata['Z'] <= args.zmax ))[0]
       transmission = transmission[ok]
       metadata = metadata[:][ok]
 ##ALMA:Added to read dla_info

    elif(args.dla=='file'):
       log.info("Read DLA information in {}".format(ifilename))
       trans_wave, transmission, metadata,dla_info= read_lya_skewers(ifilename,dla_='TRUE')  
       ok = np.where(( metadata['Z'] >= args.zmin ) & (metadata['Z'] <= args.zmax ))[0]
       transmission = transmission[ok]
       metadata = metadata[:][ok]
    else:
       log.error('Not a valid option to add DLAs. Valid options are "random" or "file"')
       sys.exit(1)

    if args.dla:
        dla_NHI, dla_z,dla_id = [],[], []
        dla_filename=os.path.join(pixdir,"dla-{}-{}.fits".format(nside,healpix))
   
    
    
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


##ALMA:added to set transmission to 1 for z>zqso, this can be removed when transmission is corrected. 
    for ii in range(len(metadata)):       
        transmission[ii][trans_wave>1215.67*(metadata[ii]['Z']+1)]=1.0

    if(args.dla=='file'):
        log.info('Adding DLAs from transmision file')
        min_trans_wave=np.min(trans_wave/1215.67 - 1)
        for ii in range(len(metadata)):
            if min_trans_wave < metadata[ii]['Z']: 
               idd=metadata['MOCKID'][ii]
               dlas=dla_info[dla_info['MOCKID']==idd]
               dlass=[]
               for i in range(len(dlas)):
##Adding only dlas between zqso and 1.95, check again for the next version of London mocks...  
                   if (dlas[i]['Z_DLA']< metadata[ii]['Z']) and (dlas[i]['Z_DLA']> 1.95) :
                      dlass.append(dict(z=dlas[i]['Z_DLA']+dlas[i]['DZ_DLA'],N=dlas[i]['N_HI_DLA']))
               if len(dlass)>0:
                  dla_model=dla_spec(trans_wave,dlass)
                  transmission[ii]=dla_model*transmission[ii]
                  dla_z+=[idla['z'] for idla in dlass]
                  dla_NHI+=[idla['N'] for idla in dlass]
                  dla_id+=[idd]*len(dlass)

    elif(args.dla=='random'):
        log.info('Adding DLAs randomly')
        min_trans_wave=np.min(trans_wave/1215.67 - 1)
        for ii in range(len(metadata)):
            if min_trans_wave < metadata[ii]['Z']: 
               idd=metadata['MOCKID'][ii]
               dlass, dla_model = insert_dlas(trans_wave, metadata[ii]['Z'])
               if len(dlass)>0:
                  transmission[ii]=dla_model*transmission[ii]
                  dla_z+=[idla['z'] for idla in dlass]
                  dla_NHI+=[idla['N'] for idla in dlass]
                  dla_id+=[idd]*len(dlass)    


    if args.dla:
       if len(dla_id)>0:
          dla_meta=Table()
          dla_meta['NHI']=dla_NHI
          dla_meta['z']=dla_z
          dla_meta['ID']=dla_id

    if args.nmax is not None :
        if args.nmax < nqso :
            log.info("Limit number of QSOs from {} to nmax={} (random subsample)".format(nqso,args.nmax))
            # take a random subsample
            indices = (np.random.uniform(size=args.nmax)*nqso).astype(int)
            transmission = transmission[indices]
            metadata = metadata[:][indices]
            nqso = args.nmax

            if args.dla:
               dla_meta=dla_meta[:][dla_meta['ID']==metadata['MOCKID']]
            
    if args.target_selection or args.mags :
        wanted_min_wave = 3329. # needed to compute magnitudes for decam2014-r (one could have trimmed the transmission file ...)
        wanted_max_wave = 55501. # needed to compute magnitudes for wise2010-W2
        
        if trans_wave[0]>wanted_min_wave :
            log.info("Increase wavelength range from {}:{} to {}:{} to compute magnitudes".format(int(trans_wave[0]),int(trans_wave[-1]),int(wanted_min_wave),int(trans_wave[-1])))
            # pad with zeros at short wavelength because we assume transmission = 0
            # and we don't need any wavelength resolution here
            new_trans_wave = np.append([wanted_min_wave,trans_wave[0]-0.01],trans_wave)
            new_transmission = np.zeros((transmission.shape[0],new_trans_wave.size))
            new_transmission[:,2:] = transmission
            trans_wave   = new_trans_wave
            transmission = new_transmission
                    
        if trans_wave[-1]<wanted_max_wave :
            log.info("Increase wavelength range from {}:{} to {}:{} to compute magnitudes".format(int(trans_wave[0]),int(trans_wave[-1]),int(trans_wave[0]),int(wanted_max_wave)))
            # pad with ones at long wavelength because we assume transmission = 1
            coarse_dwave = 2. # we don't care about resolution, we just need a decent QSO spectrum, there is no IGM transmission in this range
            n = int((wanted_max_wave-trans_wave[-1])/coarse_dwave)+1
            new_trans_wave = np.append(trans_wave,np.linspace(trans_wave[-1]+coarse_dwave,trans_wave[-1]+coarse_dwave*(n+1),n))
            new_transmission = np.ones((transmission.shape[0],new_trans_wave.size))
            new_transmission[:,:trans_wave.size] = transmission
            trans_wave   = new_trans_wave
            transmission = new_transmission
            
            
    log.info("Simulate {} QSOs".format(nqso))
    tmp_qso_flux, tmp_qso_wave, meta = model.make_templates(
        nmodel=nqso, redshift=metadata['Z'], 
        lyaforest=False, nocolorcuts=True, noresample=True, seed = seed)

    ##To add BALs to be checked by Luz and Jaime
    if (args.balprob<=1. and args.balprob >0):
       log.info("Adding BALs with probability {}".format(args.balprob))
       bal=BAL()
       tmp_qso_flux,meta_bal=bal.insert_bals(tmp_qso_wave,tmp_qso_flux, metadata['Z'], balprob=args.balprob,seed=seed)
    else:
       log.error("Probability to add BALs is not between 0 and 1")
       sys.exit(1)
   
    log.info("Resample to transmission wavelength grid")
    # because we don't want to alter the transmission field with resampling here
    qso_flux=np.zeros((tmp_qso_flux.shape[0],trans_wave.size))
    for q in range(tmp_qso_flux.shape[0]) :
        qso_flux[q]=np.interp(trans_wave,tmp_qso_wave,tmp_qso_flux[q])
    tmp_qso_flux = qso_flux
    tmp_qso_wave = trans_wave
        
    log.info("Apply lya")
    tmp_qso_flux = apply_lya_transmission(tmp_qso_wave,tmp_qso_flux,trans_wave,transmission)

    if args.metals is not None:
        lstMetals = ''
        for m in args.metals: lstMetals += m+', '
        log.info("Apply metals: {}".format(lstMetals[:-2]))
        tmp_qso_flux = apply_metals_transmission(tmp_qso_wave,tmp_qso_flux,trans_wave,transmission,args.metals)


    bbflux=None
    if args.target_selection or args.mags :
        bands=['FLUX_G','FLUX_R','FLUX_Z', 'FLUX_W1', 'FLUX_W2']
        bbflux=dict()
        # need to recompute the magnitudes to account for lya transmission
        log.info("Compute QSO magnitudes")
        maggies = decam_and_wise_filters.get_ab_maggies(
            1e-17 * tmp_qso_flux, tmp_qso_wave)
        for band, filt in zip( bands,
                            [ 'decam2014-g', 'decam2014-r', 'decam2014-z',
                              'wise2010-W1', 'wise2010-W2']  ):
            
            bbflux[band] = np.ma.getdata(1e9 * maggies[filt]) # nanomaggies
    
    if args.target_selection :
        log.info("Apply target selection")
        isqso = isQSO_colors(gflux=bbflux['FLUX_G'], rflux=bbflux['FLUX_R'], zflux=bbflux['FLUX_Z'],
                             w1flux=bbflux['FLUX_W1'], w2flux=bbflux['FLUX_W2'])
        log.info("Target selection: {}/{} QSOs selected".format(np.sum(isqso),nqso))
        selection=np.where(isqso)[0]
        if selection.size==0 : return
        tmp_qso_flux = tmp_qso_flux[selection]
        metadata     = metadata[:][selection]
        meta         = meta[:][selection]
        for band in bands : 
            bbflux[band] = bbflux[band][selection]
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

    
    meta={"HPXNSIDE":nside,"HPXPIXEL":healpix, "HPXNEST":hpxnest}
     
    if args.target_selection or args.mags :
        # today we write mags because that's what is in the fibermap
        mags=np.zeros((qso_flux.shape[0],5))
        for i,band in enumerate(bands) :
            jj=(bbflux[band]>0)
            mags[jj,i] = 22.5-2.5*np.log10(bbflux[band][jj]) # AB magnitudes
        fibermap_columns={"MAG":mags}
    else :
        fibermap_columns=None
    
    sim_spectra(qso_wave,qso_flux, args.program, obsconditions=obsconditions,spectra_filename=ofilename,sourcetype="qso", skyerr=args.skyerr,ra=metadata["RA"],dec=metadata["DEC"],targetid=targetid,meta=meta,seed=seed,fibermap_columns=fibermap_columns)
    
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
  

        if args.dla:
#This will change according to discussion 
           log.info("Updating the spectra file to add DLA metadata {}".format(ofilename))
           hdudla = pyfits.table_to_hdu(dla_meta); hdudla.name="DLA_META"
           hdul=pyfits.open(ofilename, mode='update')
           hdul.append(hdudla)
           hdul.flush()
           hdul.close()  



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
    if args.target_selection or args.mags :
        log.info("Load DeCAM and WISE filters for target selection sim.")
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
    
        
