import sys, os
import argparse
import time
import warnings

import numpy as np
from scipy.constants import speed_of_light
from astropy.table import Table,Column
import astropy.io.fits as pyfits
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import healpy

import desisim
from desiutil.log import get_logger
from desispec.io.util import write_bintable
from desispec.io.fibermap import read_fibermap
from desisim.simexp import reference_conditions
from desisim.templates import SIMQSO, QSO
from desisim.scripts.quickspectra import sim_spectra
from desisim.lya_spectra import read_lya_skewers , apply_lya_transmission, apply_metals_transmission, lambda_RF_LYA
from desisim.dla import dla_spec,insert_dlas
from desisim.bal import BAL
from desisim.io import empty_metatable
from desisim.survey_release import SurveyRelease
from desispec.interpolation import resample_flux
from desisim.resample import interp_many, resample_flux_many

from desimodel.io import load_pixweight
from speclite import filters

try:
    c_kms = speed_of_light/1000. #- km/s
except TypeError:
    #
    # This can happen in documentation builds.
    #
    c_kms = 299792458.0/1000.0

def _positive_float(value):
    """argparse type: a finite number larger than zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("{!r} is not a finite number larger than zero".format(value))
    return number


def parse(options=None):
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fast simulation of QSO Lya spectra into the final DESI format\
            (Spectra class) that can be directly used as an input to the redshift fitter\
            (redrock) or correlation function code (picca). The input file is a Lya\
            transmission skewer fits file which format is described in\
            https://desi.lbl.gov/trac/wiki/LymanAlphaWG/LyaSpecSim.")

    #- Required
    parser.add_argument('-i','--infile', type=str, nargs= "*", required=True, help="Input skewer healpix fits file(s)")

    parser.add_argument('-o','--outfile', type=str, required=False, help="Output spectra (only used if single input file)")

    parser.add_argument('--outdir', type=str, default=".", required=False, help="Output directory")

    parser.add_argument('--seed', type=int, default=None, required = False, help="Global random seed (will be used to generate a seed per each file")

    parser.add_argument('--sigma_kms_fog',type=float,default=150, help="Adds a gaussian error to the quasar \
        redshift that simulate the fingers of god effect")

    parser.add_argument('--add-LYB', action='store_true', help = "Add LYB absorption from transmision file")

    parser.add_argument('--metals', type=str, default=None, required=False, help = "list of metals to get the\
        transmission from, if 'all' runs on all metals", nargs='*')

    parser.add_argument('--metals-from-file',type=str,const='all',help = "list of metals,'SI1260,SI1207' etc, to get from HDUs in file. \
Use 'all' or no argument for mock version < 7.3 or final metal runs. ",nargs='?')

    parser.add_argument('--dla',type=str,required=False, help="Add DLA to simulated spectra either randonmly\
        (--dla random) or from transmision file (--dla file)")

    parser.add_argument('--dlaplus',action='store_true', help="Add absorption in higher order Lyman lines for DLAs")

    parser.add_argument('--balprob',type=float,required=False, help="To add BAL features with the specified probability\
        (e.g --balprob 0.5). Expect a number between 0 and 1 ")

    parser.add_argument('--no-simqso',action = "store_true", help="Does not use desisim.templates.SIMQSO\
        to generate templates, and uses desisim.templates.QSO instead.")

    parser.add_argument('--save-continuum',action = "store_true", help="Save true continum to file")

    parser.add_argument('--no-lya-transmission',action = 'store_true', help='Do not apply the Lya absorption\
        (F=1). The other absorbers are still added if requested: with --metals, metal-only spectra; with --dla,\
        DLA-only spectra; with both, metals and DLAs without Lya. Not with --metals-from-file')


    parser.add_argument('--nproc', type=int, default=1,help="number of processors to run faster")

    parser.add_argument('--mem-per-worker', type=_positive_float, default=None, help="Memory in GiB that one worker "
        "can use at its peak (about 8 for the DR2 mocks with the production arguments). If given, --nproc is "
        "lowered to what fits in the memory available to the job (the SLURM allocation if set, else the free "
        "memory of the node), so that the node is not run out of memory.")

    parser.add_argument('--fiber-chunk', type=int, default=None, help="Simulate the spectra of a healpix in chunks "
        "of this many fibres to limit the memory of the specsim simulator (about 7 GiB for 1500 quasars; a chunk "
        "of 200 needs a fraction of it). The random noise is drawn for all the fibres at once as without this "
        "option, so the output is the same.")

    parser.add_argument('--overwrite', action = "store_true" ,help="rerun if spectra exists (default is skip)")

    parser.add_argument('--nmax', type=int, default=None, help="Max number of QSO per input file, for debugging")

    parser.add_argument('--source-contr-smoothing', type=float, default=10., \
        help="When this argument > 0 A, source electrons' contribution to the noise is smoothed " \
        "by a Gaussian kernel using FFT. Pipeline does this by 10 A. " \
        "Larger smoothing might be needed for better decoupling. Does not apply to eBOSS mocks.")
    
    parser.add_argument('--from-catalog', type=str, default=None, help="Input catalog of mock objects to simulate")
    
    parser.add_argument('--raw-mock', type=str, default=None, choices=["lyacolore","saclay","ohio"], help="Input raw mock type (lyacolore, saclay or Ohio): default metal strengths")

    parser.add_argument('--metal-strengths', type=float, default=None, required=False, help = "list of strengths to appply\
        to metals. Should correspond to the --metals flag", nargs='*')


    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def get_spectra_filename(args,nside,pixel):
    if args.outfile :
        return args.outfile
    filename="{}/{}/spectra-{}-{}.fits".format(pixel//100,pixel,nside,pixel)
    return os.path.join(args.outdir,filename)


def get_zbest_filename(pixdir,nside,pixel):
    return os.path.join(pixdir,"zbest-{}-{}.fits".format(nside,pixel))


def get_truth_filename(pixdir,nside,pixel):
    return os.path.join(pixdir,"truth-{}-{}.fits".format(nside,pixel))


def is_south(dec):
    """Identify which QSOs are in the south vs the north, since these are on
    different photometric systems.  See
    https://github.com/desihub/desitarget/issues/353 for details.

    """
    return dec <= 32.125 # constant-declination cut!


def get_healpix_info(ifilename):
    """Read the header of the tranmission file to find the healpix pixel, nside
    and if we are lucky the scheme. If it fails, try to guess it from the
    filename (for backward compatibility).

    Args:
        ifilename: full path to input transmission file
    Returns:
        healpix: HEALPix pixel corresponding to the file
        nside: HEALPix nside value
        hpxnest: Whether HEALPix scheme in the file was nested
    """

    log = get_logger()

    print('ifilename',ifilename)

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
    hdulist.close()

    if healpix >= 0 and nside < 0 :
        log.error("Read healpix in header but not nside.")
        raise ValueError("Read healpix in header but not nside.")

    if healpix < 0 :
        vals = os.path.basename(ifilename).split(".")[0].split("-")
        if len(vals)<3 :
            error_msg="Could not guess healpix info from {}".format(ifilename)
            log.error(error_msg)
            raise ValueError(error_msg)
        try :
            healpix=int(vals[-1])
            nside=int(vals[-2])
        except ValueError:
            error_msg="Could not guess healpix info from {}".format(ifilename)
            log.error(error_msg)
            raise ValueError(error_msg)
        log.warning("Guessed healpix and nside from filename, assuming the healpix scheme is 'NESTED'")

    return healpix, nside, hpxnest


def get_pixel_seed(pixel, nside, global_seed):
    if global_seed is None:
        # return a random seed
        return np.random.randint(2**32, size=1)[0]
    npix=healpy.nside2npix(nside)
    np.random.seed(global_seed)
    seeds = np.unique(np.random.randint(2**32, size=10*npix))[:npix]
    pixel_seed = seeds[pixel]
    return pixel_seed


def simulate_one_healpix(ifilename, args, model, bal=None):
    log = get_logger()

    # open filename and extract basic HEALPix information
    pixel, nside, hpxnest = get_healpix_info(ifilename)

    # using global seed (could be None) get seed for this particular pixel
    global_seed = args.seed
    seed = get_pixel_seed(pixel, nside, global_seed)
    # use this seed to generate future random numbers
    np.random.seed(seed)
    # get output file (we will write there spectra for this HEALPix pixel)
    ofilename = get_spectra_filename(args,nside,pixel)
    # get directory name (we will also write there zbest file)
    pixdir = os.path.dirname(ofilename)
    # get filename for truth file
    truth_filename = get_truth_filename(pixdir,nside,pixel)
    # get filename for zbest file
    zbest_filename = get_zbest_filename(pixdir,nside,pixel)

    if not args.overwrite :
        # check whether output exists or not
        if os.path.isfile(ofilename) and os.path.isfile(zbest_filename) :
            log.info("skip existing {} and {}".format(ofilename,zbest_filename))
            return

    # create sub-directories if required
    if len(pixdir)>0 :
        if not os.path.isdir(pixdir) :
            log.info("Creating dir {}".format(pixdir))
            os.makedirs(pixdir)

    log.info("Read skewers in {}, random seed = {}".format(ifilename,seed))

    # Read transmission from files. It might include DLA information, and it
    # might add metal transmission as well (from the HDU file).
    log.info("Read transmission file {}".format(ifilename))

    trans_wave, transmission, metadata, dla_info = read_lya_skewers(
            ifilename,
            read_dlas=(args.dla=='file'),
            add_metals=args.metals_from_file,
            add_lyb=args.add_LYB
    )

    ### Add Finger-of-God, before generate the continua
    log.info("Add FOG to redshift with sigma {} to quasar redshift".format(args.sigma_kms_fog))
    DZ_FOG = args.sigma_kms_fog/c_kms*(1.+metadata['Z'])*np.random.normal(0,1,metadata['Z'].size)
    metadata['Z'] += DZ_FOG

    #- Reference observing conditions for DARK program
    obsconditions = reference_conditions["DARK"]

    if args.from_catalog is None:
        raise ValueError("Quickquasars needs an input catalog now")
    else:
        log.info(f"Getting objects from catalog {args.from_catalog}")
        catalog = Table.read(args.from_catalog)
        # Get mockobjs in pixel that are in the catalog.
        selection = np.isin(metadata['MOCKID'],catalog['MOCKID'])
        if selection.sum()==0:
            log.warning(f'No intersection with catalog')
            return
        log.info(f'Catalog has {selection.sum()} QSOs in pixel {pixel}')
        transmission = transmission[selection]
        metadata = metadata[:][selection]
        DZ_FOG = DZ_FOG[selection] 
        
        this_pixel_targets = np.isin(catalog['MOCKID'],metadata['MOCKID'])
        catalog = catalog[this_pixel_targets]
        ids_index = [np.where(catalog['MOCKID']==mockid)[0][0] for mockid in metadata['MOCKID']]
        # Ensure catalog and metadata MOCKIDS are in the same order
        catalog = catalog[ids_index]
        if 'EXPTIME' in catalog.colnames:
            exptime=np.array(catalog['EXPTIME'])
            obsconditions['EXPTIME']=exptime
        else:
            raise ValueError("Input catalog in --from-catalog must have EXPTIME column")

        # Get magnitude from catalog 
        if 'FLUX_R' in catalog.colnames:
            mags = 22.5-2.5*np.log10(catalog['FLUX_R'])
        elif 'MAG_R' in catalog.colnames:
            mags=catalog['MAG_R']
        else:
            raise ValueError("Input catalog in --from-catalog must have FLUX_R or MAG_R column")

    nqso=transmission.shape[0]
    
    if args.nmax is not None :
        if args.nmax < nqso :
            log.info("Limit number of QSOs from {} to nmax={} (random subsample)".format(nqso,args.nmax))
            # take a random subsample
            indices = np.random.choice(np.arange(nqso),args.nmax,replace=False)
            transmission = transmission[indices]
            metadata = metadata[:][indices]
            DZ_FOG = DZ_FOG[indices]
            nqso = args.nmax

    # extend the transmission skewers (LyaCoLoRe stops at 6500 A) with F = 1, so that the continua cover the
    # whole simulated range (3500-10000 A); the grid is the one the broad-band fluxes used to need
    wanted_min_wave = 3329. # needed to compute magnitudes for decam2014-r (one could have trimmed the transmission file ...)
    wanted_max_wave = 55501. # needed to compute magnitudes for wise2010-W2

    if trans_wave[0]>wanted_min_wave :
        log.info("Increase wavelength range from {}:{} to {}:{}".format(int(trans_wave[0]),int(trans_wave[-1]),int(wanted_min_wave),int(trans_wave[-1])))
        # pad with ones at short wavelength, we assume F = 1 for z <~ 1.7
        # we don't need any wavelength resolution here
        new_trans_wave = np.append([wanted_min_wave,trans_wave[0]-0.01],trans_wave)
        new_transmission = np.ones((transmission.shape[0],new_trans_wave.size))
        new_transmission[:,2:] = transmission
        trans_wave   = new_trans_wave
        transmission = new_transmission

    if trans_wave[-1]<wanted_max_wave :
        log.info("Increase wavelength range from {}:{} to {}:{}".format(int(trans_wave[0]),int(trans_wave[-1]),int(trans_wave[0]),int(wanted_max_wave)))
        # pad with ones at long wavelength because we assume F = 1
        coarse_dwave = 2. # we don't care about resolution, we just need a decent QSO spectrum, there is no IGM transmission in this range
        n = int((wanted_max_wave-trans_wave[-1])/coarse_dwave)+1
        new_trans_wave = np.append(trans_wave,np.linspace(trans_wave[-1]+coarse_dwave,trans_wave[-1]+coarse_dwave*(n+1),n))
        new_transmission = np.ones((transmission.shape[0],new_trans_wave.size))
        new_transmission[:,:trans_wave.size] = transmission
        trans_wave   = new_trans_wave
        transmission = new_transmission

    # whether to use QSO or SIMQSO to generate quasar continua.  Simulate
    # spectra in the north vs south separately because they're on different
    # photometric systems.
    south = np.where( is_south(metadata['DEC']) )[0]
    north = np.where( ~is_south(metadata['DEC']) )[0]
    meta, qsometa = empty_metatable(nqso, objtype='QSO', simqso=not args.no_simqso)
    if args.no_simqso:
        log.info("Simulate {} QSOs with QSO templates".format(nqso))
        tmp_qso_flux = np.zeros([nqso, len(model.eigenwave)], dtype='f4')
        tmp_qso_wave = np.zeros_like(tmp_qso_flux)
    else:
        log.info("Simulate {} QSOs with SIMQSO templates".format(nqso))
        tmp_qso_flux = np.zeros([nqso, len(model.basewave)], dtype='f4')
        tmp_qso_wave = model.basewave

    for these, issouth in zip( (north, south), (False, True) ):

        # number of quasars in these
        nt = len(these)
        if nt<=0: continue

        _tmp_qso_flux, _tmp_qso_wave, _meta, _qsometa = model.make_templates(
                nmodel=nt,
                redshift=metadata['Z'][these],
                mag=mags[these],
                lyaforest=False,
                nocolorcuts=True,
                noresample=True,
                seed=seed,
                south=issouth
        )

        _meta['TARGETID'] = metadata['MOCKID'][these]
        _qsometa['TARGETID'] = metadata['MOCKID'][these]
        meta[these] = _meta
        qsometa[these] = _qsometa
        tmp_qso_flux[these, :] = _tmp_qso_flux

        if args.no_simqso:
            tmp_qso_wave[these, :] = _tmp_qso_wave

    log.info("Resample to transmission wavelength grid")
    qso_flux=np.zeros((tmp_qso_flux.shape[0],trans_wave.size))
    if args.no_simqso:
        for q in range(tmp_qso_flux.shape[0]) :
            qso_flux[q]=np.interp(trans_wave,tmp_qso_wave[q],tmp_qso_flux[q])
    else:
        # one wavelength grid for all the QSOs
        qso_flux = interp_many(trans_wave, tmp_qso_wave, tmp_qso_flux)

    tmp_qso_flux = qso_flux
    tmp_qso_wave = trans_wave

    if args.save_continuum :
        true_wmin=3500.0
        true_wmax=10000.0
        true_dwave=2.0
        true_wave=np.linspace(true_wmin, true_wmax, int((true_wmax - true_wmin)/true_dwave)+1)
        true_flux=resample_flux_many(true_wave,tmp_qso_wave,tmp_qso_flux)
        continum_meta=Table()
        continum_meta['TARGETID'] = qsometa['TARGETID']
        continum_meta['TRUE_CONT'] = true_flux
        hdu_trueCont = pyfits.convenience.table_to_hdu(continum_meta)
        hdu_trueCont.name = "TRUE_CONT"
        hdu_trueCont.header['wmin'] = true_wmin
        hdu_trueCont.header['wmax'] = true_wmax
        hdu_trueCont.header['dwave'] = true_dwave

        del(continum_meta,true_wave,true_flux)
        log.info("True continum to be saved in {}".format(truth_filename))

    # if requested, add BAL features to the quasar continua
    if args.balprob:
        if args.balprob <= 1. and args.balprob > 0:
            from desisim.io import find_basis_template
            log.info("Adding BALs with probability {}".format(args.balprob))
            # save current random state
            rnd_state = np.random.get_state()
            tmp_qso_flux,meta_bal = bal.insert_bals(tmp_qso_wave, tmp_qso_flux, metadata['Z'],
                                                  balprob= args.balprob, seed=seed, qsoid=metadata['MOCKID'])
            # restore random state to get the same random numbers later
            # as when we don't insert BALs
            np.random.set_state(rnd_state)
            w = np.isin(qsometa['TARGETID'], meta_bal['TARGETID'])
            qsometa['BAL_TEMPLATEID'][w] = meta_bal['BAL_TEMPLATEID']
            hdu_bal=pyfits.convenience.table_to_hdu(meta_bal); hdu_bal.name="BAL_META"
            #Trim to only show the version, assuming it is located in os.environ['DESI_BASIS_TEMPLATES']
            hdu_bal.header["BALTEMPL"]=find_basis_template(objtype='BAL').split('basis_templates/')[1]
            del meta_bal
        else:
            balstr=str(args.balprob)
            log.error("BAL probability is not between 0 and 1 : "+balstr)
            sys.exit(1)

    if not args.no_lya_transmission:
        log.info("Applying Lya transmission")
        tmp_qso_flux = apply_lya_transmission(
            tmp_qso_wave, tmp_qso_flux, trans_wave, transmission)
    else:
        log.info("Skipping Lya transmission (--no-lya-transmission mode)")

    # Apply metals if requested
    if args.metals is not None:
        if args.metals_from_file is not None:
            raise ValueError("Cannot use --metals and --metals-from-file at the same time.")
        # the metal absorption is computed from the Lya skewers (tau_metal = coef * tau_Lya) and multiplies the
        # flux on its own, so it is also added without the Lya absorption (metal-only spectra)
        lstMetals = ', '.join(args.metals)
        log.info("Applying metals: {}".format(lstMetals))
        tmp_qso_flux = apply_metals_transmission(
            tmp_qso_wave, tmp_qso_flux,
            trans_wave, transmission,
            args.metals, mocktype=args.raw_mock, strengths=args.metal_strengths)
    # the Lya and metal absorption are applied: the skewers are no longer needed
    del transmission

    # if requested, add DLA to the transmission skewers. They multiply the flux after the Lya and metal
    # absorption, so the metals are computed from the Lya skewers alone (HCDs decoupled from metals).
    if args.dla is not None :
        # if adding random DLAs, we will need a new random generator
        if args.dla=='random':
            log.info('Adding DLAs randomly')
            random_state_just_for_dlas = np.random.RandomState(seed)
        elif args.dla=='file':
            log.info('Adding DLAs from transmission file')
        else:
            log.error("Wrong option for args.dla: "+args.dla)
            sys.exit(1)

        # if adding DLAs, the information will be printed here
        dla_filename=os.path.join(pixdir,"dla-{}-{}.fits".format(nside,pixel))
        dla_NHI, dla_z, dla_qid,dla_id = [], [], [],[]

        # Identify the minimum redshift at which a DLA can be detected in the spectrum, given the wavelength range of the spectrum.
        wmin = 3500. # first wavelength of the simulated spectra (not of the extended skewers)
        min_lya_z = np.min(wmin/lambda_RF_LYA - 1)

        # loop over quasars in pixel
        for ii in range(len(metadata)):
            # quasars with z < min_z will not have any DLA in spectrum
            if min_lya_z>metadata['Z'][ii]: continue
            # quasar ID
            idd=metadata['MOCKID'][ii]
            dlas=[]
            if args.dla=='file':
                for dla in dla_info[dla_info['MOCKID']==idd]:
                    # Adding only DLAs with z < zqso
                    if dla['Z_DLA_RSD']>=metadata['Z'][ii]: continue
                    dlas.append(dict(z=dla['Z_DLA_RSD'],N=dla['N_HI_DLA'],dlaid=dla['DLAID']))
                if len(dlas)>0:
                    transmission_dla = dla_spec(tmp_qso_wave,dlas,dlaplus=args.dlaplus)

            elif args.dla=='random':
                dlas, transmission_dla = insert_dlas(tmp_qso_wave, metadata['Z'][ii], rstate=random_state_just_for_dlas, dlaplus=args.dlaplus)
                for idla in dlas:
                   idla['dlaid']+=idd*1000      #Added to have unique DLA ids. Same format as DLAs from file.

            # multiply transmissions and store information for the DLA file
            if len(dlas)>0:
                tmp_qso_flux[ii] *= transmission_dla.astype(tmp_qso_flux.dtype, copy=False)  # DLA absorption of this quasar
                dla_z += [idla['z'] for idla in dlas]
                dla_NHI += [idla['N'] for idla in dlas]
                dla_id += [idla['dlaid'] for idla in dlas]
                dla_qid += [idd]*len(dlas)
        log.info('Added {} DLAs'.format(len(dla_id)))
        if args.dlaplus:
            log.info('Added higher order lines to DLAs')
        # write file with DLA information
        if len(dla_id)>0:
            dla_meta=Table()
            dla_meta['NHI'] = dla_NHI
            dla_meta['Z_DLA'] = dla_z  #This is Z_DLA_RSD in transmision.
            dla_meta['TARGETID']=dla_qid
            dla_meta['DLAID'] = dla_id
            hdu_dla = pyfits.convenience.table_to_hdu(dla_meta)
            hdu_dla.name="DLA_META"
            del(dla_meta)
            log.info("DLA metadata to be saved in {}".format(truth_filename))
        else:
            hdu_dla=pyfits.PrimaryHDU()
            hdu_dla.name="DLA_META"
        
        # Add comment to DLA_META extension if dlaplus is enabled
        if args.dlaplus:
            hdu_dla.header['COMMENT'] = 'Added higher order lines to DLAs'
                         
    log.info("Simulate DESI observation and write output file")
    if "MOCKID" in metadata.dtype.names :
        targetid=np.array(metadata["MOCKID"]).astype(int)
    elif "ID" in metadata.dtype.names :
        log.warning("Using ID as TARGETID")
        targetid=np.array(metadata["ID"]).astype(int)
    else :
        log.warning("No TARGETID")
        targetid=None

    log.info("Resample to a linear wavelength grid (needed by DESI sim.)")
    # careful integration of bins, not just a simple interpolation (hardcoded values)
    dwave=0.2
    wmin=3500.0
    wmax=10000.0
    qso_wave=np.linspace(wmin, wmax, int((wmax-wmin)/dwave)+1)
    qso_flux=resample_flux_many(qso_wave,tmp_qso_wave,tmp_qso_flux)

    ### use Poisson = False to get reproducible results.
    specmeta={"HPXNSIDE":nside,"HPXPIXEL":pixel, "HPXNEST":hpxnest}
    resolution=sim_spectra(qso_wave, qso_flux, program="DARK", obsconditions=obsconditions, spectra_filename=ofilename,
                           sourcetype="qso", ra=metadata["RA"], dec=metadata["DEC"], targetid=targetid,
                           meta=specmeta, seed=seed, fibermap_columns=None, use_poisson=False,
                           specsim_config_file='desi', dwave_out=0.8,
                           save_resolution=False, source_contribution_smoothing=args.source_contr_smoothing, fiber_chunk=args.fiber_chunk)

    ### Keep input redshift
    Z_spec = metadata['Z'].copy()
    Z_input = metadata['Z'].copy()-DZ_FOG

    ## Write the truth file, including metadata for DLAs and BALs
    log.info('Writing a truth file  {}'.format(truth_filename))
    meta.rename_column('REDSHIFT','Z')
    meta.add_column(Column(Z_spec,name='TRUEZ'))
    meta.add_column(Column(Z_input,name='Z_INPUT'))
    meta.add_column(Column(DZ_FOG,name='DZ_FOG'))
    if 'Z_noRSD' in metadata.dtype.names:
        meta.add_column(Column(metadata['Z_noRSD'],name='Z_NORSD'))
    else:
        log.info('Z_noRSD field not present in transmission file. Z_NORSD not saved to truth file')
    meta.add_column(Column(exptime,name='EXPTIME'))

    #Save global seed and pixel seed to primary header
    hdr=pyfits.Header()
    hdr['GSEED']=global_seed
    hdr['PIXSEED']=seed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*nanomaggies.*")
        hdu = pyfits.convenience.table_to_hdu(meta)

    hdu.header['EXTNAME'] = 'TRUTH'
    hduqso=pyfits.convenience.table_to_hdu(qsometa)
    hduqso.header['EXTNAME'] = 'TRUTH_QSO'
    hdulist=pyfits.HDUList([pyfits.PrimaryHDU(header=hdr),hdu,hduqso])

    # save DLA, BAL and true continuum metadata
    if args.dla :
        hdulist.append(hdu_dla)
    if  args.balprob :
        hdulist.append(hdu_bal)
    if args.save_continuum :
        hdulist.append(hdu_trueCont)

    # Save one resolution matrix per camera to the truth file instead of one per quasar to the spectra files.
    for band in resolution.keys():
        hdu = pyfits.ImageHDU(name="{}_RESOLUTION".format(band.upper()))
        hdu.data = resolution[band].astype("f4")
        hdulist.append(hdu)

    hdulist.writeto(truth_filename, overwrite=True)
    hdulist.close()

    # write zbest file
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
    zbest['CHI2'][:] = 0.
    zbest['Z'][:] = metadata['Z']
    zbest['ZERR'][:] = 0.
    zbest['ZWARN'][:] = 0
    zbest['SPECTYPE'][:] = 'QSO'
    zbest['SUBTYPE'][:] = ''
    zbest['TARGETID'][:] = metadata['MOCKID']
    zbest['DELTACHI2'][:] = 25.
    hzbest = pyfits.convenience.table_to_hdu(zbest); hzbest.name='ZBEST'
    hfmap  = pyfits.convenience.table_to_hdu(fibermap);  hfmap.name='FIBERMAP'
    hdulist =pyfits.HDUList([pyfits.PrimaryHDU(),hzbest,hfmap])
    hdulist.writeto(zbest_filename, overwrite=True)
    hdulist.close() # see if this helps with memory issue


def _release_freed_memory():
    """Give the memory freed by the last healpix back to the OS (glibc keeps much of it in the heap of the
    process, so the workers grew healpix after healpix)."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

def _func(arg) :
    """ Used for the ProcessPoolExecutor in main() """
    try:
        return simulate_one_healpix(**arg)
    finally:
        _release_freed_memory()

def _memory_limited_nproc(nproc, mem_per_worker_gib):
    """Lower ``nproc`` to what fits in the memory available to this job.

    The memory is the SLURM allocation of the node (``SLURM_MEM_PER_NODE``, in MiB) if it is set,
    else the available memory of the node, whichever is smaller."""
    log = get_logger()
    if not (np.isfinite(mem_per_worker_gib) and mem_per_worker_gib > 0):
        raise ValueError("mem_per_worker_gib must be a finite number larger than zero, got {!r}".format(mem_per_worker_gib))
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        log.warning("psutil is not available, --mem-per-worker is ignored")
        return nproc
    if os.environ.get("SLURM_MEM_PER_NODE", "").isdigit():
        available = min(available, int(os.environ["SLURM_MEM_PER_NODE"]) * 2**20)
    fit = max(1, int(available / (mem_per_worker_gib * 2**30)))
    if fit < nproc:
        log.warning("--nproc lowered from {} to {}: {:.0f} GiB available for {} GiB per worker".format(
            nproc, fit, available / 2**30, mem_per_worker_gib))
    else:
        log.info("--nproc {} fits in the {:.0f} GiB available ({} GiB per worker)".format(
            nproc, available / 2**30, mem_per_worker_gib))
    return min(nproc, fit)


def main(args=None):

    log = get_logger()
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if args.no_lya_transmission and args.metals_from_file is not None:
        log.error("--metals-from-file cannot be used with --no-lya-transmission (the metals of the file are part of its skewers); use --metals")
        return 1

    if args.outfile is not None and len(args.infile)>1 :
        log.error("Cannot specify single output file with multiple inputs, use --outdir option instead")
        return 1

    if not os.path.isdir(args.outdir) :
        log.info("Creating dir {}".format(args.outdir))
        os.makedirs(args.outdir)

    if args.no_simqso:
        log.info("Load QSO model")
        model=QSO()
    else:
        log.info("Load SIMQSO model")
        #lya_simqso_model.py is located in $DESISIM/py/desisim/scripts/.
        #Uses a different emmision lines model than the default SIMQSO. 
        #We will update this soon to match with the one used in select_mock_targets. 
        model=SIMQSO(nproc=1,sqmodel='lya_simqso_model')

    if args.balprob:
        bal=BAL()
    else:
        bal=None

    if args.mem_per_worker is not None and args.nproc > 1:
        args.nproc = _memory_limited_nproc(args.nproc, args.mem_per_worker)

    if args.nproc > 1:
        func_args = [ {"ifilename":filename, "args":args, "model":model, \
                       "bal":bal} for i,filename in enumerate(args.infile) ]
        # ProcessPoolExecutor rather than multiprocessing.Pool.map: if a worker dies
        # (e.g. killed by the OOM killer) Pool.map waits forever for the lost task,
        # whereas the executor raises BrokenProcessPool. One task per healpix keeps
        # the workers balanced.
        executor = ProcessPoolExecutor(max_workers=args.nproc)
        futures = {executor.submit(_func, fa): fa["ifilename"] for fa in func_args}
        try:
            for ndone, future in enumerate(as_completed(futures), 1):
                future.result()
                log.debug("Finished {}/{} healpix".format(ndone, len(futures)))
        except BrokenProcessPool:
            unfinished = [f for fut, f in futures.items()
                          if not (fut.done() and not fut.cancelled() and fut.exception() is None)]
            log.error("A worker process died (possibly killed for exceeding the node "
                      "memory); {} of {} healpix did not finish. Consider fewer --nproc. "
                      "Unfinished inputs:\n{}".format(len(unfinished), len(futures), "\n".join(unfinished)))
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    else:
        for i,ifilename in enumerate(args.infile) :
            simulate_one_healpix(ifilename, args, model, bal=bal)
