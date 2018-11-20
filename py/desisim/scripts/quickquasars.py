from __future__ import absolute_import, division, print_function

import sys, os
import argparse
import time

import numpy as np
from scipy.constants import speed_of_light
from scipy.stats import cauchy
from astropy.table import Table,Column
import astropy.io.fits as pyfits
import multiprocessing
import healpy

from desiutil.log import get_logger
from desispec.io.util import write_bintable
from desispec.io.fibermap import read_fibermap
from desisim.simexp import reference_conditions
from desisim.templates import SIMQSO, QSO
from desisim.scripts.quickspectra import sim_spectra
from desisim.lya_spectra import read_lya_skewers , apply_lya_transmission, apply_metals_transmission
from desisim.dla import dla_spec,insert_dlas
from desisim.bal import BAL
from desisim.io import empty_metatable
from desisim.eboss import FootprintEBOSS, sdss_subsample, RedshiftDistributionEBOSS, sdss_subsample_redshift
from desispec.interpolation import resample_flux

from desimodel.io import load_pixweight
from desimodel import footprint
from speclite import filters
from desitarget.cuts import isQSO_colors

c = speed_of_light/1000. #- km/s
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
    parser.add_argument('--seed', type=int, default=None, required = False, help="Global random seed (will be used to generate a seed per each file")
    parser.add_argument('--skyerr', type=float, default=0.0, help="Fractional sky subtraction error")
    parser.add_argument('--nmax', type=int, default=None, help="Max number of QSO per input file, for debugging")
    parser.add_argument('--downsampling', type=float, default=None,help="fractional random down-sampling (value between 0 and 1)")
    parser.add_argument('--zmin', type=float, default=0,help="Min redshift")
    parser.add_argument('--zmax', type=float, default=10,help="Max redshift")
    parser.add_argument('--wmin', type=float, default=3500,help="Min wavelength (obs. frame)")
    parser.add_argument('--wmax', type=float, default=10000,help="Max wavelength (obs. frame)")
    parser.add_argument('--dwave', type=float, default=0.2,help="Internal wavelength step (don't change this)")
    parser.add_argument('--nproc', type=int, default=1,help="number of processors to run faster")

    parser.add_argument('--zbest', action = "store_true",help="add a zbest file per spectrum either with the truth redshift or adding some error (optionally use it with --sigma_kms_fog and/or --sigma_kms_zfit)")

    parser.add_argument('--sigma_kms_fog',type=float,default=150, help="Adds a gaussian error to the quasar redshift that simulate the fingers of god effect")

    parser.add_argument('--sigma_kms_zfit',nargs='?',type=float,const=400,help="Adds a gaussian error to the quasar redshift, to simulate the redshift fitting step. E.g. --sigma_kms_zfit 200 will use a sigma value of 200 km/s. If a number is not specified the default is used.")

    parser.add_argument('--shift_kms_los',nargs='?',type=float,const=100,help="Adds a shift to the quasar redshift as in arXiv:1708.02225. If a number is not specified the default (100km/s) is used.")
    parser.add_argument('--overwrite', action = "store_true" ,help="rerun if spectra exists (default is skip)")
    parser.add_argument('--target-selection', action = "store_true" ,help="apply QSO target selection cuts to the simulated quasars")
    parser.add_argument('--mags', action = "store_true", help="DEPRECATED; use --bbflux")
    parser.add_argument('--bbflux', action = "store_true", help="compute and write the QSO broad-band fluxes in the fibermap")
    parser.add_argument('--desi-footprint', action = "store_true" ,help="select QSOs in DESI footprint")
    parser.add_argument('--metals', type=str, default=None, required=False, help = "list of metals to get the transmission from, if 'all' runs on all metals", nargs='*')
    parser.add_argument('--metals-from-file', action = 'store_true', help = "add metals from HDU in file")
    parser.add_argument('--dla',type=str,required=False, help="Add DLA to simulated spectra either randonmly (--dla random) or from transmision file (--dla file)")
    parser.add_argument('--balprob',type=float,required=False, help="To add BAL features with the specified probability (e.g --balprob 0.5). Expect a number between 0 and 1 ")
    parser.add_argument('--no-simqso',action = "store_true", help="Does not use desisim.templates.SIMQSO to generate templates, and uses desisim.templates.QSO instead.")
    parser.add_argument('--no-transmission',action = 'store_true', help='Do not multiply continuum by transmission, use F=1 everywhere')
    parser.add_argument('--eboss',action = 'store_true', help='Setup footprint, number density, redshift distribution, and exposure time to generate eBOSS-like mocks')

    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args


def mod_cauchy(loc,scale,size,cut):
    samples=cauchy.rvs(loc=loc,scale=scale,size=3*size)
    samples=samples[abs(samples)<cut]   
    if len(samples)>=size:
       samples=samples[:size] 
    else:     
        samples=mod_cauchy(loc,scale,size,cut)   ##Only added for the very unlikely case that there are not enough samples after the cut.
    return samples
             
def get_spectra_filename(args,nside,pixel):
    if args.outfile :
        return args.outfile
    filename="{}/{}/spectra-{}-{}.fits".format(pixel//100,pixel,nside,pixel)
    return os.path.join(args.outdir,filename)


def get_zbest_filename(args,pixdir,nside,pixel):
    if args.zbest :
        return os.path.join(pixdir,"zbest-{}-{}.fits".format(nside,pixel))
    return None


def get_truth_filename(args,pixdir,nside,pixel):
    return os.path.join(pixdir,"truth-{}-{}.fits".format(nside,pixel))


def is_south(dec):
    """Identify which QSOs are in the south vs the north, since these are on
    different photometric systems.  See
    https://github.com/desihub/desitarget/issues/353 for details.

    """
    return dec <= 32.125 # constant-declination cut!


def get_healpix_info(ifilename):
    """
    Read the header of the tranmission file to find the healpix pixel, nside
    and if we are lucky the scheme. If it fails, try to guess it from the
    filename (for backward compatibility).
    Inputs:
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


def simulate_one_healpix(ifilename,args,model,obsconditions,decam_and_wise_filters,
                         bassmzls_and_wise_filters,footprint_healpix_weight,
                         footprint_healpix_nside,
                         bal=None,eboss=None) :
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
    truth_filename = get_truth_filename(args,pixdir,nside,pixel)

    # get filename for zbest file
    zbest_filename = get_zbest_filename(args,pixdir,nside,pixel)

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

    # create sub-directories if required
    if len(pixdir)>0 :
        if not os.path.isdir(pixdir) :
            log.info("Creating dir {}".format(pixdir))
            os.makedirs(pixdir)

    log.info("Read skewers in {}, random seed = {}".format(ifilename,seed))

    # Read transmission from files. It might include DLA information, and it
    # might add metal transmission as well (from the HDU file).
    log.info("Read transmission file {}".format(ifilename))
    trans_wave, transmission, metadata, dla_info = read_lya_skewers(ifilename,read_dlas=(args.dla=='file'),add_metals=args.metals_from_file)

###ADD dz_fog before generate the continua
    #metadata.add_column(Column(metadata['Z'],name='TRUEZ_noFOG'))
    Z_noFOG=np.copy(metadata['Z'])
    log.info("Add FOG to redshift with sigma {} to quasar redshift".format(args.sigma_kms_fog)) 
    dz_fog=(args.sigma_kms_fog/c)*(1.+metadata['Z'])*np.random.normal(0,1,len(metadata['Z']))    
    metadata['Z']+=dz_fog
    ok = np.where(( metadata['Z'] >= args.zmin ) & (metadata['Z'] <= args.zmax ))[0]
    transmission = transmission[ok]
    metadata = metadata[:][ok]
    Z_noFOG=Z_noFOG[ok]

    # option to make for BOSS+eBOSS
    if not eboss is None:
        if args.downsampling or args.desi_footprint:
            raise ValueError("eboss option can not be run with "
                    +"desi_footprint or downsampling")

        # Get the redshift distribution from SDSS
        selection = sdss_subsample_redshift(metadata["RA"],metadata["DEC"],metadata["Z"],eboss['redshift'])
        log.info("Select QSOs in BOSS+eBOSS redshift distribution {} -> {}".format(metadata["Z"].size,selection.sum()))
        if selection.sum()==0:
            log.warning("No intersection with BOSS+eBOSS redshift distribution")
            return
        transmission = transmission[selection]
        metadata = metadata[:][selection]

        # figure out the density of all quasars
        N_highz = metadata["Z"].size
        # area of healpix pixel, in degrees
        area_deg2 = healpy.pixelfunc.nside2pixarea(nside,degrees=True)
        input_highz_dens_deg2 = N_highz/area_deg2
        selection = sdss_subsample(metadata["RA"], metadata["DEC"],
                        input_highz_dens_deg2,eboss['footprint'])
        log.info("Select QSOs in BOSS+eBOSS footprint {} -> {}".format(transmission.shape[0],selection.size))
        if selection.size == 0 :
            log.warning("No intersection with BOSS+eBOSS footprint")
            return
        transmission = transmission[selection]
        metadata = metadata[:][selection]

    if args.desi_footprint :
        footprint_healpix = footprint.radec2pix(footprint_healpix_nside, metadata["RA"], metadata["DEC"])
        selection = np.where(footprint_healpix_weight[footprint_healpix]>0.99)[0]
        log.info("Select QSOs in DESI footprint {} -> {}".format(transmission.shape[0],selection.size))
        if selection.size == 0 :
            log.warning("No intersection with DESI footprint")
            return
        transmission = transmission[selection]
        metadata = metadata[:][selection]
        Z_noFOG=Z_noFOG[selection]

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
<<<<<<< HEAD
        Z_noFOG=Z_noFOG[indices]
        nqso = transmission.shape[0] 
=======
        nqso = transmission.shape[0]
>>>>>>> master

    if args.nmax is not None :
        if args.nmax < nqso :
            log.info("Limit number of QSOs from {} to nmax={} (random subsample)".format(nqso,args.nmax))
            # take a random subsample
            indices = (np.random.uniform(size=args.nmax)*nqso).astype(int)
            transmission = transmission[indices]
            metadata = metadata[:][indices]
            Z_noFOG=Z_noFOG[indices]
            nqso = args.nmax

    # In previous versions of the London mocks we needed to enforce F=1 for
    # z > z_qso here, but this is not needed anymore. Moreover, now we also
    # have metal absorption that implies F < 1 for z > z_qso
    #for ii in range(len(metadata)):
    #    transmission[ii][trans_wave>1215.67*(metadata[ii]['Z']+1)]=1.0

    # if requested, add DLA to the transmission skewers
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
        dla_NHI, dla_z, dla_id = [], [], []

        # identify minimum Lya redshift in transmission files
        min_lya_z = np.min(trans_wave/1215.67 - 1)

        # loop over quasars in pixel

        for ii in range(len(metadata)):

            # quasars with z < min_z will not have any DLA in spectrum
            if min_lya_z > metadata[ii]['Z']:
                continue

            # quasar ID
            idd=metadata['MOCKID'][ii]
            dlas=[]

            if args.dla=='file':

                for dla in dla_info[dla_info['MOCKID']==idd]:
                    # Adding only DLAs with z < zqso
                    if (dla['Z_DLA']< metadata[ii]['Z']):
                        dlas.append(dict(z=dla['Z_DLA']+dla['DZ_DLA'],N=dla['N_HI_DLA']))
                transmission_dla = dla_spec(trans_wave,dlas)

            elif args.dla=='random':

                dlas, transmission_dla = insert_dlas(trans_wave, metadata[ii]['Z'], rstate=random_state_just_for_dlas)

            # multiply transmissions and store information for the DLA file
            if len(dlas)>0:
                transmission[ii] = transmission_dla * transmission[ii]
                dla_z += [idla['z'] for idla in dlas]
                dla_NHI += [idla['N'] for idla in dlas]
                dla_id += [idd]*len(dlas)
        log.info('Added {} DLAs'.format(len(dla_id)))
        # write file with DLA information
        if len(dla_id)>0:
            dla_meta=Table()
            dla_meta['NHI']=dla_NHI
            dla_meta['z']=dla_z
            dla_meta['ID']=dla_id

            hdu_dla = pyfits.convenience.table_to_hdu(dla_meta)
            hdu_dla.name="DLA_META"
            del(dla_meta)
            log.info("DLA metadata to be saved in {}".format(truth_filename))
        else:
            hdu_dla=pyfits.PrimaryHDU()
            hdu_dla.name="DLA_META"

    # if requested, extend transmission skewers to cover full spectrum
    if args.target_selection or args.bbflux :
        wanted_min_wave = 3329. # needed to compute magnitudes for decam2014-r (one could have trimmed the transmission file ...)
        wanted_max_wave = 55501. # needed to compute magnitudes for wise2010-W2

        if trans_wave[0]>wanted_min_wave :
            log.info("Increase wavelength range from {}:{} to {}:{} to compute magnitudes".format(int(trans_wave[0]),int(trans_wave[-1]),int(wanted_min_wave),int(trans_wave[-1])))
            # pad with ones at short wavelength, we assume F = 1 for z <~ 1.7
            # we don't need any wavelength resolution here
            new_trans_wave = np.append([wanted_min_wave,trans_wave[0]-0.01],trans_wave)
            new_transmission = np.ones((transmission.shape[0],new_trans_wave.size))
            new_transmission[:,2:] = transmission
            trans_wave   = new_trans_wave
            transmission = new_transmission

        if trans_wave[-1]<wanted_max_wave :
            log.info("Increase wavelength range from {}:{} to {}:{} to compute magnitudes".format(int(trans_wave[0]),int(trans_wave[-1]),int(trans_wave[0]),int(wanted_max_wave)))
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

<<<<<<< HEAD
    

=======
>>>>>>> master
    for these, issouth in zip( (north, south), (False, True) ):

        # number of quasars in these
        nt = len(these)
        if nt<=0: continue

        if not eboss is None:
            # for eBOSS, generate only quasars with r<22
            magrange = (17.0, 22.0)
            _tmp_qso_flux, _tmp_qso_wave, _meta, _qsometa \
                = model.make_templates(nmodel=nt,
                    redshift=metadata['Z'][these], magrange=magrange,
                    lyaforest=False, nocolorcuts=True,
                    noresample=True, seed=seed, south=issouth)
        else:
            _tmp_qso_flux, _tmp_qso_wave, _meta, _qsometa \
                = model.make_templates(nmodel=nt,
                    redshift=metadata['Z'][these],
                    lyaforest=False, nocolorcuts=True,
                    noresample=True, seed=seed, south=issouth)

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
        for q in range(tmp_qso_flux.shape[0]) :
            qso_flux[q]=np.interp(trans_wave,tmp_qso_wave,tmp_qso_flux[q])

    tmp_qso_flux = qso_flux
    tmp_qso_wave = trans_wave

    # if requested, add BAL features to the quasar continua
    if args.balprob:
        if args.balprob<=1. and args.balprob >0:
            log.info("Adding BALs with probability {}".format(args.balprob))
            # save current random state
<<<<<<< HEAD
            rnd_state = np.random.get_state() 
            tmp_qso_flux,meta_bal=bal.insert_bals(tmp_qso_wave,tmp_qso_flux,metadata['Z'],
=======
            rnd_state = np.random.get_state()
            tmp_qso_flux,meta_bal=bal.insert_bals(tmp_qso_wave,tmp_qso_flux, metadata['Z'],
>>>>>>> master
                                                  balprob=args.balprob,seed=seed)
            # restore random state to get the same random numbers later
            # as when we don't insert BALs
            np.random.set_state(rnd_state)
            hdu_bal=pyfits.convenience.table_to_hdu(meta_bal); hdu_bal.name="BAL_META"
            del meta_bal
        else:
            balstr=str(args.balprob)
            log.error("BAL probability is not between 0 and 1 : "+balstr)
            sys.exit(1)

    # Multiply quasar continua by transmitted flux fraction
    # (at this point transmission file might include Ly-beta, metals and DLAs)
    log.info("Apply transmitted flux fraction")
    if not args.no_transmission:
        tmp_qso_flux = apply_lya_transmission(tmp_qso_wave,tmp_qso_flux,
                            trans_wave,transmission)

    # if requested, compute metal transmission on the fly
    # (if not included already from the transmission file)
    if args.metals is not None:
        if args.metals_from_file:
            log.error('you cannot add metals twice')
            raise ValueError('you cannot add metals twice')
        if args.no_transmission:
            log.error('you cannot add metals if asking for no-transmission')
            raise ValueError('can not add metals if using no-transmission')
        lstMetals = ''
        for m in args.metals: lstMetals += m+', '
        log.info("Apply metals: {}".format(lstMetals[:-2]))
        tmp_qso_flux = apply_metals_transmission(tmp_qso_wave,tmp_qso_flux,
                            trans_wave,transmission,args.metals)

    # if requested, compute magnitudes and apply target selection.  Need to do
    # this calculation separately for QSOs in the north vs south.
    bbflux=None
    if args.target_selection or args.bbflux :
        bands=['FLUX_G','FLUX_R','FLUX_Z', 'FLUX_W1', 'FLUX_W2']
        bbflux=dict()
        bbflux['SOUTH'] = is_south(metadata['DEC'])
        for band in bands:
            bbflux[band] = np.zeros(nqso)
        # need to recompute the magnitudes to account for lya transmission
        log.info("Compute QSO magnitudes")

        for these, filters in zip( (~bbflux['SOUTH'], bbflux['SOUTH']),
                                   (bassmzls_and_wise_filters, decam_and_wise_filters) ):
            if np.count_nonzero(these) > 0:
                maggies = filters.get_ab_maggies(1e-17 * tmp_qso_flux[these, :], tmp_qso_wave)
                for band, filt in zip( bands, maggies.colnames ):
                    bbflux[band][these] = np.ma.getdata(1e9 * maggies[filt]) # nanomaggies

    if args.target_selection :
        log.info("Apply target selection")
        isqso = np.ones(nqso, dtype=bool)
        for these, issouth in zip( (~bbflux['SOUTH'], bbflux['SOUTH']), (False, True) ):
            if np.count_nonzero(these) > 0:
                # optical cuts only if using QSO vs SIMQSO
                isqso[these] &= isQSO_colors(gflux=bbflux['FLUX_G'][these],
                                             rflux=bbflux['FLUX_R'][these],
                                             zflux=bbflux['FLUX_Z'][these],
                                             w1flux=bbflux['FLUX_W1'][these],
                                             w2flux=bbflux['FLUX_W2'][these],
                                             south=issouth, optical=args.no_simqso)

        log.info("Target selection: {}/{} QSOs selected".format(np.sum(isqso),nqso))
        selection=np.where(isqso)[0]
        if selection.size==0 : return
        tmp_qso_flux = tmp_qso_flux[selection]
        metadata     = metadata[:][selection]
        meta         = meta[:][selection]
        qsometa      = qsometa[:][selection]
        Z_noFOG      = Z_noFOG[selection]

        for band in bands :
            bbflux[band] = bbflux[band][selection]
        nqso         = selection.size

    log.info("Resample to a linear wavelength grid (needed by DESI sim.)")
    # careful integration of bins, not just a simple interpolation
    qso_wave=np.linspace(args.wmin,args.wmax,int((args.wmax-args.wmin)/args.dwave)+1)
    qso_flux=np.zeros((tmp_qso_flux.shape[0],qso_wave.size))
    for q in range(tmp_qso_flux.shape[0]) :
        qso_flux[q]=resample_flux(qso_wave,tmp_qso_wave,tmp_qso_flux[q])

    log.info("Simulate DESI observation and write output file")
    if "MOCKID" in metadata.dtype.names :
        #log.warning("Using MOCKID as TARGETID")
        targetid=np.array(metadata["MOCKID"]).astype(int)
    elif "ID" in metadata.dtype.names :
        log.warning("Using ID as TARGETID")
        targetid=np.array(metadata["ID"]).astype(int)
    else :
        log.warning("No TARGETID")
        targetid=None

    specmeta={"HPXNSIDE":nside,"HPXPIXEL":pixel, "HPXNEST":hpxnest}

    if args.target_selection or args.bbflux :
        fibermap_columns = dict(
            FLUX_G = bbflux['FLUX_G'],
            FLUX_R = bbflux['FLUX_R'],
            FLUX_Z = bbflux['FLUX_Z'],
            FLUX_W1 = bbflux['FLUX_W1'],
            FLUX_W2 = bbflux['FLUX_W2'],
            )
        photsys = np.full(len(bbflux['FLUX_G']), 'N', dtype='S1')
        photsys[bbflux['SOUTH']] = b'S'
        fibermap_columns['PHOTSYS'] = photsys
    else :
        fibermap_columns=None


<<<<<<< HEAD
     
=======
>>>>>>> master
    sim_spectra(qso_wave,qso_flux, args.program, obsconditions=obsconditions,spectra_filename=ofilename,
                sourcetype="qso", skyerr=args.skyerr,ra=metadata["RA"],dec=metadata["DEC"],targetid=targetid,
                meta=specmeta,seed=seed,fibermap_columns=fibermap_columns,use_poisson=False) # use Poisson = False to get reproducible results.

<<<<<<< HEAD
##Adedd to write the truth file, including metadata for DLA's and BALs      
=======

##Adedd to write the truth file, includen metadata for DLA's and BALs
    log.info("Added FOG to redshift with sigma {} to zbest".format(args.sigma_kms_fog))
    dz_fog=(args.sigma_kms_fog/299792.458)*(1.+metadata['Z'])*np.random.normal(0,1,nqso)
    meta.rename_column('REDSHIFT','TRUEZ_noFOG')
>>>>>>> master
    log.info('Writing a truth file  {}'.format(truth_filename))
    meta.rename_column('REDSHIFT','TRUEZ')
    meta.add_column(Column(Z_noFOG,name='TRUEZ_noFOG'))

    if 'Z_noRSD' in metadata.dtype.names:
        meta.add_column(Column(metadata['Z_noRSD'],name='TRUEZ_noRSD'))
    else:
        log.info('Z_noRSD field not present in transmission file. Z_noRSD not saved to truth file')
<<<<<<< HEAD
    if args.shift_kms_los:
       metadata['Z']+=(args.shift_kms_los/c*(1.+metadata['Z']))
       log.info('Added a shift of {} km/s to the redshift'.format(str(args.shift_kms_los)))
=======

    meta.add_column(Column(metadata['Z']+dz_fog,name='TRUEZ'))
>>>>>>> master
    hdu = pyfits.convenience.table_to_hdu(meta)
    hdu.header['EXTNAME'] = 'TRUTH'
    hduqso=pyfits.convenience.table_to_hdu(qsometa)
    hduqso.header['EXTNAME'] = 'QSO_META'
    hdulist=pyfits.HDUList([pyfits.PrimaryHDU(),hdu,hduqso])
    if args.dla:
       hdulist.append(hdu_dla)
    if args.balprob:
       hdulist.append(hdu_bal)
    hdulist.writeto(truth_filename, overwrite=True)
    hdulist.close()




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

        if args.sigma_kms_zfit:
           log.info("Added zfit error with sigma {} to zbest".format(args.sigma_kms_zfit))
           dz_fit=mod_cauchy(loc=0,scale=args.sigma_kms_zfit,size=nqso,cut=3000)*(1.+metadata['Z'])/c
           zbest["Z"]+=dz_fit
        zbest["ZERR"][:]      = 0
        zbest["ZWARN"][:]     = 0
        zbest["SPECTYPE"][:]  = "QSO"
        zbest["SUBTYPE"][:]   = ""
        zbest["TARGETID"]     = fibermap["TARGETID"]
        zbest["DELTACHI2"][:] = 25.
        hzbest = pyfits.convenience.table_to_hdu(zbest); hzbest.name="ZBEST"
        hfmap  = pyfits.convenience.table_to_hdu(fibermap);  hfmap.name="FIBERMAP"
        hdulist =pyfits.HDUList([pyfits.PrimaryHDU(),hzbest,hfmap])
        hdulist.writeto(zbest_filename, overwrite=True)
        hdulist.close() # see if this helps with memory issue

def _func(arg) :
    """ Used for multiprocessing.Pool """
    return simulate_one_healpix(**arg)

def main(args=None):

    log = get_logger()
    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)

    if args.outfile is not None and len(args.infile)>1 :
        log.error("Cannot specify single output file with multiple inputs, use --outdir option instead")
        return 1

    if not os.path.isdir(args.outdir) :
        log.info("Creating dir {}".format(args.outdir))
        os.makedirs(args.outdir)

    if args.mags:
        log.warning('--mags is deprecated; please use --bbflux instead')
        args.bbflux = True

    exptime = args.exptime
    if exptime is None :
        exptime = 1000. # sec
        if args.eboss:
            exptime = 1000. # sec (added here in case we change the default)

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

    if args.no_simqso:
        log.info("Load QSO model")
        model=QSO()
    else:
        log.info("Load SIMQSO model")
        model=SIMQSO(nproc=1)

    decam_and_wise_filters = None
    bassmzls_and_wise_filters = None
    if args.target_selection or args.bbflux :
        log.info("Load DeCAM and WISE filters for target selection sim.")
        # ToDo @moustakas -- load north/south filters
        decam_and_wise_filters = filters.load_filters('decam2014-g', 'decam2014-r', 'decam2014-z',
                                                      'wise2010-W1', 'wise2010-W2')
        bassmzls_and_wise_filters = filters.load_filters('BASS-g', 'BASS-r', 'MzLS-z',
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

    if args.sigma_kms_zfit and not args.zbest:
       log.info("Setting --zbest to true as required by --sigma_kms_zfit")
       args.zbest = True

    if args.balprob:
        bal=BAL()
    else:
        bal=None

    if args.eboss:
        eboss = { 'footprint':FootprintEBOSS(), 'redshift':RedshiftDistributionEBOSS() }
    else:
        eboss = None

    if args.nproc > 1:
        func_args = [ {"ifilename":filename , \
                       "args":args, "model":model , \
                       "obsconditions":obsconditions , \
                       "decam_and_wise_filters": decam_and_wise_filters , \
                       "bassmzls_and_wise_filters": bassmzls_and_wise_filters , \
                       "footprint_healpix_weight": footprint_healpix_weight , \
                       "footprint_healpix_nside": footprint_healpix_nside , \
                       "bal":bal, "eboss":eboss \
                   } for i,filename in enumerate(args.infile) ]
        pool = multiprocessing.Pool(args.nproc)
        pool.map(_func, func_args)
    else:
        for i,ifilename in enumerate(args.infile) :
            simulate_one_healpix(ifilename,args,model,obsconditions,
                    decam_and_wise_filters,bassmzls_and_wise_filters,
                    footprint_healpix_weight,footprint_healpix_nside,
                    bal=bal,eboss=eboss)
