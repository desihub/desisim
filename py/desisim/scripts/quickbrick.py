"""
desisim.scripts.quickbrick
==========================

Quickly generate DESI brick files.

TODO (@moustakas): Use the correct fiberloss file for BGS objects.
"""

from __future__ import division, print_function

import os
import sys
import numpy as np
import argparse

from astropy.table import Table, Column, vstack
import astropy.units as u
from astropy.io import fits

from specsim.simulator import Simulator
from desimodel.io import load_desiparams
from desispec.io import fitsheader, empty_fibermap, Brick
from desispec.resolution import Resolution
from desispec.log import get_logger, DEBUG
from desisim.targets import sample_objtype
from desisim.obs import get_night
import desisim.templates
import desiutil.io

def _add_truth(hdus, header, meta, trueflux, sflux, wave, channel):
    """Utility function for adding truth to an output FITS file."""
    hdus.append(
        fits.ImageHDU(trueflux[channel], name='_TRUEFLUX', header=header))
    if channel == 'b':
        swave = wave.astype(np.float32)
        hdus.append(fits.ImageHDU(swave, name='_SOURCEWAVE', header=header))
        hdus.append(fits.ImageHDU(sflux, name='_SOURCEFLUX', header=header))
        metatable = desiutil.io.encode_table(meta, encoding='ascii')
        metahdu = fits.convenience.table_to_hdu(meta)
        metahdu.header['EXTNAME'] = '_TRUTH'
        hdus.append(metahdu)

def parse(options=None):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Quickly generate brick files.')

    # Mandatory input
    parser.add_argument('-b', '--brickname', type=str, help='unique output brickname suffix (required input)', metavar='')

    # Simulation options
    parser.add_argument('--objtype', type=str,  help='ELG, LRG, QSO, BGS, MWS, WD, DARK_MIX, or BRIGHT_MIX', default='DARK_MIX', metavar='')
    parser.add_argument('--config', type=str, help='specsim configuration', default='desi', metavar='')
    parser.add_argument('-n', '--nspec', type=int,  help='number of spectra to simulate', default=100, metavar='')
    parser.add_argument('-a', '--airmass', type=float,  help='airmass', default=1.25, metavar='') # Science Req. Doc L3.3.2
    parser.add_argument('-e', '--exptime', type=float,  help='exposure time (s) (default based on config)', metavar='')
    parser.add_argument('-s', '--seed', type=int,  help='random seed', default=None, metavar='')
    parser.add_argument('-o', '--outdir', type=str,  help='output directory', default='.', metavar='')
    parser.add_argument('-v', '--verbose', action='store_true', help='toggle on verbose output')
    parser.add_argument('--outdir-truth', type=str,  help='optional alternative output directory for truth files', metavar='')

    # Object type-specific options.
    obj_parser = parser.add_argument_group('options for specific object types')
    obj_parser.add_argument('--zrange-qso', type=float, default=(0.5, 4.0), nargs=2, metavar='', 
                            help='minimum and maximum redshift range for QSO')
    obj_parser.add_argument('--zrange-elg', type=float, default=(0.6, 1.6), nargs=2, metavar='', 
                            help='minimum and maximum redshift range for ELG')
    obj_parser.add_argument('--zrange-lrg', type=float, default=(0.5, 1.1), nargs=2, metavar='', 
                            help='minimum and maximum redshift range for LRG')
    obj_parser.add_argument('--zrange-bgs', type=float, default=(0.01, 0.4), nargs=2, metavar='', 
                            help='minimum and maximum redshift range for BGS')
    obj_parser.add_argument('--rmagrange-bgs', type=float, default=(15.0, 19.5), nargs=2, metavar='',
                            help='Minimum and maximum BGS r-band (AB) magnitude range')

    obj_parser.add_argument('--sne-rfluxratiorange', type=float, default=(0.1, 1.0), nargs=2, metavar='', 
                            help='r-band flux ratio of the SNeIa spectrum relative to the galaxy')
    obj_parser.add_argument('--add-SNeIa', action='store_true', help='include SNeIa spectra')
  
    # Options corresponding to the bright-time survey only.
    bts_parser = parser.add_argument_group('options for Bright Time Surveys (BGS and MWS)')
    bts_parser.add_argument('--moon-phase', type=float,  help='moon phase (0=full, 1=new)', default=None, metavar='')
    bts_parser.add_argument('--moon-angle', type=float,  help='separation angle to the moon (0-180 deg)', default=None, metavar='')
    bts_parser.add_argument('--moon-zenith', type=float,  help='zenith angle of the moon (0-90 deg)', default=None, metavar='')

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)
    return args

def main(args):

    # Set up the logger.
    if args.verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    # Basic error checking.
    if args.brickname is None:
        log.critical('BRICKNAME input required')
        return -1

    known_objtype = ('ELG', 'LRG', 'QSO', 'BGS', 'MWS', 'WD', 'DARK_MIX', 'BRIGHT_MIX')
    if args.objtype.upper() not in known_objtype:
        log.critical('Unknown OBJTYPE {}'.format(args.objtype))
        return -1
        
    np.random.seed(args.seed)
    random_state = np.random.RandomState(args.seed)

    # Initialize the quick simulator object and its optional parameters.
    log.debug('Initializing specsim Simulator with configuration file {}'.format(args.config))
    desiparams = load_desiparams()
    qsim = Simulator(args.config)

    objtype = args.objtype.upper()
    log.debug('Using OBJTYPE {}'.format(objtype))
    if objtype == 'BGS' or objtype == 'MWS' or objtype == 'BRIGHT_MIX':
        qsim.observation.exposure_time = desiparams['exptime_bright'] * u.s
        if args.moon_phase is None:
            qsim.atmosphere.moon.moon_phase = 0.7
        else:
            qsim.atmosphere.moon.moon_phase = args.moon_phase
        if args.moon_angle is None:
            qsim.atmosphere.moon.separation_angle = 50 * u.deg
        else:
            qsim.atmosphere.moon.separation_angle = args.moon_angle * u.deg
        if args.moon_zenith is None:
            qsim.atmosphere.moon.moon_zenith = 30 * u.deg
        else:
            qsim.atmosphere.moon.moon_zenith = args.moon_zenith * u.deg
    else:
        qsim.observation.exposure_time = desiparams['exptime_dark'] * u.s

    if args.exptime is not None:
        qsim.observation.exposure_time = args.exptime * u.s

    qsim.atmosphere.airmass = args.airmass

    # Get the appropriate mixture of object types
    if objtype == 'DARK_MIX':
        true_objtype, target_objtype = sample_objtype(args.nspec, 'DARK')
    elif objtype == 'BRIGHT_MIX':
        true_objtype, target_objtype = sample_objtype(args.nspec, 'BRIGHT')
    else:
        true_objtype = np.tile(np.array([objtype]),(args.nspec))

    # Default output wavelength range.
    wave = qsim.source.wavelength_out.to(u.Angstrom).value
    npix = len(wave)

    # Initialize the output truth table.
    truth = dict()
    meta = Table()
    truth['OBJTYPE'] = np.zeros(args.nspec, dtype=(str, 10))
    truth['FLUX'] = np.zeros((args.nspec, npix))
    truth['WAVE'] = wave
    jj = list()

    for thisobj in set(true_objtype):
        ii = np.where(true_objtype == thisobj)[0]
        nobj = len(ii)
        truth['OBJTYPE'][ii] = thisobj

        # Generate the templates
        if thisobj == 'ELG':
            elg = desisim.templates.ELG(wave=wave, add_SNeIa=args.add_SNeIa)
            flux, tmpwave, meta1 = elg.make_templates(nmodel=nobj, seed=args.seed, zrange=args.zrange_elg,
                                                      sne_rfluxratiorange=args.sne_rfluxratiorange)
        elif thisobj == 'LRG':
            lrg = desisim.templates.LRG(wave=wave, add_SNeIa=args.add_SNeIa)
            flux, tmpwave, meta1 = lrg.make_templates(nmodel=nobj, seed=args.seed, zrange=args.zrange_lrg,
                                                      sne_rfluxratiorange=args.sne_rfluxratiorange)
        elif thisobj == 'QSO':
            qso = desisim.templates.QSO(wave=wave)
            flux, tmpwave, meta1 = qso.make_templates(nmodel=nobj, seed=args.seed, zrange=args.zrange_qso)
        elif thisobj == 'BGS':
            bgs = desisim.templates.BGS(wave=wave, add_SNeIa=args.add_SNeIa)
            flux, tmpwave, meta1 = bgs.make_templates(nmodel=nobj, seed=args.seed, zrange=args.zrange_bgs,
                                                      rmagrange=args.rmagrange_bgs,
                                                      sne_rfluxratiorange=args.sne_rfluxratiorange)
        elif thisobj =='STD':
            fstd = desisim.templates.FSTD(wave=wave)
            flux, tmpwave, meta1 = fstd.make_templates(nmodel=nobj, seed=args.seed)
        elif thisobj == 'QSO_BAD': # use STAR template no color cuts
            star = desisim.templates.STAR(wave=wave)
            flux, tmpwave, meta1 = star.make_templates(nmodel=nobj, seed=args.seed)
        elif thisobj == 'MWS_STAR' or thisobj == 'MWS':
            mwsstar = desisim.templates.MWS_STAR(wave=wave)
            flux, tmpwave, meta1 = mwsstar.make_templates(nmodel=nobj, seed=args.seed)
        elif thisobj == 'SKY':
            flux = np.zeros((nobj, npix))
            meta1 = Table(dict(REDSHIFT=np.zeros(nobj, dtype=np.float32)))
        elif thisobj == 'TEST':
            flux = np.zeros((args.nspec, npix))
            indx = np.where(wave>5800.0-1E-6)[0][0]
            ref_integrated_flux = 1E-10
            ref_cst_flux_density = 1E-17
    
            single_line = (np.arange(args.nspec)%2 == 0).astype(np.float32)
            continuum   = (np.arange(args.nspec)%2 == 1).astype(np.float32)
    
            for spec in range(args.nspec) :
                flux[spec,indx] = single_line[spec]*ref_integrated_flux/np.gradient(wave)[indx] # single line
                flux[spec] += continuum[spec]*ref_cst_flux_density # flat continuum
    
            meta1 = Table(dict(REDSHIFT=np.zeros(args.nspec, dtype=np.float32),
                               LINE=wave[indx]*np.ones(args.nspec, dtype=np.float32),
                               LINEFLUX=single_line*ref_integrated_flux,
                               CONSTFLUXDENSITY=continuum*ref_cst_flux_density))
        else:
            log.fatal('Unknown object type {}'.format(thisobj))
            sys.exit(1)

        # Pack it in.
        truth['FLUX'][ii] = flux
        meta = vstack([meta, meta1])
        jj.append(ii.tolist())
        
        # Sanity check on units; templates currently return ergs, not 1e-17 ergs...
        assert (thisobj == 'SKY') or (np.max(truth['FLUX']) < 1e-6)

    # Create a blank fake fibermap; this is re-used by all channels.
    fibermap = empty_fibermap(args.nspec)
    targetids = random_state.randint(2**62, size=args.nspec)
    fibermap['TARGETID'] = targetids
    night = get_night()
    expid = 0

    # Sort the metadata table.
    jj = sum(jj,[])
    meta_new = Table()
    for k in range(args.nspec):
        index = int(np.where(np.array(jj) == k)[0])
        meta_new = vstack([meta_new, meta[index]])
    meta = meta_new

    # Add TARGETID and the true OBJTYPE to the metadata table.
    meta.add_column(Column(true_objtype, dtype=(str, 10), name='TRUE_OBJTYPE'))
    meta.add_column(Column(targetids, name='TARGETID'))

    # Rename REDSHIFT -> TRUEZ anticipating later table joins with zbest.Z
    meta.rename_column('REDSHIFT', 'TRUEZ')

    # Initialize per-camera output arrays that will be saved to the brick files.
    cwave, trueflux, noisyflux, obsivar, resolution, sflux = {}, {}, {}, {}, {}, {}
    for camera in qsim.instrument.cameras:
        cwave[camera.name] = (
            camera.output_wavelength.to(u.Angstrom).value.astype(np.float32))
        nwave = len(cwave[camera.name])
        trueflux[camera.name] = np.empty((args.nspec, nwave), dtype=np.float32)
        noisyflux[camera.name] = np.empty_like(trueflux[camera.name])
        obsivar[camera.name] = np.empty_like(trueflux[camera.name])
        # Lookup this camera's resolution matrix and convert to the sparse
        # format used in desispec.
        R = Resolution(camera.get_output_resolution_matrix())
        resolution[camera.name] = np.tile(R.to_fits_array(), [args.nspec, 1, 1])
        # Source flux uses the high-resolution simulation grid.
        sflux = np.empty((args.nspec, npix), dtype=np.float32)

    # Actually do the simulations for each target
    fluxunits = u.erg / (u.s * u.cm ** 2 * u.Angstrom)
    for ii in range(args.nspec):
        # If objtype is QSO_BAD or TEST then simulate a star
        if (truth['OBJTYPE'][ii] == 'MWS' or truth['OBJTYPE'][ii] == 'MWS_STAR' or \
            truth['OBJTYPE'][ii] == 'STD' or truth['OBJTYPE'][ii] == 'QSO_BAD' or \
            truth['OBJTYPE'][ii] == 'TEST'):
            thisobjtype = 'STAR'
        elif truth['OBJTYPE'][ii] == 'BGS':
            thisobjtype = 'ELG' # TODO (@moustakas): Fix this!
        else:
            thisobjtype = truth['OBJTYPE'][ii]

        # Update the source model to simulate.
        qsim.source.update_in(
            'Quickbrick source {0}'.format(ii), thisobjtype.lower(),
            truth['WAVE'] * u.Angstrom, truth['FLUX'][ii] * fluxunits)
        qsim.source.update_out()
        sflux[ii][:] = 1e17 * qsim.source.flux_out.to(fluxunits).value

        # Run the simulation for this source and add noise.
        qsim.simulate()
        qsim.generate_random_noise(random_state)

        # Fill brick arrays from the results.
        for camera, output in zip(qsim.instrument.cameras, qsim.camera_output):
            assert output['observed_flux'].unit == fluxunits
            trueflux[camera.name][ii][:] = 1e17 * output['observed_flux']
            noisyflux[camera.name][ii][:] = 1e17 * (
                output['observed_flux'] +
                output['flux_calibration'] * output['random_noise_electrons'])
            obsivar[camera.name][ii][:] = 1e-34 * output['flux_inverse_variance']

    # Write brick output.
    for channel in 'brz':
        filename = 'brick-{}-{}.fits'.format(channel, args.brickname)
        filepath = os.path.join(args.outdir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        log.debug('Writing {}'.format(filepath))

        header = dict(BRICKNAM=args.brickname, CHANNEL=channel)
        brick = Brick(filepath, mode='update', header=header)
        brick.add_objects(
            noisyflux[channel], obsivar[channel],
            cwave[channel], resolution[channel], fibermap, night, expid)
        brick.close()

        # Append truth to the file. Note: we add the resolution-convolved true
        # flux, not the high resolution source flux, which makes chi2
        # calculations easier.
        header = fitsheader(header)
        if args.outdir_truth is None : # add truth in same file
            fx = fits.open(filepath, mode='append')
            _add_truth(fx, header, meta, trueflux, sflux, wave, channel)
            fx.flush()
            fx.close()
        else:
            filename = 'truth-brick-{}-{}.fits'.format(channel, args.brickname)
            filepath = os.path.join(args.outdir_truth, filename)
            hdulist = fits.HDUList([fits.PrimaryHDU(header=header)])
            _add_truth(hdulist, header, meta, trueflux, sflux, wave, channel)
            hdulist.writeto(filepath, clobber=True)

