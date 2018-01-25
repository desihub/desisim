from __future__ import absolute_import, division, print_function
import sys, os
import numpy as np
from astropy.table import Table
import astropy.units as u

import specsim.simulator

from desispec.frame import Frame
import desispec.io
from desispec.resolution import Resolution

import desisim.io
import desisim.simexp
from desisim.util import dateobs2night
import desisim.specsim

#-------------------------------------------------------------------------

def parse(options=None):
    import argparse
    parser = argparse.ArgumentParser(usage = "{prog} [options]")
    parser.add_argument("--simspec", type=str,  help="input simspec file")
    parser.add_argument("--outdir", type=str,  help="output directory")
    parser.add_argument("--firstspec", type=int, default=0,
                        help="first spectrum to simulate")
    parser.add_argument("--nspec", type=int, default=5000,
                        help="number of spectra to simulate")
    parser.add_argument("--cframe", action="store_true",
                        help="directly write cframe")
    parser.add_argument("--dwave", type=float, default=0.8, help="output wavelength step, in Angstrom")
    
    if options is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(options)

    return args

def main(args=None):
    '''
    Converts simspec -> frame files; see fastframe --help for usage options
    '''
    #- TODO: use desiutil.log

    if isinstance(args, (list, tuple, type(None))):
        args = parse(args)
    
    print('Reading files')
    simspec = desisim.io.read_simspec(args.simspec)

    if simspec.flavor == 'arc':
        print('arc exposure; no frames to output')
        return

    fibermap = simspec.fibermap
    obs = simspec.obs
    night = simspec.header['NIGHT']
    expid = simspec.header['EXPID']

    firstspec = args.firstspec
    nspec = min(args.nspec, len(fibermap)-firstspec)

    print('Simulating spectra {}-{}'.format(firstspec, firstspec+nspec))
    wave = simspec.wave['brz']
    flux = simspec.flux
    ii = slice(firstspec, firstspec+nspec)
    if simspec.flavor == 'science':
        sim = desisim.simexp.simulate_spectra(wave, flux[ii],
                fibermap=fibermap[ii], obsconditions=obs, dwave_out=args.dwave)
    elif simspec.flavor in ['arc', 'flat', 'calib']:
        x = fibermap['X_TARGET']
        y = fibermap['Y_TARGET']
        fiber_area = desisim.simexp.fiber_area_arcsec2(
                fibermap['X_TARGET'], fibermap['Y_TARGET'])
        surface_brightness = (flux.T / fiber_area).T
        config = desisim.simexp._specsim_config_for_wave(wave, dwave_out=args.dwave)
        # sim = specsim.simulator.Simulator(config, num_fibers=nspec)
        sim = desisim.specsim.get_simulator(config, num_fibers=nspec)
        sim.observation.exposure_time = simspec.header['EXPTIME'] * u.s
        sbunit = 1e-17 * u.erg / (u.Angstrom * u.s * u.cm ** 2 * u.arcsec ** 2)
        xy = np.vstack([x, y]).T * u.mm
        sim.simulate(calibration_surface_brightness=surface_brightness[ii]*sbunit,
                     focal_positions=xy[ii])
    else:
        raise ValueError('Unknown simspec flavor {}'.format(simspec.flavor))

    sim.generate_random_noise()

    for i, results in enumerate(sim.camera_output):
        results = sim.camera_output[i]
        wave = results['wavelength']
        scale=1e17
        if args.cframe :
            phot = scale*(results['observed_flux'] + results['random_noise_electrons']*results['flux_calibration']).T
            ivar = 1./scale**2*results['flux_inverse_variance'].T
        else :
            phot = (results['num_source_electrons'] + \
                    results['num_sky_electrons'] + \
                    results['num_dark_electrons'] + \
                    results['random_noise_electrons']).T        
            ivar = 1.0 / results['variance_electrons'].T
        
        R = Resolution(sim.instrument.cameras[i].get_output_resolution_matrix())
        Rdata = np.tile(R.data.T, nspec).T.reshape(
                        nspec, R.data.shape[0], R.data.shape[1])
        assert np.all(Rdata[0] == R.data)
        assert phot.shape == (nspec, len(wave))
        for spectro in range(10):
            imin = max(firstspec, spectro*500) - firstspec
            imax = min(firstspec+nspec, (spectro+1)*500) - firstspec
            if imax <= imin:
                continue
            xphot = phot[imin:imax]
            xivar = ivar[imin:imax]
            xfibermap = fibermap[ii][imin:imax]
            camera = '{}{}'.format(sim.camera_names[i], spectro)
            meta = simspec.header.copy()
            meta['CAMERA'] = camera
            if args.cframe :
                units = '1e-17 erg/(s cm2 A)'
            else :
                #units = 'photon/bin'
                
                # we want to save electrons per angstrom and not electrons per bin
                # to be consistent with the extraction code (specter.extract.ex2d)
                units = 'electrons/Angstrom'
                dwave=np.gradient(wave)
                xphot /= dwave
                xivar *= dwave**2
            
            if 'BUNIT' in meta :
                meta['BUNIT']=units
            
            frame = Frame(wave, xphot, xivar, resolution_data=Rdata[0:imax-imin],
                          spectrograph=spectro, fibermap=xfibermap, meta=meta)
            if args.cframe :
                outfile = desispec.io.findfile('cframe', night, expid, camera,
                                               outdir=args.outdir)
            else :
                outfile = desispec.io.findfile('frame', night, expid, camera,
                                               outdir=args.outdir)
            print('writing {}'.format(outfile))
            desispec.io.write_frame(outfile, frame, units=units)
