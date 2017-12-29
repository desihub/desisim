"""
desisim.io
==========

I/O routines for desisim
"""

from __future__ import absolute_import, division, print_function

import os
import time
from glob import glob
import warnings

from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import numpy as np

from desispec.interpolation import resample_flux
from desispec.io.util import write_bintable, native_endian, header2wave
import desispec.io
import desimodel.io

from desispec.image import Image
import desispec.io.util

from desiutil.log import get_logger
log = get_logger()

from desisim.util import spline_medfilt2d

#-------------------------------------------------------------------------
def findfile(filetype, night, expid, camera=None, outdir=None, mkdir=True):
    """Return canonical location of where a file should be on disk

    Args:
        filetype (str): file type, e.g. 'pix' or 'pixsim'
        night (str): YEARMMDD string
        expid (int): exposure id integer
        camera (str): e.g. 'b0', 'r1', 'z9'
        outdir (Optional[str]): output directory; defaults to $DESI_SPECTRO_SIM/$PIXPROD
        mkdir (Optional[bool]): create output directory if needed; default True

    Returns:
        str: full file path to output file

    Also see desispec.io.findfile() which has equivalent functionality for
    real data files; this function is only be for simulation files.
    """

    #- outdir default = $DESI_SPECTRO_SIM/$PIXPROD/{night}/
    if outdir is None:
        outdir = simdir(night)

    #- Definition of where files go
    location = dict(
        simspec = '{outdir:s}/simspec-{expid:08d}.fits',
        simpix = '{outdir:s}/simpix-{expid:08d}.fits',
        simfibermap = '{outdir:s}/fibermap-{expid:08d}.fits',
        pix = '{outdir:s}/pix-{camera:s}-{expid:08d}.fits',
        fastframelog = '{outdir:s}/fastframe-{expid:08d}.log',
        newexplog = '{outdir:s}/newexp-{expid:08d}.log',
    )

    #- Do we know about this kind of file?
    if filetype not in location:
        raise ValueError("Unknown filetype {}; known types are {}".format(filetype, list(location.keys())))

    #- Some but not all filetypes require camera
    if filetype == 'pix' and camera is None:
        raise ValueError('camera is required for filetype '+filetype)

    #- get outfile location and cleanup extraneous // from path
    outfile = location[filetype].format(
        outdir=outdir, night=night, expid=expid, camera=camera)
    outfile = os.path.normpath(outfile)

    #- Create output directory path if needed
    #- Do this only after confirming that all previous parsing worked
    if mkdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    return outfile


#-------------------------------------------------------------------------
#- simspec

def write_simspec(sim, truth, fibermap, obs, expid, night, outdir=None, filename=None,
    header=None, overwrite=False):
    '''
    Write a simspec file

    Args:
        sim: specsim Simulator object
        truth: truth metadata Table
        fibermap: fibermap Table
        obs: dict-like observation conditions with keys
            SEEING (arcsec), EXPTIME (sec), AIRMASS,
            MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
        expid: integer exposure ID
        night: YEARMMDD string

    Options:
        outdir: output directory
        filename: if None, auto-derive from envvars, night, expid, and outdir
        header: dict-like header to include in HDU0
        overwrite: overwrite pre-existing files
    
    Notes:
        calibration exposures can use truth=None and obs=None
    '''
    import astropy.table
    import astropy.units as u
    import desiutil.depend
    from desiutil.log import get_logger
    log = get_logger()

    if filename is None:
        filename = findfile('simspec', night, expid, outdir=outdir)

    # sim.simulated is table of pre-convolution quantities that we want
    # to ouput.  sim.camera_output is post-convolution.

    #- Create HDU 0 header with keywords to propagate
    header = desispec.io.util.fitsheader(header)
    desiutil.depend.add_dependencies(header)
    header['EXPID'] = expid
    header['NIGHT'] = night
    header['EXPTIME'] = sim.observation.exposure_time.to('s').value
    if obs is not None:
        try:
            keys = obs.keys()
        except AttributeError:
            keys = obs.dtype.names

        for key in keys:
            shortkey = key[0:8]  #- FITS keywords can only be 8 char
            if shortkey not in header:
                header[shortkey] = obs[key]
    if 'DOSVER' not in header:
        header['DOSVER'] = 'SIM'
    if 'FEEVER' not in header:
        header['FEEVER'] = 'SIM'

    if 'FLAVOR' not in header:
        log.warning('FLAVOR not provided; guessing "science"')
        header['FLAVOR'] = 'science'    #- optimistically guessing

    if 'DATE-OBS' not in header:
        header['DATE-OBS'] = sim.observation.exposure_start.isot

    log.info('DATE-OBS {} UTC'.format(header['DATE-OBS']))

    #- Check truth and obs for science exposures
    if header['FLAVOR'] == 'science':
        if obs is None:
            raise ValueError('obs Table must be included for science exposures')
        if truth is None:
            raise ValueError('truth Table must be included for science exposures')

    hx = fits.HDUList()
    header['EXTNAME'] = 'WAVE'
    header['BUNIT'] = 'Angstrom'
    header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')

    wave = sim.simulated['wavelength'].to('Angstrom').value
    hx.append(fits.PrimaryHDU(wave, header=header))

    fluxunits = 1e-17 * u.erg / (u.s * u.cm**2 * u.Angstrom)
    flux32 = sim.simulated['source_flux'].to(fluxunits).astype(np.float32).value.T
    nspec = flux32.shape[0]
    assert flux32.shape == (nspec, wave.shape[0])
    hdu_flux = fits.ImageHDU(flux32, name='FLUX')
    hdu_flux.header['BUNIT'] = str(fluxunits)
    hx.append(hdu_flux)

    #- sky_fiber_flux is not flux per fiber area, it is normal flux density
    skyflux32 = sim.simulated['sky_fiber_flux'].to(fluxunits).astype(np.float32).value.T
    assert skyflux32.shape == (nspec, wave.shape[0])
    hdu_skyflux = fits.ImageHDU(skyflux32, name='SKYFLUX')
    hdu_skyflux.header['BUNIT'] = str(fluxunits)
    hx.append(hdu_skyflux)

    #- DEPRECATE?  per-camera photons (derivable from flux and throughput)
    for i, camera in enumerate(sorted(sim.camera_names)):
        wavemin = sim.camera_output[i]['wavelength'][0]
        wavemax = sim.camera_output[i]['wavelength'][-1]
        ii = (wavemin <= wave) & (wave <= wavemax)
        hx.append(fits.ImageHDU(wave[ii], name='WAVE_'+camera.upper()))

        phot32 = sim.simulated['num_source_electrons_'+camera][ii].astype(np.float32).T

        assert phot32.shape == (nspec, wave[ii].shape[0])
        hdu_phot = fits.ImageHDU(phot32, name='PHOT_'+camera.upper())
        hdu_phot.header['BUNIT'] = 'photon'
        hx.append(hdu_phot)

        skyphot32 = sim.simulated['num_sky_electrons_'+camera][ii].astype(np.float32).T
        assert skyphot32.shape == (nspec, wave[ii].shape[0])
        hdu_skyphot = fits.ImageHDU(skyphot32, name='SKYPHOT_'+camera.upper())
        hdu_skyphot.header['BUNIT'] = 'photon'
        hx.append(hdu_skyphot)

    #- TRUTH HDU: table with truth metadata
    if truth is not None:
        assert len(truth) == nspec
        truthhdu = fits.table_to_hdu(Table(truth))
        truthhdu.header['EXTNAME'] = 'TRUTH'
        hx.append(truthhdu)

    #- FIBERMAP HDU
    assert len(fibermap) == nspec
    fibermap_hdu = fits.table_to_hdu(Table(fibermap))
    fibermap_hdu.header['EXTNAME'] = 'FIBERMAP'
    hx.append(fibermap_hdu)

    #- OBSCONDITIONS HDU: Table with 1 row with observing conditions
    #- is None for flat calibration calibration exposures
    if obs is not None:
        if isinstance(obs, astropy.table.Row):
            obstable = astropy.table.Table(obs)
        else:
            obstable = astropy.table.Table([obs,])
        obs_hdu = fits.table_to_hdu(obstable)
        obs_hdu.header['EXTNAME'] = 'OBSCONDITIONS'
        hx.append(obs_hdu)

    log.info('Writing {}'.format(filename))
    hx.writeto(filename, clobber=overwrite)

def write_simspec_arc(filename, wave, phot, header, fibermap, overwrite=False):
    '''
    Alternate writer for arc simspec files which just have photons
    '''
    import astropy.table
    import astropy.units as u

    hx = fits.HDUList()
    hdr = desispec.io.util.fitsheader(header)
    hdr['FLAVOR'] = 'arc'
    if 'DOSVER' not in hdr:
        hdr['DOSVER'] = 'SIM'
    if 'FEEVER' not in header:
        hdr['FEEVER'] = 'SIM'

    hx.append(fits.PrimaryHDU(None, header=hdr))

    for camera in ['b', 'r', 'z']:
        thru = desimodel.io.load_throughput(camera)
        ii = (thru.wavemin <= wave) & (wave <= thru.wavemax)
        hdu_wave = fits.ImageHDU(wave[ii], name='WAVE_'+camera.upper())
        hdu_wave.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
        hx.append(hdu_wave)

        phot32 = phot[:,ii].astype(np.float32)
        hdu_phot = fits.ImageHDU(phot32, name='PHOT_'+camera.upper())
        hdu_phot.header['BUNIT'] = 'photon'
        hx.append(hdu_phot)

    #- FIBERMAP HDU
    fibermap_hdu = fits.table_to_hdu(fibermap)
    fibermap_hdu.header['EXTNAME'] = 'FIBERMAP'
    hx.append(fibermap_hdu)

    log.info('Writing {}'.format(filename))
    hx.writeto(filename, clobber=overwrite)
    return filename


class SimSpec(object):
    """Lightweight wrapper object for simspec data.

    Args:
        flavor (str): e.g. 'arc', 'flat', 'dark', 'mws', ...
        wave : dictionary with per-channel wavelength grids, keyed by
            'b', 'r', 'z'.  Optionally also has 'brz' key for channel
            independent wavelength grid
        phot : dictionary with per-channel photon counts per bin

    Optional:
        flux : channel-independent flux [erg/s/cm^2/A]
        skyflux : channel-indepenent sky flux [erg/s/cm^2/A/arcsec^2]
        skyphot : dictionary with per-channel sky photon counts per bin
        metadata : table of metadata information about these spectra
        header : FITS header from HDU0
        fibermap : fibermap Table
        obs : (dict-like) observing conditions; see keys in notes below

    Notes:
      * input arguments become attributes
      * wave[channel] is the wavelength grid for phot[channel] and
            skyphot[channel] where channel = 'b', 'r', or 'z'
      * wave['brz'] is the wavelength grid for flux and skyflux
      * obsconditions keys SEEING (arcsec), EXPTIME (sec), AIRMASS,
        MOONFRAC (0-1), MOONALT (deg), MOONSEP (deg)
    """
    def __init__(self, flavor, wave, phot, flux=None, skyflux=None,
                 skyphot=None, metadata=None, fibermap=None, obs=None, header=None):
        for channel in ('b', 'r', 'z'):
            assert wave[channel].ndim == 1
            assert phot[channel].ndim == 2
            assert wave[channel].shape[0] == phot[channel].shape[1]

        assert phot['b'].shape[0] == phot['r'].shape[0] == phot['z'].shape[0]

        self.flavor = flavor
        self.nspec = phot['b'].shape[0]
        self.wave = wave
        self.phot = phot

        #- Optional items; may be None
        self.skyphot = skyphot
        self.flux = flux
        self.skyflux = skyflux
        self.metadata = metadata
        self.fibermap = fibermap
        self.obs = obs
        self.header = header


def read_simspec_mpi(filename, comm, spectrographs=None):
    """
    Read simspec data from filename and return SimSpec object.
    """
    import astropy.table

    # rank 0 opens file and gets the metadata and wavelength
    # grids, which will be kept on all processes.

    hdr = None
    flavor = None
    fibermap = None
    obs = None
    wave = dict()
    totalspec = None

    if comm.rank == 0:
        fx = fits.open(filename, memmap=True)
        hdr = fx[0].header.copy()
        flavor = hdr['FLAVOR']
        if 'WAVE' in fx:
            wave['brz'] = native_endian(fx['WAVE'].data.copy())
        for channel in ('b', 'r', 'z'):
            hname = 'WAVE_'+channel.upper()
            wave[channel] = native_endian(fx[hname].data.copy())
        if 'FIBERMAP' in fx:
            fibermap = astropy.table.Table(fx['FIBERMAP'].data.copy())
            totalspec = len(fibermap)
        else:
            # Get the number of spectra from one of the photon HDUs
            totalspec = fx['PHOT_B'].header['NAXIS2']
        if 'OBSCONDITIONS' in fx:
            obs = astropy.table.Table(fx['OBSCONDITIONS'].data.copy())[0]
        fx.close()
        # Memmap file handle should close here when fx goes out of scope...

    hdr = comm.bcast(hdr, root=0)
    flavor = comm.bcast(flavor, root=0)
    obs = comm.bcast(obs, root=0)
    totalspec = comm.bcast(totalspec, root=0)
    wave = comm.bcast(wave, root=0)
    fibermap = comm.bcast(fibermap, root=0)

    # Based on the spectrographs for this process, compute the range of 
    # spectra we need to store.  The number of spectra per spectrograph (500)
    # is hard-coded several places in desisim.  This should be put in desimodel
    # somewhere...

    fibers = None
    if fibermap is not None:
        fibers = np.array(fibermap['FIBER'], dtype=np.int32)
    else:
        fibers = np.arange(totalspec, dtype=np.int32)

    if spectrographs is None:
        spectrographs = np.arange(10, dtype=np.int32)

    specslice = np.in1d(fibers//500, spectrographs)

    if fibermap is not None:
        fibermap = fibermap[specslice]

    # Now read one HDU at a time, broadcast, and every process grabs its slice.

    # Note: this is global scope within the function, so memmap file handle
    # will not close until the function returns (which is fine).
    hdus = None

    if comm.rank == 0:
        hdus = fits.open(filename, memmap=True)

    # Read photons

    phot = dict()
    for channel in ('b', 'r', 'z'):
        hname = 'PHOT_'+channel.upper()
        hdata = None
        if comm.rank == 0:
            hdata = native_endian(hdus[hname].data.copy().astype('f8'))
        hdata = comm.bcast(hdata, root=0)
        phot[channel] = hdata[specslice].copy()
        del hdata

    # Read sky photons

    skyphot = dict()
    for channel in ('b', 'r', 'z'):
        hname = 'SKYPHOT_'+channel.upper()
        found = False
        if comm.rank == 0:
            if hname in hdus:
                found = True
        found = comm.bcast(found, root=0)
        if found:
            hdata = None
            if comm.rank == 0:
                hdata = native_endian(hdus[hname].data.copy().astype('f8'))
            hdata = comm.bcast(hdata, root=0)
            skyphot[channel] = hdata[specslice].copy()
            del hdata
        else:
            skyphot[channel] = np.zeros_like(phot[channel])
        assert phot[channel].shape == skyphot[channel].shape

    # flux

    flux = None
    hname = 'FLUX'
    found = False
    if comm.rank == 0:
        if hname in hdus:
            found = True
    found = comm.bcast(found, root=0)
    if found:
        hdata = None
        if comm.rank == 0:
            hdata = native_endian(hdus[hname].data.copy().astype('f8'))
        hdata = comm.bcast(hdata, root=0)
        flux = hdata[specslice].copy()
        del hdata

    # skyflux

    skyflux = None
    hname = 'SKYFLUX'
    found = False
    if comm.rank == 0:
        if hname in hdus:
            found = True
    found = comm.bcast(found, root=0)
    if found:
        hdata = None
        if comm.rank == 0:
            hdata = native_endian(hdus[hname].data.copy().astype('f8'))
        hdata = comm.bcast(hdata, root=0)
        skyflux = hdata[specslice].copy()
        del hdata

    # metadata / truth

    metadata = None
    hname = 'TRUTH'
    found = False
    if comm.rank == 0:
        if hname in hdus:
            found = True
    found = comm.bcast(found, root=0)
    if not found:
        hname = 'METADATA'
        if comm.rank == 0:
            if hname in hdus:
                found = True
        found = comm.bcast(found, root=0)
    if found:
        hdata = None
        if comm.rank == 0:
            hdata = astropy.table.Table(hdus[hname].data.copy())
        hdata = comm.bcast(hdata, root=0)
        metadata = hdata[specslice].copy()
        del hdata

    if comm.rank == 0:
        hdus.close()

    return SimSpec(flavor, wave, phot, flux=flux, skyflux=skyflux,
        skyphot=skyphot, metadata=metadata, fibermap=fibermap, obs=obs,
        header=hdr)


def read_simspec(filename, nspec=None, firstspec=0):
    """Read simspec data from filename and return SimSpec object.
    """
    import astropy.table
    with fits.open(filename, memmap=False) as fx:
        hdr = fx[0].header
        flavor = hdr['FLAVOR']

        #- All flavors have photons
        wave = dict()
        phot = dict()
        skyphot = dict()
        for channel in ('b', 'r', 'z'):
            wave[channel] = native_endian(fx['WAVE_'+channel.upper()].data)
            phot[channel] = native_endian(fx['PHOT_'+channel.upper()].data.astype('f8'))

            skyext = 'SKYPHOT_'+channel.upper()
            if skyext in fx:
                skyphot[channel] = native_endian(fx[skyext].data.astype('f8'))
            else:
                skyphot[channel] = np.zeros_like(phot[channel])

            assert phot[channel].shape == skyphot[channel].shape

        #- Check for flux, skyflux, and metadata
        flux = None
        skyflux = None
        if 'WAVE' in fx:
            wave['brz'] = native_endian(fx['WAVE'].data)
        if 'FLUX' in fx:
            flux = native_endian(fx['FLUX'].data.astype('f8'))
        if 'SKYFLUX' in fx:
            skyflux = native_endian(fx['SKYFLUX'].data.astype('f8'))

        if 'TRUTH' in fx:
            metadata = astropy.table.Table(fx['TRUTH'].data)
        #- For backwards compatibility
        elif 'METADATA' in fx:
            metadata = astropy.table.Table(fx['METADATA'].data)
        else:
            metadata = None

        if 'FIBERMAP' in fx:
            fibermap = astropy.table.Table(fx['FIBERMAP'].data)
        else:
            fibermap = None

        if 'OBSCONDITIONS' in fx:
            obs = astropy.table.Table(fx['OBSCONDITIONS'].data)[0]
        else:
            obs = None

    #- Trim down if requested
    if nspec is None:
        nspec = phot['b'].shape[0]

    if firstspec > 0 or firstspec+nspec<phot['b'].shape[0]:
        for channel in ('b', 'r', 'z'):
            phot[channel] = phot[channel][firstspec:firstspec+nspec]
            skyphot[channel] = skyphot[channel][firstspec:firstspec+nspec]
        if flux is not None:
            flux = flux[firstspec:firstspec+nspec]
        if skyflux is not None:
            skyflux = skyflux[firstspec:firstspec+nspec]
        if metadata is not None:
            metadata = metadata[firstspec:firstspec+nspec]

    return SimSpec(flavor, wave, phot, flux=flux, skyflux=skyflux,
        skyphot=skyphot, metadata=metadata, fibermap=fibermap, obs=obs,
        header=hdr)


def _read_simspec_orig(filename):
    """Read simspec data from filename and return SimSpec object.
    """

    fx = fits.open(filename)
    hdr = fx[0].header
    flavor = hdr['FLAVOR']

    #- All flavors have photons
    wave = dict()
    phot = dict()
    skyphot = dict()
    for channel in ('b', 'r', 'z'):
        wave[channel] = fx['WAVE_'+channel.upper()].data
        phot[channel] = fx['PHOT_'+channel.upper()].data
        skyext = 'SKYPHOT_'+channel.upper()
        if skyext in fx:
            skyphot[channel] = fx[skyext].data
        else:
            skyphot[channel] = np.zeros_like(phot[channel])

    if flavor == 'arc':
        fx.close()
        return SimSpec(flavor, wave, phot, skyphot=skyphot, header=hdr)

    elif flavor == 'flat':
        wave['brz'] = fx['WAVE'].data
        flux = fx['FLUX'].data
        fx.close()
        return SimSpec(flavor, wave, phot, skyphot=skyphot, flux=flux, header=hdr)

    else:  #- multiple science flavors: dark, bright, bgs, mws, etc.
        wave['brz'] = fx['WAVE'].data
        flux = fx['FLUX'].data
        metadata = fx['METADATA'].data
        skyflux = fx['SKYFLUX'].data
        skyphot = dict()
        for channel in ('b', 'r', 'z'):
            extname = 'SKYPHOT_'+channel.upper()
            skyphot[channel] = fx[extname].data

        fx.close()
        return SimSpec(flavor, wave, phot, flux=flux, skyflux=skyflux,
            skyphot=skyphot, metadata=metadata, header=hdr)


def write_simpix(outfile, image, camera, meta):
    """Write simpix data to outfile.

    Args:
        outfile : output file name, e.g. from io.findfile('simpix', ...)
        image : 2D noiseless simulated image (numpy.ndarray)
        meta : dict-like object that should include FLAVOR and EXPTIME,
            e.g. from HDU0 FITS header of input simspec file
    """

    meta = desispec.io.util.fitsheader(meta)

    #- Create a new file with a blank primary HDU if needed
    if not os.path.exists(outfile):
        header = meta.copy()
        try:
            import specter
            header['DEPNAM00'] = 'specter'
            header['DEPVER00'] = (specter.__version__, 'Specter version')
        except ImportError:
            pass

        fits.PrimaryHDU(None, header=header).writeto(outfile)

    #- Add the new HDU
    hdu = fits.ImageHDU(image.astype(np.float32), header=meta, name=camera.upper())
    hdus = fits.open(outfile, mode='append', memmap=False)
    hdus.append(hdu)
    hdus.flush()
    hdus.close()

def load_simspec_summary(indir, verbose=False):
    '''
    Combine fibermap and simspec files under indir into single truth catalog

    Args:
        indir: path to input directory; search this and all subdirectories

    Returns:
        astropy.table.Table with true Z catalog
    '''
    import astropy.table
    truth = list()
    for fibermapfile in desispec.io.iterfiles(indir, 'fibermap'):
        fibermap = astropy.table.Table.read(fibermapfile, 'FIBERMAP')
        if verbose:
            print('')
        #- skip calibration frames
        if 'FLAVOR' in fibermap.meta:
            if fibermap.meta['FLAVOR'].lower() in ('arc', 'flat', 'bias'):
                continue
        elif 'OBSTYPE' in fibermap.meta:
            if fibermap.meta['OBSTYPE'].lower() in ('arc', 'flat', 'bias', 'dark'):
                continue

        simspecfile = fibermapfile.replace('fibermap-', 'simspec-')
        if not os.path.exists(simspecfile):
            raise IOError('fibermap without matching simspec: {}'.format(fibermapfile))

        simspec = astropy.table.Table.read(simspecfile, 'METADATA')

        #- cleanup prior to merging
        if 'REDSHIFT' in simspec.colnames:
            simspec.rename_column('REDSHIFT', 'TRUEZ')
        if 'OBJTYPE' in simspec.colnames:
            simspec.rename_column('OBJTYPE', 'TRUETYPE')
        for key in ('DATASUM', 'CHECKSUM', 'TELRA', 'TELDEC', 'EXTNAME'):
            if key in fibermap.meta:
                del fibermap.meta[key]
            if key in simspec.meta:
                del simspec.meta[key]

        #- convert some header keywords to new columns
        for key in ('TILEID', 'EXPID', 'FLAVOR', 'NIGHT'):
            fibermap[key] = fibermap.meta[key]
            del fibermap.meta[key]

        truth.append(astropy.table.hstack([fibermap, simspec]))

    truth = astropy.table.vstack(truth)
    return truth


#-------------------------------------------------------------------------
#- Cosmics

#- Utility function to resize an image while preserving its 2D arrangement
#- (unlike np.resize)
def _resize(image, shape):
    """
    Resize input image to have new shape, preserving its 2D arrangement

    Args:
        image : 2D ndarray
        shape : tuple (ny,nx) for desired output shape

    Returns:
        new image with image.shape == shape
    """

    #- Tile larger in odd integer steps so that sub-/super-selection can
    #- be centered on the input image
    fx = shape[1] / image.shape[1]
    fy = shape[0] / image.shape[0]
    nx = int(2*np.ceil( (fx-1) / 2) + 1)
    ny = int(2*np.ceil( (fy-1) / 2) + 1)

    newpix = np.tile(image, (ny, nx))
    ix = newpix.shape[1] // 2 - shape[1] // 2
    iy = newpix.shape[0] // 2 - shape[0] // 2
    return newpix[iy:iy+shape[0], ix:ix+shape[1]]

def find_cosmics(camera, exptime=1000, cosmics_dir=None):
    '''
    Return full path to cosmics template file to use

    Args:
        camera (str): e.g. 'b0', 'r1', 'z9'
        exptime (int, optional): exposure time in seconds
        cosmics_dir (str, optional): directory to look for cosmics templates; defaults to
            $DESI_COSMICS_TEMPLATES if set or otherwise
            $DESI_ROOT/spectro/templates/cosmics/v0.2  (note HARDCODED version)

    Exposure times <120 sec will use the bias templates; otherwise they will
    use the dark cosmics templates
    '''
    if cosmics_dir is None:
        if 'DESI_COSMICS_TEMPLATES' in os.environ:
            cosmics_dir = os.environ['DESI_COSMICS_TEMPLATES']
        else:
            cosmics_dir = os.environ['DESI_ROOT']+'/spectro/templates/cosmics/v0.2/'

    if exptime < 120:
        exptype = 'bias'
    else:
        exptype = 'dark'

    channel = camera[0].lower()
    assert channel in 'brz', 'Unknown camera {}'.format(camera)

    cosmicsfile = '{}/cosmics-{}-{}.fits'.format(cosmics_dir, exptype, channel)
    return os.path.normpath(cosmicsfile)

def read_cosmics(filename, expid=1, shape=None, jitter=True):
    """
    Reads a dark image with cosmics from the input filename.

    The input might have multiple dark images; use the `expid%n` image where
    `n` is the number of images in the input cosmics file.

    Args:
        filename : FITS filename with EXTNAME=IMAGE-*, IVAR-*, MASK-* HDUs
        expid : integer, use `expid % n` image where `n` is number of images
        shape : (ny, nx, optional) tuple for output image shape
        jitter (bool, optional): If True (default), apply random flips and rolls so you
            don't get the exact same cosmics every time

    Returns:
        `desisim.image.Image` object with attributes pix, ivar, mask
    """
    fx = fits.open(filename)
    imagekeys = list()
    for i in range(len(fx)):
        if fx[i].name.startswith('IMAGE-'):
            imagekeys.append(fx[i].name.split('-', 1)[1])

    assert len(imagekeys) > 0, 'No IMAGE-* extensions found in '+filename
    i = expid % len(imagekeys)
    pix  = native_endian(fx['IMAGE-'+imagekeys[i]].data.astype(np.float64))
    ivar = native_endian(fx['IVAR-'+imagekeys[i]].data.astype(np.float64))
    mask = native_endian(fx['MASK-'+imagekeys[i]].data)
    meta = fx['IMAGE-'+imagekeys[i]].header
    meta['CRIMAGE'] = (imagekeys[i], 'input cosmic ray image')

    #- De-trend each amplifier
    nx = pix.shape[1] // 2
    ny = pix.shape[0] // 2
    kernel_size = min(201, ny//3, nx//3)

    pix[0:ny, 0:nx] -= spline_medfilt2d(pix[0:ny, 0:nx], kernel_size)
    pix[0:ny, nx:2*nx] -= spline_medfilt2d(pix[0:ny, nx:2*nx], kernel_size)
    pix[ny:2*ny, 0:nx] -= spline_medfilt2d(pix[ny:2*ny, 0:nx], kernel_size)
    pix[ny:2*ny, nx:2*nx] -= spline_medfilt2d(pix[ny:2*ny, nx:2*nx], kernel_size)

    if shape is not None:
        if len(shape) != 2: raise ValueError('Invalid shape {}'.format(shape))
        pix = _resize(pix, shape)
        ivar = _resize(ivar, shape)
        mask = _resize(mask, shape)

    if jitter:
        #- Randomly flip left-right and/or up-down
        if np.random.uniform(0, 1) > 0.5:
            pix = np.fliplr(pix)
            ivar = np.fliplr(ivar)
            mask = np.fliplr(mask)
            meta['CRFLIPLR'] = (True, 'Input cosmics image flipped Left/Right')
        else:
            meta['CRFLIPLR'] = (False, 'Input cosmics image NOT flipped Left/Right')

        if np.random.uniform(0, 1) > 0.5:
            pix = np.flipud(pix)
            ivar = np.flipud(ivar)
            mask = np.flipud(mask)
            meta['CRFLIPUD'] = (True, 'Input cosmics image flipped Up/Down')
        else:
            meta['CRFLIPUD'] = (False, 'Input cosmics image NOT flipped Up/Down')

        #- Randomly roll image a bit
        nx, ny = np.random.randint(-100, 100, size=2)
        pix = np.roll(np.roll(pix, ny, axis=0), nx, axis=1)
        ivar = np.roll(np.roll(ivar, ny, axis=0), nx, axis=1)
        mask = np.roll(np.roll(mask, ny, axis=0), nx, axis=1)
        meta['CRSHIFTX'] = (nx, 'Input cosmics image shift in x')
        meta['CRSHIFTY'] = (nx, 'Input cosmics image shift in y')
    else:
        meta['CRFLIPLR'] = (False, 'Input cosmics image NOT flipped Left/Right')
        meta['CRFLIPUD'] = (False, 'Input cosmics image NOT flipped Up/Down')
        meta['CRSHIFTX'] = (0, 'Input cosmics image shift in x')
        meta['CRSHIFTY'] = (0, 'Input cosmics image shift in y')

    del meta['RDNOISE0']
    #- Amp 1 lower left
    nx = pix.shape[1] // 2
    ny = pix.shape[0] // 2
    iixy = np.s_[0:ny, 0:nx]
    cx = pix[iixy][mask[iixy] == 0]
    mean, median, std = sigma_clipped_stats(cx, sigma=3, iters=5)
    meta['RDNOISE1'] = std

    #- Amp 2 lower right
    iixy = np.s_[0:ny, nx:2*nx]
    cx = pix[iixy][mask[iixy] == 0]
    mean, median, std = sigma_clipped_stats(cx, sigma=3, iters=5)
    meta['RDNOISE2'] = std

    #- Amp 3 upper left
    iixy = np.s_[ny:2*ny, 0:nx]
    mean, median, std = sigma_clipped_stats(pix[iixy], sigma=3, iters=5)
    meta['RDNOISE3'] = std

    #- Amp 4 upper right
    iixy = np.s_[ny:2*ny, nx:2*nx]
    mean, median, std = sigma_clipped_stats(pix[iixy], sigma=3, iters=5)
    meta['RDNOISE4'] = std
    fx.close()

    return Image(pix, ivar, mask, meta=meta)

#-------------------------------------------------------------------------
#- desimodel

def get_tile_radec(tileid):
    """
    Return (ra, dec) in degrees for the requested tileid.

    If tileid is not in DESI, return (0.0, 0.0)
    TODO: should it raise an exception instead?
    """
    if not isinstance(tileid, (int, np.int64, np.int32, np.int16)):
        raise ValueError('tileid should be an int, not {}'.format(type(tileid)))

    tiles = desimodel.io.load_tiles()
    if tileid in tiles['TILEID']:
        i = np.where(tiles['TILEID'] == tileid)[0][0]
        return tiles[i]['RA'], tiles[i]['DEC']
    else:
        return (0.0, 0.0)

#-------------------------------------------------------------------------
#- spectral templates

#- Utility function to wrap resample_flux for multiprocessing map
def _resample_flux(args):
    return resample_flux(*args)

def find_basis_template(objtype, indir=None):
    """
    Return the most recent template in $DESI_BASIS_TEMPLATE/{objtype}_template*.fits
    """
    if indir is None:
        indir = os.environ['DESI_BASIS_TEMPLATES']

    objfile_wild = os.path.join(indir, objtype.lower()+'_templates_*.fits')
    objfiles = glob(objfile_wild)
    if len(objfiles) > 0:
        return objfiles[-1]
    else:
        raise IOError('No {} templates found in {}'.format(objtype, objfile_wild))

def _qso_format_version(filename):
    '''Return 1 or 2 depending upon QSO basis template file structure'''
    with fits.open(filename) as fx:
        if fx[1].name == 'METADATA':
            return 1
        elif fx[1].name == 'BOSS_PCA':
            return 2
        else:
            raise IOError('Unknown QSO basis template format '+filename)

def read_basis_templates(objtype, subtype='', outwave=None, nspec=None,
                         infile=None, onlymeta=False, verbose=False):
    """Return the basis (continuum) templates for a given object type.  Optionally
    returns a randomly selected subset of nspec spectra sampled at
    wavelengths outwave.

    Args:

        objtype (str): object type to read (e.g., ELG, LRG, QSO, STAR, FSTD, WD,
          MWS_STAR, BGS).
        subtype (str, optional): template subtype, currently only for white
            dwarfs.  The choices are DA and DB and the default is to read both
            types.
        outwave (numpy.array, optional): array of wavelength at which to sample
            the spectra.
        nspec (int, optional): number of templates to return
        infile (str, optional): full path to input template file to read,
            over-riding the contents of the $DESI_BASIS_TEMPLATES environment
            variable.
        onlymeta (Bool, optional): read just the metadata table and return
        verbose: bool
            Be verbose. (Default: False)

    Returns:
        Tuple of (outflux, outwave, meta) where
        outflux is an Array [ntemplate,npix] of flux values [erg/s/cm2/A];
        outwave is an Array [npix] of wavelengths for FLUX [Angstrom];
        meta is a Meta-data table for each object.  The contents of this
        table varies depending on what OBJTYPE has been read.

    Raises:
        EnvironmentError: If the required $DESI_BASIS_TEMPLATES environment
            variable is not set.
        IOError: If the basis template file is not found.

    """
    from desiutil.log import get_logger, DEBUG
    if verbose:
        log = get_logger(DEBUG)
    else:
        log = get_logger()

    ltype = objtype.lower()
    if objtype == 'FSTD':
        ltype = 'star'
    if objtype == 'MWS_STAR':
        ltype = 'star'

    if infile is None:
        infile = find_basis_template(ltype)

    if onlymeta:
        log.info('Reading {} metadata.'.format(infile))
        meta = Table(fits.getdata(infile, 1))

        if (objtype.upper() == 'WD') and (subtype != ''):
            keep = np.where(meta['WDTYPE'] == subtype.upper())[0]
            if len(keep) == 0:
                log.warning('Unrecognized white dwarf subtype {}!'.format(subtype))
            else:
                meta = meta[keep]

        return meta

    log.info('Reading {}'.format(infile))

    if objtype.upper() == 'QSO':
        with fits.open(infile) as fx:
            format_version = _qso_format_version(infile)
            if format_version == 1:
                flux = fx[0].data * 1E-17
                hdr = fx[0].header
                from desispec.io.util import header2wave
                wave = header2wave(hdr)
                meta = Table(fx[1].data)
            elif format_version == 2:
                flux = fx['SDSS_EIGEN'].data.copy()
                wave = fx['SDSS_EIGEN_WAVE'].data.copy()
                meta = Table([np.arange(flux.shape[0]),], names=['PCAVEC',])
            else:
                raise IOError('Unknown QSO basis template format version {}'.format(format_version))
    else:
        flux, hdr = fits.getdata(infile, 0, header=True)
        meta = Table(fits.getdata(infile, 1))
        wave = fits.getdata(infile, 2)

        if (objtype.upper() == 'WD') and (subtype != ''):
            if 'WDTYPE' not in meta.colnames:
                raise RuntimeError('Please upgrade to basis_templates >=2.3 to get WDTYPE support')

            keep = np.where(meta['WDTYPE'] == subtype.upper())[0]
            if len(keep) == 0:
                log.warning('Unrecognized white dwarf subtype {}!'.format(subtype))
            else:
                meta = meta[keep]
                flux = flux[keep, :]

    # Optionally choose a random subset of spectra. There must be a fast way to
    # do this using fitsio.
    ntemplates = flux.shape[0]
    if nspec is not None:
        these = np.random.choice(np.arange(ntemplates),nspec)
        flux = flux[these,:]
        meta = meta[these]

    # Optionally resample the templates at specific wavelengths.  Use
    # multiprocessing to speed this up.
    if outwave is None:
        outflux = flux # Do I really need to copy these variables!
        outwave = wave
    else:
        args = list()
        for jj in range(nspec):
            args.append((outwave, wave, flux[jj,:]))
        import multiprocessing
        ncpu = multiprocessing.cpu_count() // 2   #- avoid hyperthreading
        pool = multiprocessing.Pool(ncpu)
        outflux = pool.map(_resample_flux, args)
        outflux = np.array(outflux)

    return outflux, outwave, meta

def write_templates(outfile, flux, wave, meta):
    """Write out simulated galaxy templates.

    Args:
        outfile (str): Output file name.
        flux (numpy.ndarray): Flux vector (1e-17 erg/s/cm2/A)
        wave (numpy.ndarray): Wavelength vector (Angstrom).
        meta (astropy.table.Table): metadata table.
    
    """
    from astropy.io import fits
    from desispec.io.util import makepath

    # Create the path to OUTFILE if necessary.
    outfile = makepath(outfile)

    hx = fits.HDUList()
    hdu_wave = fits.PrimaryHDU(wave)
    hdu_wave.header['EXTNAME'] = 'WAVE'
    hdu_wave.header['BUNIT'] = 'Angstrom'
    hdu_wave.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
    hx.append(hdu_wave)    
    
    hdu_flux = fits.ImageHDU(flux)
    hdu_flux.header['EXTNAME'] = 'FLUX'
    hdu_flux.header['BUNIT'] = str(fluxunits)
    hx.append(hdu_flux)
    
    hdu_meta = fits.table_to_hdu(meta)
    hdu_meta.header['EXTNAME'] = 'METADATA'
    hx.append(hdu_meta)

    log.info('Writing {}'.format(outfile))
    try:
        hx.writeto(outfile, overwrite=True)
    except:
        hx.writeto(outfile, clobber=True)

#-------------------------------------------------------------------------
#- Utility functions

def simdir(night='', mkdir=False):
    """
    Return $DESI_SPECTRO_SIM/$PIXPROD/{night}
    If mkdir is True, create directory if needed
    """
    dirname = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'), str(night))
    if mkdir and not os.path.exists(dirname):
        os.makedirs(dirname)

    return dirname

def _parse_filename(filename):
    """
    Parse filename and return (prefix, camera, expid)

    camera=None if the filename isn't camera specific

    e.g. /blat/foo/simspec-00000003.fits -> ('simspec', None, 3)
    e.g. /blat/foo/pix-r2-00000003.fits -> ('pix', 'r2', 3)
    """
    base = os.path.basename(os.path.splitext(filename)[0])
    x = base.split('-')
    if len(x) == 2:
        return x[0], None, int(x[1])
    elif len(x) == 3:
        return x[0], x[1].lower(), int(x[2])

def empty_metatable(nmodel=1, objtype='ELG', subtype='', add_SNeIa=None):
    """Initialize the metadata table for each object type."""
    from astropy.table import Table, Column

    meta = Table()
    meta.add_column(Column(name='OBJTYPE', length=nmodel, dtype='U10'))
    meta.add_column(Column(name='SUBTYPE', length=nmodel, dtype='U10'))
    meta.add_column(Column(name='TEMPLATEID', length=nmodel, dtype='i4',
                           data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='SEED', length=nmodel, dtype='int64',
                           data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='REDSHIFT', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)))
    meta.add_column(Column(name='MAG', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='mag'))
    meta.add_column(Column(name='FLUX_G', length=nmodel, dtype='f4',
                           unit='nanomaggies'))
    meta.add_column(Column(name='FLUX_R', length=nmodel, dtype='f4',
                           unit='nanomaggies'))
    meta.add_column(Column(name='FLUX_Z', length=nmodel, dtype='f4',
                           unit='nanomaggies'))
    meta.add_column(Column(name='FLUX_W1', length=nmodel, dtype='f4',
                           unit='nanomaggies'))
    meta.add_column(Column(name='FLUX_W2', length=nmodel, dtype='f4',
                           unit='nanomaggies'))

    meta.add_column(Column(name='OIIFLUX', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='erg/(s*cm2)'))
    meta.add_column(Column(name='HBETAFLUX', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='erg/(s*cm2)'))
    meta.add_column(Column(name='EWOII', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='Angstrom'))
    meta.add_column(Column(name='EWHBETA', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='Angstrom'))

    meta.add_column(Column(name='D4000', length=nmodel, dtype='f4', data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='VDISP', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='km/s'))
    meta.add_column(Column(name='OIIDOUBLET', length=nmodel, dtype='f4', data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='OIIIHBETA', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='dex'))
    meta.add_column(Column(name='OIIHBETA', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='dex'))
    meta.add_column(Column(name='NIIHBETA', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='dex'))
    meta.add_column(Column(name='SIIHBETA', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='dex'))

    meta.add_column(Column(name='ZMETAL', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='AGE', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='Gyr'))

    meta.add_column(Column(name='TEFF', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='K'))
    meta.add_column(Column(name='LOGG', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1, unit='m/(s**2)'))
    meta.add_column(Column(name='FEH', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1))

    if add_SNeIa:
        meta.add_column(Column(name='SNE_TEMPLATEID', length=nmodel, dtype='i4',
                               data=np.zeros(nmodel)-1))
        meta.add_column(Column(name='SNE_RFLUXRATIO', length=nmodel, dtype='f4',
                               data=np.zeros(nmodel)-1))
        meta.add_column(Column(name='SNE_EPOCH', length=nmodel, dtype='f4',
                               data=np.zeros(nmodel)-1, unit='days'))

    meta['OBJTYPE'] = objtype.upper()
    meta['SUBTYPE'] = subtype.upper()

    return meta

def empty_star_properties(nstar=1):
    """Initialize a "star_properties" table for desisim.templates."""
    from astropy.table import Table, Column

    star_properties = Table()
    star_properties.add_column(Column(name='REDSHIFT', length=nstar, dtype='f4'))
    star_properties.add_column(Column(name='MAG', length=nstar, dtype='f4'))
    star_properties.add_column(Column(name='TEFF', length=nstar, dtype='f4'))
    star_properties.add_column(Column(name='LOGG', length=nstar, dtype='f4'))
    star_properties.add_column(Column(name='FEH', length=nstar, dtype='f4'))
    star_properties.add_column(Column(name='SEED', length=nstar, dtype='int64',
                                      data=np.zeros(nstar)-1))

    return star_properties
