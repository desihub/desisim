"""
I/O routines for desisim
"""

from __future__ import absolute_import, division, print_function

import os
import time
from glob import glob

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np
import multiprocessing

from desispec.interpolation import resample_flux
from desispec.io.util import write_bintable, native_endian, header2wave
import desispec.io
import desimodel.io

from desispec.image import Image
import desispec.io.util

from desispec.log import get_logger
log = get_logger()

from desisim.util import spline_medfilt2d

#-------------------------------------------------------------------------
def findfile(filetype, night, expid, camera=None, outdir=None, mkdir=True):
    """
    Return canonical location of where a file should be on disk

    Args:
        filetype : file type, e.g. 'pix' or 'pixsim'
        night : YEARMMDD string
        expid : exposure id integer
        camera : e.g. 'b0', 'r1', 'z9'

    Optional:
        outdir : output directory; defaults to $DESI_SPECTRO_SIM/$PIXPROD
        mkdir : create output directory if needed; default True

    Returns:
        full file path to output file

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
        pix = '{outdir:s}/pix-{camera:s}-{expid:08d}.fits',
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

def write_simspec(meta, truth, expid, night, header=None, outfile=None):
    """
    Write $DESI_SPECTRO_SIM/$PIXPROD/{night}/simspec-{expid}.fits

    Args:
        meta : metadata table to write to "METADATA" HDU
        truth : dictionary with keys:
            FLUX - 2D array [nspec, nwave] in erg/s/cm2/A
            WAVE - 1D array of vacuum wavelengths [Angstroms]
            SKYFLUX - array of sky flux [erg/s/cm2/A/arcsec],
                      either 1D [nwave] or 2D [nspec, nwave]
            PHOT_{B,R,Z} - 2D array [nspec, nwave] of object photons/bin
            SKYPHOT_{B,R,Z} - 1D or 2D array of sky photons/bin
        expid : integer exposure ID
        night : string YEARMMDD
        header : optional dictionary of header items to add to output
        outfile : optional filename to write (otherwise auto-derived)

    Returns:
        full file path of output file written

    """
    #- Where should this go?
    if outfile is None:
        outdir = simdir(night, mkdir=True)
        outfile = '{}/simspec-{:08d}.fits'.format(outdir, expid)

    #- Primary HDU is just a header from the input
    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU(None, header=desispec.io.util.fitsheader(header)))

    #- Object flux HDU (might not exist, e.g. for an arc)
    if 'FLUX' in truth:
        x = fits.ImageHDU(truth['WAVE'], name='WAVE')
        x.header['BUNIT']  = ('Angstrom', 'Wavelength units')
        x.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
        hx.append(x)

        x = fits.ImageHDU(truth['FLUX'].astype(np.float32), name='FLUX')
        x.header['BUNIT'] = '1e-17 erg/s/cm2/A'
        hx.append(x)

    #- Sky flux HDU
    if 'SKYFLUX' in truth:
        x = fits.ImageHDU(truth['SKYFLUX'].astype(np.float32), name='SKYFLUX')
        x.header['BUNIT'] = '1e-17 erg/s/cm2/A/arcsec2'
        hx.append(x)

    #- Write object photon and sky photons for each channel
    for channel in ['B', 'R', 'Z']:
        x = fits.ImageHDU(truth['WAVE_'+channel], name='WAVE_'+channel)
        x.header['BUNIT']  = ('Angstrom', 'Wavelength units')
        x.header['AIRORVAC']  = ('vac', 'Vacuum wavelengths')
        hx.append(x)

        extname = 'PHOT_'+channel
        x = fits.ImageHDU(truth[extname].astype(np.float32), name=extname)
        x.header['EXTNAME'] = (extname, channel+' channel object photons per bin')
        hx.append(x)

        extname = 'SKYPHOT_'+channel
        if extname in truth:
            x = fits.ImageHDU(truth[extname].astype(np.float32), name=extname)
            x.header['EXTNAME'] = (extname, channel+' channel sky photons per bin')
            hx.append(x)

    #- Write the file
    hx.writeto(outfile, clobber=True)

    #- Add Metadata table HDU; use write_bintable to get units and comments
    if meta is not None:
        comments = dict(
            OBJTYPE     = 'Object type (ELG,LRG,QSO,STD,STAR,MWS_STAR,BGS)',
            REDSHIFT    = 'true object redshift',
            TEMPLATEID  = 'input template ID',
            OIIFLUX     = '[OII] flux [erg/s/cm2]',
            D4000       = '4000-A break'
        )

        units = dict(
            # OBJTYPE     = 'Object type (ELG, LRG, QSO, STD, STAR, MWS_STAR, BGS)',
            # REDSHIFT    = 'true object redshift',
            # TEMPLATEID  = 'input template ID',
            OIIFLUX      = 'erg/s/cm2',
        )

        write_bintable(outfile, meta, header=None, extname="METADATA",
            comments=comments, units=units)

    return outfile

class SimSpec(object):
    """Lightweight wrapper object for simspec data"""
    def __init__(self, flavor, wave, phot, flux=None, skyflux=None,
                 skyphot=None, metadata=None, header=None):
        """
        Args:
            flavor : e.g. 'arc', 'flat', 'dark', 'mws', ...
            wave : dictionary with per-channel wavelength grids, keyed by
                'b', 'r', 'z'.  Optionally also has 'brz' key for channel
                independent wavelength grid
            phot : dictinoary with per-channel photon counts per bin

        Optional:
            flux : channel-independent flux [erg/s/cm^2/A]
            skyflux : channel-indepenent sky flux [erg/s/cm^2/A/arcsec^2]
            skyphot : dictionary with per-channel sky photon counts per bin
            metadata : table of metadata information about these spectra
            header : FITS header from HDU0

        notes:
          * input arguments become attributes
          * wave[channel] is the wavelength grid for phot[channel] and
                skyphot[channel] where channel = 'b', 'r', or 'z'
          * wave['brz'] is the wavelength grid for flux and skyflux
        """
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
        self.header = header

def read_simspec(filename):
    """
    Read simspec data from filename and return SimSpec object
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
    """
    Write simpix data to outfile

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
    
    Options:
        exptime (int): exposure time in seconds
        cosmics_dir: directory to look for cosmics templates; defaults to
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

    Optional:
        shape : (ny, nx) tuple for output image shape
        jitter : If True (default), apply random flips and rolls so you
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

    return Image(pix, ivar, mask, meta=meta)

#-------------------------------------------------------------------------
#- desimodel

def get_tile_radec(tileid):
    """
    Return (ra, dec) in degrees for the requested tileid.

    If tileid is not in DESI, return (0.0, 0.0)
    TODO: should it raise and exception instead?
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

def read_basis_templates(objtype, outwave=None, nspec=None, infile=None):
    """Return the basis (continuum) templates for a given object type.  Optionally
       returns a randomly selected subset of nspec spectra sampled at
       wavelengths outwave.

    Args:
      objtype (str): object type to read (e.g., ELG, LRG, QSO, STAR, FSTD, WD, MWS_STAR, BGS).
      outwave (numpy.array, optional): array of wavelength at which to sample
        the spectra.
      nspec (int, optional): number of templates to return
      infile (str, optional): full path to input template file to read,
        over-riding the contents of the $DESI_BASIS_TEMPLATES environment
        variable.

    Returns:
      outflux (numpy.ndarray): Array [ntemplate,npix] of flux values [erg/s/cm2/A].
      outwave (numpy.ndarray): Array [npix] of wavelengths for FLUX [Angstrom].
      meta (astropy.Table): Meta-data table for each object.  The contents of this
        table varies depending on what OBJTYPE has been read.

    Raises:
      EnvironmentError: If the required $DESI_BASIS_TEMPLATES environment
        variable is not set.
      IOError: If the basis template file is not found.

    """
    from glob import glob
    from astropy.io import fits
    from astropy.table import Table

    ltype = objtype.lower()
    if objtype == 'FSTD':
        ltype = 'star'
    if objtype == 'MWS_STAR':
        ltype = 'star'

    if infile is None:
        infile = find_basis_template(ltype)

    log.info('Reading {}'.format(infile))

    if objtype.upper() == 'QSO':
        fx = fits.open(infile)
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

        fx.close()
    else:
        flux, hdr = fits.getdata(infile, 0, header=True)
        meta = Table(fits.getdata(infile, 1))
        wave = fits.getdata(infile, 2)

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

        ncpu = multiprocessing.cpu_count() // 2   #- avoid hyperthreading
        pool = multiprocessing.Pool(ncpu)
        outflux = pool.map(_resample_flux, args)
        outflux = np.array(outflux)

    return outflux, outwave, meta

def write_templates(outfile, flux, wave, meta, objtype=None,
                    comments=None, units=None):
    """Write out simulated galaxy templates.  (Incomplete documentation...)

        Args:
          outfile (str): Output file name.

        Returns:

        Raises

    """
    from astropy.io import fits
    from desispec.io.util import fitsheader, write_bintable, makepath

    # Create the path to OUTFILE if necessary.
    outfile = makepath(outfile)

    header = dict(
        OBJTYPE = (objtype, 'Object type'),
        CUNIT = ('Angstrom', 'units of wavelength array'),
        CRPIX1 = (1, 'reference pixel number'),
        CRVAL1 = (wave[0], 'Starting wavelength [Angstrom]'),
        CDELT1 = (wave[1]-wave[0], 'Wavelength step [Angstrom]'),
        LOGLAM = (0, 'linear wavelength steps, not log10'),
        AIRORVAC = ('vac', 'wavelengths in vacuum (vac) or air'),
        BUNIT = ('erg/s/cm2/A', 'spectrum flux units')
        )
    hdr = fitsheader(header)

    fits.writeto(outfile,flux.astype(np.float32),header=hdr,clobber=True)
    write_bintable(outfile, meta, header=hdr, comments=comments,
                   units=units, extname='METADATA')


#-------------------------------------------------------------------------
#- Utility functions

def simdir(night='', mkdir=False):
    """
    Return $DESI_SPECTRO_SIM/$PIXPROD/{night}
    If mkdir is True, create directory if needed
    """
    dirname = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'), night)
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

def empty_metatable(nmodel=1, objtype='ELG', add_SNeIa=None):
    """Initialize the metadata table for each object type.""" 
    from astropy.table import Table, Column

    meta = Table()
    meta.add_column(Column(name='OBJTYPE', length=nmodel, dtype='S10'))
    meta.add_column(Column(name='TEMPLATEID', length=nmodel, dtype='i4',
                           data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='SEED', length=nmodel, dtype='int64',
                           data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='REDSHIFT', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)))
    meta.add_column(Column(name='MAG', length=nmodel, dtype='f4',
                           data=np.zeros(nmodel)-1))
    meta.add_column(Column(name='DECAM_FLUX', shape=(6,), length=nmodel, dtype='f4'))
    meta.add_column(Column(name='WISE_FLUX', shape=(2,), length=nmodel, dtype='f4'))

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

    return meta

