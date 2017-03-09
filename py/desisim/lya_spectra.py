"""
desisim.lya_spectra
===================

Function to simulate a QSO spectrum including Lyman-alpha absorption.
"""

from __future__ import division, print_function
import numpy as np

from pkg_resources import resource_filename

def get_spectra(lyafile, nqso=None, wave=None, templateid=None, normfilter='sdss2010-g',
                seed=None, rand=None, qso=None, add_dlas=False):
    '''Generate a QSO spectrum which includes Lyman-alpha absorption.

    Args:
        lyafile (str): name of the Lyman-alpha spectrum file to read.
        nqso (int, optional): number of spectra to generate (starting from the
          first spectrum; if more flexibility is needed use TEMPLATEID).
        wave (numpy.ndarray, optional): desired output wavelength vector.
        templateid (int numpy.ndarray, optional): indices of the spectra
          (0-indexed) to read from LYAFILE (default is to read everything).  If
          provided together with NQSO, TEMPLATEID wins.
        normfilter (str, optional): normalization filter
        seed (int, optional): Seed for random number generator.
        rand (numpy.RandomState, optional): RandomState object used for the
          random number generation.  If provided together with SEED, this
          optional input superseeds the numpy.RandomState object instantiated by
          SEED.
        qso (desisim.templates.QSO, optional): object with which to generate
          individual spectra/templates.
        add_dlas (bool): Inject damped Lya systems into the Lya forest
          These are done according to the current best estimates for the incidence dN/dz
          These are *not* inserted according to overdensity along the sightline

    Returns:
        flux (numpy.ndarray): Array [nmodel, npix] of observed-frame spectra
          (erg/s/cm2/A).
        wave (numpy.ndarray): Observed-frame [npix] wavelength array (Angstrom).
        meta (astropy.Table): Table of meta-data [nmodel] for each output spectrum
          with columns defined in desisim.io.empty_metatable *plus* RA, DEC.

    '''
    from scipy.interpolate import interp1d

    import fitsio

    from speclite.filters import load_filters
    from desisim.templates import QSO
    from desisim.io import empty_metatable

    h = fitsio.FITS(lyafile)
    if templateid is None:
        if nqso is None:
            nqso = len(h)-1
        templateid = np.arange(nqso)
    else:
        templateid = np.asarray(templateid)
        nqso = len(templateid)

    if rand is None:
        rand = np.random.RandomState(seed)
    templateseed = rand.randint(2**32, size=nqso)

    #heads = [head.read_header() for head in h[templateid + 1]]
    heads = []
    for indx in templateid:
        heads.append(h[indx + 1].read_header())

    zqso = np.array([head['ZQSO'] for head in heads])
    ra = np.array([head['RA'] for head in heads])
    dec = np.array([head['DEC'] for head in heads])
    mag_g = np.array([head['MAG_G'] for head in heads])

    # Hard-coded filtername!
    normfilt = load_filters(normfilter)

    if qso is None:
        qso = QSO(normfilter=normfilter, wave=wave)
        
    wave = qso.wave
    flux = np.zeros([nqso, len(wave)], dtype='f4')

    meta = empty_metatable(objtype='QSO', nmodel=nqso)
    meta['TEMPLATEID'] = templateid
    meta['REDSHIFT'] = zqso
    meta['MAG'] = mag_g
    meta['SEED'] = templateseed
    meta['RA'] = ra
    meta['DEC'] = dec
    
    for ii, indx in enumerate(templateid):
        flux1, _, meta1 = qso.make_templates(nmodel=1, redshift=np.array([zqso[ii]]),
                                             mag=np.array([mag_g[ii]]), seed=templateseed[ii])

        # read lambda and forest transmission
        data = h[indx + 1].read()
        la = data['LAMBDA'][:]
        tr = data['FLUX'][:]

        if len(tr):
            # Interpolate the transmission at the spectral wavelengths, if
            # outside the forest, the transmission is 1.
            itr = interp1d(la, tr, bounds_error=False, fill_value=1.0)
            flux1 *= itr(wave)

        padflux, padwave = normfilt.pad_spectrum(flux1, wave, method='edge')
        normmaggies = np.array(normfilt.get_ab_maggies(padflux, padwave, 
                                                       mask_invalid=True)[normfilter])
        flux1 *= 10**(-0.4 * mag_g[ii]) / normmaggies

        # Inject a DLA?
        if add_dlas:
            flux1 = insert_dlas(wave, flux1, zqso[ii])
        flux[ii, :] = flux1[:]

    h.close()

    return flux, wave, meta


def insert_dlas(wave, flux, zem, rstate=None, seed=None):
    """ Insert zero, one or more DLAs into a given spectrum towards a source
    with a given redshift
    :param wave:
    :param flux:
    :param zem:
    :return:
    """
    #from pyigm.fN import dla as pyi_fd
    #from pyigm.abssys.dla import DLASystem
    #from pyigm.abssys.lls import LLSSystem
    #from pyigm.abssys.utils import hi_model

    # Init
    if rstate is None:
        rstate = np.random.RandomState(seed)
    if fNHI is None:
        fNHI = init_fNHI(slls=slls, mix=mix, high=high)

    # Allowed redshift placement
    ## Cut on zem and 910A rest-frame
    zlya = spec.wavelength.value/1215.67 - 1
    dz = np.roll(zlya,-1)-zlya
    dz[-1] = dz[-2]
    gdz = (zlya < zem) & (spec.wavelength > 910.*u.AA*(1+zem))

    # l(z) -- Uses DLA for SLLS too which is fine
    lz = pyi_fd.lX(zlya[gdz], extrap=True, calc_lz=True)
    cum_lz = np.cumsum(lz*dz[gdz])
    tot_lz = cum_lz[-1]
    fzdla = interpolate.interp1d(cum_lz/tot_lz, zlya[gdz],
                                 bounds_error=False,fill_value=np.min(zlya[gdz]))#

    # n DLA
    nDLA = 0
    while nDLA == 0:
        nval = rstate.poisson(tot_lz, 100)
        gdv = nval > 0
        if np.sum(gdv) == 0:
            continue
        else:
            nDLA = nval[np.where(gdv)[0][0]]

    # Generate DLAs
    dlas = []
    for jj in range(nDLA):
        # Random z
        zabs = float(fzdla(rstate.random_sample()))
        # Random NHI
        NHI = float(fNHI(rstate.random_sample()))
        if (slls or mix):
            dla = LLSSystem((0.,0), zabs, None, NHI=NHI)
        else:
            dla = DLASystem((0.,0), zabs, (None,None), NHI)
        dlas.append(dla)

    # Insert
    vmodel, _ = hi_model(dlas, spec, fwhm=3.)
    # Add noise
    rand = rstate.randn(spec.npix)
    noise = rand * spec.sig * (1-vmodel.flux.value)
    final_spec = XSpectrum1D.from_tuple((vmodel.wavelength,
                                         spec.flux.value*vmodel.flux.value+noise,
                                         spec.sig))
    # Return
    return final_spec, dlas


def init_fNHI(slls=False, mix=False, high=False):
    """ Generate the interpolator for log NHI

    Returns
    -------
    fNHI : scipy.interpolate.interp1d function
    """
    from astropy.io import fits
    from scipy import interpolate as scii
    # f(N)
    fN_file = resource_filename('desisim','data/fN_spline_z24.fits.gz')
    hdu = fits.open(fN_file)
    fN_data = hdu[1].data
    # Instantiate
    pivots=np.array(fN_data['LGN']).flatten()
    param = dict(sply=np.array(fN_data['FN']).flatten())
    fNHI_model = scii.PchipInterpolator(pivots, param['sply'], extrapolate=True)  # scipy 0.16



    # Integrate on NHI
    if slls:
        lX, cum_lX, lX_NHI = calculate_lox(fN_model.zpivot,
                                                    19.5, NHI_max=20.3, cumul=True)
    elif high:
        lX, cum_lX, lX_NHI = calculate_lox(fN_model.zpivot,
                                                    21.2, NHI_max=22.5, cumul=True)
    elif mix:
        lX, cum_lX, lX_NHI = calculate_lox(fN_model.zpivot,
                                                    19.5, NHI_max=22.5, cumul=True)
    else:
        lX, cum_lX, lX_NHI = calculate_lox(fN_model.zpivot,
                                                20.3, NHI_max=22.5, cumul=True)
    # Interpolator
    cum_lX /= cum_lX[-1] # Normalize
    fNHI = interpolate.interp1d(cum_lX, lX_NHI,
                                bounds_error=False,fill_value=lX_NHI[0])
    return fNHI


    # Evaluate
    def evaluate(self, NHI, z, vel_array=None, cosmo=None):
        """ Evaluate the f(N,X) model at a set of NHI values

        Parameters
        ----------
        NHI : array
          log NHI values
        z : float or array
          Redshift for evaluation
        vel_array : ndarray, optional
          Velocities relative to z
        cosmo : astropy.cosmology, optional


        Returns
        -------
        log_fNX : float, array, or 2D array
          f(NHI,X)[z] values
          Float if given one NHI,z value each. Otherwise 2D array
          If 2D, it is [NHI,z] on the axes

        """
        # Tuple?
        if isinstance(NHI, tuple):  # All values packed into NHI parameter
            z = NHI[1]
            NHI = NHI[0]
            flg_1D = 1
        else:  # NHI and z separate
            flg_1D = 0

        # NHI
        if isiterable(NHI):
            NHI = np.array(NHI)  # Insist on array
        else:
            NHI = np.array([NHI])
        lenNHI = len(NHI)

        # Redshift
        if vel_array is not None:
            z_val = z + (1+z) * vel_array/(const.c.to('km/s').value)
        else:
            z_val = z
        if isiterable(z_val):
            z_val = np.array(z_val)
        else:
            z_val = np.array([z_val])
        lenz = len(z_val)

        # Check on zmnx
        bad = np.where((z_val < self.zmnx[0]) | (z_val > self.zmnx[1]))[0]
        if len(bad) > 0:
            raise ValueError(
                'fN.model.eval: z={:g} not within self.zmnx={:g},{:g}'.format(
                    z_val[bad[0]], *(self.zmnx)))

        if self.mtype == 'Hspline':
            # Evaluate without z dependence
            log_fNX = self.model.__call__(NHI)

            # Evaluate
            if (not isiterable(z_val)) | (flg_1D == 1):  # scalar or 1D array wanted
                log_fNX += self.gamma * np.log10((1+z_val)/(1+self.zpivot))
            else:
                # Matrix algebra to speed things up
                lgNHI_grid = np.outer(log_fNX, np.ones(len(z_val)))
                lenfX = len(log_fNX)
                #
                z_grid1 = 10**(np.outer(np.ones(lenfX)*self.gamma,
                                        np.log10(1+z_val)))  #; (1+z)^gamma
                z_grid2 = np.outer( np.ones(lenfX)*((1./(1+self.zpivot))**self.gamma),
                            np.ones(len(z_val)))
                log_fNX = lgNHI_grid + np.log10(z_grid1*z_grid2)

        # Gamma function (e.g. Inoue+14)
        elif self.mtype == 'Gamma':
            # Setup the parameters
            Nl, Nu, Nc, bval = [self.param['common'][key]
                                for key in ['Nl', 'Nu', 'Nc', 'bval']]
            # gNHI
            Bi = self.param['Bi']
            ncomp = len(Bi)
            log_gN = np.zeros((lenNHI, ncomp))
            beta = [self.param[itype]['beta'] for itype in ['LAF', 'DLA']]
            for kk in range(ncomp):
                log_gN[:, kk] += (np.log10(Bi[kk]) + NHI*(-1 * beta[kk])
                                + (-1. * 10.**(NHI-Nc) / np.log(10)))  # log10 [ exp(-NHI/Nc) ]
            # f(z)
            fz = np.zeros((lenz, 2))
            # Loop on NHI
            itypes = ['LAF', 'DLA']
            for kk in range(ncomp):
                if kk == 0:  # LyaF
                    zcuts = self.param['LAF']['zcuts']
                    gamma = self.param['LAF']['gamma']
                else:        # DLA
                    zcuts = self.param['DLA']['zcuts']
                    gamma = self.param['DLA']['gamma']
                zcuts = [0] + zcuts + [999.]
                Aval = self.param[itypes[kk]]['Aval']
                # Cut on z
                for ii in range(1,len(zcuts)):
                    izcut = np.where( (z_val < zcuts[ii]) &
                                      (z_val > zcuts[ii-1]) )[0]
                    liz = len(izcut)
                    # Evaluate (at last!)
                    if (ii <=2) & (liz > 0):
                        fz[izcut, kk] = Aval * ( (1+z_val[izcut]) /
                                                 (1+zcuts[1]) )**gamma[ii-1]
                    elif (ii == 3) & (liz > 0):
                        fz[izcut, kk] = Aval * ( ( (1+zcuts[2]) /
                                                   (1+zcuts[1]) )**gamma[ii-2] *
                                                    ((1+z_val[izcut]) / (1+zcuts[2]) )**gamma[ii-1] )
            # dX/dz
            if cosmo is None:
                cosmo = self.cosmo
            dXdz = pyigmu.cosm_xz(z_val, cosmo=cosmo, flg_return=1)

            # Final steps
            if flg_1D == 1:
                fnX = np.sum(fz * 10.**log_gN, 1) / dXdz
                log_fNX = np.log10(fnX)
            else:
                # Generate the matrix
                fnz = np.zeros((lenNHI, lenz))
                for kk in range(ncomp):
                    fnz += np.outer(10.**log_gN[:, kk], fz[:, kk])
                # Finish up
                log_fNX = np.log10(fnz) - np.log10( np.outer(np.ones(lenNHI), dXdz))
        elif self.mtype == 'PowerLaw':
            log_fNX = self.param['B'] + self.param['beta'] * NHI
            #
            if (not isiterable(z_val)) | (flg_1D == 1):  # scalar or 1D array wanted
                log_fNX += self.gamma * np.log10((1+z_val)/(1+self.zpivot))
            else:
                lgNHI_grid = np.outer(log_fNX, np.ones(len(z_val)))
                lenfX = len(log_fNX)
                #
                z_grid1 = 10**(np.outer(np.ones(lenfX)*self.gamma,
                                        np.log10(1+z_val)))  #; (1+z)^gamma
                z_grid2 = np.outer( np.ones(lenfX)*((1./(1+self.zpivot))**self.gamma),
                                    np.ones(len(z_val)))
                log_fNX = lgNHI_grid + np.log10(z_grid1*z_grid2)
        else:
            raise ValueError('fN.model: Not ready for this model type {:%s}'.format(self.mtype))

        # Return
        if (lenNHI + lenz) == 2:
            return log_fNX.flatten()[0]  # scalar
        else:
            return log_fNX