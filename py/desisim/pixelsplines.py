"""
desisim.pixelsplines
====================

Pixel-integrated spline utilities.

Written by A. Bolton, U. of Utah, 2010-2013.
"""

import numpy as n
from scipy import linalg as la
from scipy import sparse as sp
from scipy import special as sf

def compute_duck_slopes(pixbound, flux):
    """
        Compute the slope of the illuminating quadratic spline at
        the locations of the 'ducks', i.e., the pixel boundaries,
        given the integrated flux per unit baseline within the pixels.

        ARGUMENTS:
        pixbound: (npix + 1) ndarray of pixel boundaries, in units of
        wavelength or log-wavelength or frequency or whatever you like.
        flux: (npix) ndarray of spectral flux (energy or counts) per
        abscissa unit, averaged over the extent of the pixel

        RETURNS:
        an (npix+1) ndarray of the slope of the underlying/illuminating
        flux per unit abscissa spectrum at the position of the pixel
        boundaries, a.k.a. 'ducks'.  The end conditions are taken to
        be zero slope, so the exterior points of the output are zeros.
        """
    npix = len(flux)
    # Test for correct argument dimensions:
    if (len(pixbound) - npix) != 1:
        print('Need one more element in pixbound than in flux!')
        return 0
    # The array of "delta-x" values:
    dxpix = pixbound[1:] - pixbound[:-1]
    # Test for monotonif increase:
    if dxpix.min() <= 0.:
        print('Pixel boundaries not monotonically increasing!')
        return 0
    # Encode the tridiagonal matrix that needs to be solved:
    maindiag = (dxpix[:-1] + dxpix[1:]) / 3.
    offdiag = dxpix[1:-1] / 6.
    upperdiag = n.append(0., offdiag)
    lowerdiag = n.append(offdiag, 0.)
    band_matrix = n.vstack((upperdiag, maindiag, lowerdiag))
    # The right-hand side:
    rhs = flux[1:] - flux[:-1]
    # Solve the banded matrix and return:
    acoeff = la.solve_banded((1,1), band_matrix, rhs)
    acoeff = n.append(n.append(0., acoeff), 0.)
    return acoeff

def cen2bound(pixelcen):
    """
        Convenience function to do the obvious thing to transform
        pixel centers to pixel boundaries.
        """
    pixbound = 0.5 * (pixelcen[1:] + pixelcen[:-1])
    lo_val = 2. * pixbound[0] - pixbound[1]
    hi_val = 2. * pixbound[-1] - pixbound[-2]
    pixbound = n.append(n.append(lo_val, pixbound), hi_val)
    return pixbound

def gauss_blur_matrix(pixbound, sig_conv):
    """
        Function to generate a Gaussian blurring matrix for a pixelized
        spectrum, from specified pixel boundaries and 'sigma' vector.
        The matrix will be flux-conserving if the spectrum to which it is
        applied has units of 'counts per unit x', and pixbound and sig_conv
        both have units of x.

        pixbound should have one more element than sig_conv.

        Output is a scipy sparse matrix that can implement the blurring as:
        blurflux = gauss_blur_matrix * flux
        where 'flux' has the same dimensions as 'sig_conv'.
        """
    # Derived values and error checks:
    npix = len(pixbound) - 1
    if (len(sig_conv) != npix):
        raise PixSplineError('Need one more element in pixbound than in \
                             sig_conv!')
    if (sig_conv.min() <= 0.):
        raise PixSplineError('sig_conv must be > 0 everywhere!')
    xcen = 0.5 * (pixbound[1:] + pixbound[:-1])
    dxpix = pixbound[1:] - pixbound[:-1]
    if (dxpix.min() <= 0.):
        raise PixSplineError('Pixel boundaries not monotonically increasing!')
    # Which "new" pixels does each "old" pixel touch?
    # Let's go +/- 6 sigma for all:
    sig_width = 6.0
    # A minor correction factor to preserve flux conservation:
    cfact = 1./sf.erf(sig_width / n.sqrt(2.))
    xblur_lo = xcen - sig_width * sig_conv
    xblur_hi = xcen + sig_width * sig_conv
    bin_lo = n.digitize(xblur_lo, pixbound) - 1
    bin_hi = n.digitize(xblur_hi, pixbound) - 1
    # Restrict the ranges:
    #xblur_lo = n.where((xblur_lo > pixbound[0]), xblur_lo, pixbound[0])
    #xblur_lo = n.where((xblur_lo < pixbound[-1]), xblur_lo, pixbound[-1])
    #xblur_hi = n.where((xblur_hi > pixbound[0]), xblur_hi, pixbound[0])
    #xblur_hi = n.where((xblur_hi < pixbound[-1]), xblur_hi, pixbound[-1])
    bin_lo = n.where((bin_lo >= 0), bin_lo, 0)
    #bin_lo = n.where((bin_lo < npix), bin_lo, npix-1)
    #bin_hi = n.where((bin_hi >= 0), bin_hi, 0)
    bin_hi = n.where((bin_hi < npix), bin_hi, npix-1)
    # Compute total number of non-zero elements in the broadening matrix:
    n_each = bin_hi - bin_lo + 1
    n_entries = n_each.sum()
    ij = n.zeros((2, n_entries), dtype=int)
    v_vec = n.zeros(n_entries, dtype=float)
    # Loop over pixels in the "old" spectrum:
    pcount = 0
    roottwo = n.sqrt(2.)
    bin_vec = n.arange(npix, dtype=int)
    for k in range(npix):
        xbound = pixbound[bin_lo[k]:bin_hi[k]+2]
        # Gaussian integral in terms of error function:
        erf_terms = cfact * 0.5 * sf.erf((xbound - xcen[k]) / (roottwo *
                                                               sig_conv[k]))
        erf_int = (erf_terms[1:] - erf_terms[:-1]) * \
                dxpix[k] / dxpix[bin_lo[k]:bin_hi[k]+1]
        ij[0,pcount:pcount+n_each[k]] = bin_vec[bin_lo[k]:bin_hi[k]+1]
        ij[1,pcount:pcount+n_each[k]] = k
        v_vec[pcount:pcount+n_each[k]] = erf_int
        pcount += n_each[k]
    conv_matrix = sp.coo_matrix((v_vec, ij), shape=(npix,npix))
    return conv_matrix.tocsr()

class PixSplineError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class PixelSpline:
    """
        Pixel Spline object class.

        Initialize as follows:
        PS = PixelSpline(pixbound, flux)
        where
        pixbound = array of pixel boundaries in baseline units
        and
        flux = array of specific flux values in baseline units.

        Assumptions:
        'pixbound' should have one more element than 'flux', and
        units of 'flux' are -per-unit-baseline, for the baseline
        units in which pixbound is expressed, averaged over the
        extent of each pixel.
        """
    def __init__(self, pixbound, flux):
        npix = len(flux)
        # Test for correct argument dimensions:
        if (len(pixbound) - npix) != 1:
            raise PixSplineError('Need one more element in pixbound \
                                 than in flux!')
        # The array of "delta-x" values:
        dxpix = pixbound[1:] - pixbound[:-1]
        # Test for monotonic increase:
        if dxpix.min() <= 0.:
            raise PixSplineError('Pixel boundaries not monotonically \
                                 increasing!')
        self.npix = npix
        self.pixbound = pixbound.copy()
        self.dxpix = dxpix.copy()
        self.xcen = 0.5 * (pixbound[1:] + pixbound[:-1]).copy()
        self.flux = flux.copy()
        maindiag = (dxpix[:-1] + dxpix[1:]) / 3.
        offdiag = dxpix[1:-1] / 6.
        upperdiag = n.append(0., offdiag)
        lowerdiag = n.append(offdiag, 0.)
        band_matrix = n.vstack((upperdiag, maindiag, lowerdiag))
        # The right-hand side:
        rhs = flux[1:] - flux[:-1]
        # Solve the banded matrix for the slopes at the ducks:
        acoeff = la.solve_banded((1,1), band_matrix, rhs)
        self.duckslopes = n.append(n.append(0., acoeff), 0.)
    def point_evaluate(self, xnew, missing=0.):
        """
            Evaluate underlying pixel spline at array of points
            BUG: input currently needs to be at least 1D array.
            """
        # Initialize output array:
        outflux = 0. * self.flux[0] * xnew + missing
        # Digitize into bins:
        bin_idx = n.digitize(xnew, self.pixbound)
        # Find the indices of those that are actually in-bounds:
        wh_in = n.where((bin_idx > 0) * (bin_idx < len(self.pixbound)))
        if len(wh_in[0]) == 0:
            return outflux
        xnew_in = xnew[wh_in]
        idx_in = bin_idx[wh_in] - 1
        # The pixel centers as per the algorithm in use:
        adiff = self.duckslopes[idx_in+1] - self.duckslopes[idx_in]
        asum = self.duckslopes[idx_in+1] + self.duckslopes[idx_in]
        xdiff = xnew_in - self.xcen[idx_in]
        fluxvals = adiff * xdiff**2 / (2. * self.dxpix[idx_in]) + asum * xdiff \
                / 2. + self.flux[idx_in] - adiff * self.dxpix[idx_in] / 24.
        outflux[wh_in] = fluxvals
        return outflux
    def find_extrema(self, minima=False):
        # Find the formal extrema positions:
        x_ext = self.xcen - 0.5 * self.dxpix * \
                (self.duckslopes[1:] + self.duckslopes[:-1]) / \
                (self.duckslopes[1:] - self.duckslopes[:-1])
        # Digitize these into bins:
        bin_ext = n.digitize(x_ext, self.pixbound) - 1
        # The second derivatives, flipped in sign if minima is set:
        curvat = (-1)**(minima == True) * (self.duckslopes[1:] -
                                           self.duckslopes[:-1]) / self.dxpix
        # Find in-bin maxima:
        wh_ext = n.where((bin_ext == n.arange(self.npix)) * (curvat < 0))
        if len(wh_ext[0]) < 1:
            return n.array([])
        x_ext = x_ext[wh_ext]
        return x_ext
    def subpixel_average(self, ipix, xlo, xhi):
        adiff = self.duckslopes[ipix+1] - self.duckslopes[ipix]
        asum = self.duckslopes[ipix+1] + self.duckslopes[ipix]
        xlo_c = xlo - self.xcen[ipix]
        xhi_c = xhi - self.xcen[ipix]
        outval = adiff * ((xhi-xlo)**2 / 6. + xhi_c * xlo_c / 2.) / \
                self.dxpix[ipix] + asum * (xhi_c + xlo_c) / 4. - adiff * \
                self.dxpix[ipix] / 24. + self.flux[ipix]
        return outval
    def resample(self, pb_new):
        """
            Method to resample a pixelspline analytically onto a new
            set of pixel boundaries.
            """
        npix_new = len(pb_new) - 1
        xnew_lo = pb_new[:-1].copy()
        xnew_hi = pb_new[1:].copy()
        # Test for monotonic:
        new_fulldx = xnew_hi - xnew_lo
        if new_fulldx.min() <= 0.:
            raise PixSplineError('New pixel boundaries not monotonically \
                                 increasing!')
        # Digitize the new boundaries into the original bins:
        bin_idx = n.digitize(pb_new, self.pixbound) - 1
        bin_lo = bin_idx[:-1].copy()
        bin_hi = bin_idx[1:].copy()
        # Array for accumulating new counts:
        new_counts = n.zeros(npix_new, dtype=self.flux.dtype)
        # Array for accumulating new pixel widths by pieces.
        # Only used for debugging so far, but may be useful in future.
        #new_dxpix = n.zeros(npix_new, dtype=self.flux.dtype)
        # For convenience, we define the following.
        # Careful not to modify them... they are views, not copies!
        xold_lo = self.pixbound[:-1]
        xold_hi = self.pixbound[1:]
        # 4 cases to cover:
        # Case 1: both bin_hi and bin_lo in the same bin:
        wh_this = n.where((bin_hi == bin_lo) * (bin_lo >= 0) * \
                          (bin_hi < self.npix))
        if (len(wh_this[0]) > 0):
            dx_this = xnew_hi[wh_this] - xnew_lo[wh_this]
            avgval_this = self.subpixel_average(bin_lo[wh_this],
                                                xnew_lo[wh_this],
                                                xnew_hi[wh_this])
            #new_dxpix[wh_this] += dx_this
            new_counts[wh_this] += avgval_this * dx_this
        # Case 2: more than one bin, lower segment:
        wh_this = n.where((bin_hi > bin_lo) * (bin_lo >= 0))
        if (len(wh_this[0]) > 0):
            dx_this = xold_hi[bin_lo[wh_this]] - xnew_lo[wh_this]
            avgval_this = self.subpixel_average(bin_lo[wh_this],
                                                xnew_lo[wh_this],
                                                xold_hi[bin_lo[wh_this]])
            #new_dxpix[wh_this] += dx_this
            new_counts[wh_this] += avgval_this * dx_this
        # Case 3: more than one bin, upper segment:
        wh_this = n.where((bin_hi > bin_lo) * (bin_hi < self.npix))
        if (len(wh_this[0]) > 0):
            dx_this = xnew_hi[wh_this] - xold_lo[bin_hi[wh_this]]
            avgval_this = self.subpixel_average(bin_hi[wh_this],
                                                xold_lo[bin_hi[wh_this]],
                                                xnew_hi[wh_this])
            #new_dxpix[wh_this] += dx_this
            new_counts[wh_this] += avgval_this * dx_this
        # Case 4: enire bins covered, whole pixels:
        wh_this = n.where(bin_hi > (bin_lo+1))
        nwhole = len(wh_this[0])
        if (nwhole > 0):
            pcounts = self.flux * self.dxpix
            icounts_this = n.array([pcounts[bin_lo[wh_this[0][ii]]+1:\
                                            bin_hi[wh_this[0][ii]]].sum()
                                    for ii in range(nwhole)])
            #new_dxpix[wh_this] += dx_this
            new_counts[wh_this] += icounts_this
        # Divide out for average and return:
        return new_counts / new_fulldx

class WeightedRebinCoadder:
    """
        Objet class for weighted rebinning and coaddition of spectra

        Initialize as follows:
        WRC = WeighedRebinCoadder(fluxes, invvars, pixbounds)
        where
        fluxes = list of arrays of specific flux values
        invvars = list of arrays of associated inverse variances
        pixbounds = list of arrays of pixel boundaries in baseline units
        """
    def __init__(self, fluxes, invvars, pixbounds):
        # Determine minimum and maximum values of independent variable:
        self.min_indep = [this_bound.min() for this_bound in pixbounds]
        self.max_indep = [this_bound.max() for this_bound in pixbounds]
        self._n_input = len(fluxes)
        # Compute pixel widths:
        dpixes = [this_bound[1:] - this_bound[:-1] for this_bound in pixbounds]
        # Compute "specific inverse variances":
        sp_invvars = [invvars[i] / dpixes[i] for i in range(self._n_input)]
        # Compute pixelspline objects for fluxes:
        self._PXS_fluxes = [PixelSpline(pixbounds[i], fluxes[i]) for i in \
                            range(self._n_input)]
        # Compute pixelspline objects for specific inverse variances:
        self._PXS_sp_invvars = [PixelSpline(pixbounds[i], sp_invvars[i]) for \
                                i in range(self._n_input)]
    def coadd(self, pixbound_out):
        # Compute coverage masks:
        masks = [(pixbound_out[:-1] > self.min_indep[i]) *
                 (pixbound_out[1:] < self.max_indep[i]) for i in \
                 range(self._n_input)]
        # Compute output pixel widths:
        dpix_out = pixbound_out[1:] - pixbound_out[:-1]
        # Compute interpolated fluxes:
        new_fluxes = [this_PXS.resample(pixbound_out) for this_PXS in \
                      self._PXS_fluxes]
        # Compute interpolated specific inverse variances (converted
        # to inverse variances):
        new_invvars = [dpix_out * this_PXS.resample(pixbound_out) for \
                       this_PXS in self._PXS_sp_invvars]
        # Compute coadded flux and inverse variance and return:
        flux_coadd = 0.
        invvar_coadd = 0.
        for i in range(self._n_input):
            flux_coadd += new_fluxes[i] * new_invvars[i] * masks[i]
            invvar_coadd += new_invvars[i] * masks[i]
        is_good = n.where(invvar_coadd > 0.)
        flux_coadd[is_good] /= invvar_coadd[is_good]
        return flux_coadd, invvar_coadd
