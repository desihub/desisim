"""Faster versions of the three simqso functions that take most of the time of the quasar templates.

Profile of quickquasars (3 healpixes, 12.6k quasars): of the 45 s spent in simqso, 14 s are in
``render_gaussians`` (a Python loop over the ~50 emission lines of every quasar), 13.5 s in
``BrokenPowerLawContinuumVar._normalize`` (numpy operations on arrays of the wavelength grid, for every
quasar) and 3.3 s in ``_Mtoflam`` (an astropy unit conversion of the same constant for every quasar).

``install()`` replaces them in ``simqso.sqgrids`` by versions that do the same arithmetic in compiled loops
(``numba``) and cache the constant. The results agree with the originals to rounding (the exponential and
the power are computed by the compiled code instead of numpy: a few units in the last place).
"""
import numba
import numpy as np

_INSTALLED = False


@numba.njit(cache=True)
def _add_gaussians(emspec, wave, i1, i2, amp, center, twosig2):
    """emspec[i1[l]:i2[l]] += amp[l] * exp(-(wave - center[l])**2 / twosig2[l]) for every line l."""
    for line in range(amp.size):
        for k in range(i1[line], i2[line]):
            emspec[k] += amp[line] * np.exp(-(wave[k] - center[line]) ** 2 / twosig2[line])


def _render_gaussians(wave, z, lines):
    """Same as ``simqso.sqgrids.render_gaussians``. The scalars of each line are computed with the same
    numpy expressions as the original (so with the same precision: the line parameters and the redshift
    are float32 in desisim), only the loop that adds the Gaussians over the wavelength grid is compiled."""
    emspec = np.zeros_like(wave)
    lines = lines[lines[:, 1] > 0]
    line_wave, eq_width, sigma = lines.T * (1 + z)
    amp = eq_width / (np.sqrt(2 * np.pi) * sigma)
    twosig2 = 2 * sigma ** 2
    nsig = (np.sqrt(-2 * np.log(1e-3 / amp)) * np.array([[-1.], [1]])).T
    ii = np.where(np.logical_and(line_wave > wave[0], line_wave < wave[-1]))[0]
    if ii.size == 0:
        return emspec
    edges = np.searchsorted(wave, line_wave[ii, None] + nsig[ii] * sigma[ii, None])
    _add_gaussians(emspec, np.ascontiguousarray(wave, dtype=np.float64), edges[:, 0], edges[:, 1],
                   np.ascontiguousarray(amp[ii], dtype=np.float64),
                   np.ascontiguousarray(line_wave[ii], dtype=np.float64),
                   np.ascontiguousarray(twosig2[ii], dtype=np.float64))
    return emspec


@numba.njit(cache=True)
def _broken_power_law(wave, bins, f0, breakpts, alpha_lams):
    """``f0[ii] * (wave / breakpts[ii]) ** alpha_lams[ii]`` with ``ii = np.digitize(wave, bins)``."""
    spec = np.empty(wave.size)
    for k in range(wave.size):
        ii = 0
        for b in range(bins.size):
            if bins[b] <= wave[k]:
                ii = b + 1
        spec[k] = f0[ii] * (wave[k] / breakpts[ii]) ** alpha_lams[ii]
    return spec


def install():
    """Replace render_gaussians, BrokenPowerLawContinuumVar._normalize and _Mtoflam in simqso.sqgrids."""
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return
    from simqso import sqgrids  # pylint: disable=import-outside-toplevel
    from astropy import units as u  # pylint: disable=import-outside-toplevel

    render_gaussians = _render_gaussians

    nu0_cache = {}

    def _Mtoflam(lam0, M, z, DM):  # pylint: disable=invalid-name
        if lam0 not in nu0_cache:
            nu0_cache[lam0] = (lam0 * u.Angstrom).to(u.Hz, equivalencies=u.spectral()).value
        nu0 = nu0_cache[lam0]
        fnu0 = 10 ** (-0.4 * (M + DM(z) + 48.599934))
        flam0 = nu0 * fnu0 / lam0
        return flam0 / (1 + z)

    original_normalize = sqgrids.BrokenPowerLawContinuumVar._normalize

    def _normalize(self, wave, z, slopes, fluxNorm=None, minwave=12.4, getspec=True):
        if not getspec or not self.breakPts[0] > minwave:
            # other cases keep the float32 arithmetic of the original
            return original_normalize(self, wave, z, slopes, fluxNorm, minwave, getspec)
        z1 = 1 + z
        alpha_lams = -(2 + np.asarray(slopes))  # a_nu --> a_lam
        alpha_lams = alpha_lams.squeeze()       # comes in as [1,N] from grid
        breakpts = self.breakPts
        if breakpts[0] > minwave:
            breakpts = np.concatenate([[minwave], breakpts]) * z1
        f0 = np.ones_like(breakpts)
        for i, breakwv in enumerate(breakpts[1:], start=1):
            wv1 = breakpts[i - 1]
            f0[i] = f0[i - 1] * (breakwv / wv1) ** alpha_lams[i - 1]
        if fluxNorm is not None:
            normwv = fluxNorm['wavelength']
            fnorm = sqgrids._Mtoflam(normwv, fluxNorm['M_AB'], z, fluxNorm['DM'])
            i = np.searchsorted(self.breakPts, normwv)
            fatnorm = f0[i] * (normwv / self.breakPts[i - 1]) ** alpha_lams[i]
            f0 *= fnorm / fatnorm
        if getspec:
            bins = np.asarray(self.breakPts * z1, dtype=np.float64)
            return _broken_power_law(np.ascontiguousarray(wave, dtype=np.float64), bins,
                                     np.ascontiguousarray(f0, dtype=np.float64),
                                     np.ascontiguousarray(breakpts, dtype=np.float64),
                                     np.ascontiguousarray(alpha_lams, dtype=np.float64))
        return f0, alpha_lams, breakpts

    sqgrids.render_gaussians = render_gaussians
    sqgrids._Mtoflam = _Mtoflam
    sqgrids.BrokenPowerLawContinuumVar._normalize = _normalize
    _INSTALLED = True
