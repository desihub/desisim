import numpy as np

try:
    from scipy import constants
    C_LIGHT = constants.c/1000.0
except TypeError: # This can happen during documentation builds.
    C_LIGHT = 299792458.0/1000.0

# code to make mock Lya spectra following McDonald et al. (2006)
# copied from c++ code in Cosmology/LNLyaF
# Modified by Karacayli et al. 2020 functions

def power_amplitude(z):
    """Add redshift evolution to the Gaussian power spectrum."""
    return 58.6*pow((1+z)/4.0,-2.82)

def power_kms(z_c, k_kms, white_noise):
    """Return Gaussian P1D at different wavenumbers k_kms (in s/km), fixed z_c.

    """
    if white_noise: return np.ones_like(k_kms)*100.0
    # power used to make mocks in from Karacayli et al. (2020)
    A = power_amplitude(z_c)
    n     = 0.5
    alpha = 0.26
    gamma = 1.8

    k_0 = 0.001
    k_1 = 0.04

    q0 = k_kms / k_0 + 1e-10

    result = np.power(q0, n - alpha * np.log(q0)) / (1. + np.power(k_kms/k_1, gamma))
    return A*result

def get_tau(z, density):
    """transform lognormal density to optical depth, at each z"""
    # add redshift evolution to mean optical depth
    A = 0.55 * np.power((1. + z) / 4., 5.1)
    return A*density

class MockMaker(object):
    """Class to generate 1D mock Lyman alpha skewers."""

    # central redshift, sets center of skewer and pivot point in z-evolution
    z_c=3.0

    def __init__(self, N2=17, dv_kms=4.0, seed=666, white_noise=False):
        """Construct object to make 1D Lyman alpha mocks.
        
          Optional arguments:
            N2: the number of cells in the skewer will be 2^N2
            dv_kms: cell width (in km/s)
            seed: starting seed for the random number generator
            white_noise: use constant power instead of realistic P1D. """
        self.n_cells = np.power(2, N2)
        self.dv_kms = dv_kms
        # setup random number generator using seed
        self.gen = np.random.default_rng(seed)
        self.white_noise = white_noise

    def get_density(self, var_delta, z, delta):
        """Transform Gaussian field delta to lognormal density, at each z."""
        tau_pl=2.0
        # relative amplitude
        rel_amp = power_amplitude(z)/power_amplitude(self.z_c)
        return np.exp(tau_pl*(delta*np.sqrt(rel_amp)-0.5*var_delta*rel_amp))

    def get_redshifts(self):
        """Get redshifts for each cell in the array (centered at z_c).
            Uses logarithmic transformation
        """
        velocity_values  = self.dv_kms * (np.arange(self.n_cells) - self.n_cells/2)

        return (1+self.z_c)*np.exp(velocity_values / C_LIGHT) - 1

    def get_gaussian_fields(self, n_spectra=1, new_seed=None):
        """Generate Ns Gaussian fields at redshift z_c.

          If new_seed is set, it will reset random generator with it."""
        if new_seed:
            self.gen = np.random.default_rng(new_seed)

        # get frequencies (wavenumbers in units of s/km)
        k_kms = 2*np.pi*np.fft.rfftfreq(self.n_cells, d=self.dv_kms)
        # get power evaluated at each k
        P_kms = power_kms(self.z_c, k_kms, self.white_noise)

        # compute also expected variance, will be used in lognormal transform
        var_delta=np.sum(P_kms) * k_kms[1] / np.pi

        # generate random Fourier modes
        delta = self.gen.standard_normal((n_spectra, self.n_cells))
        delta_k  = np.fft.rfft(delta, axis=1) * self.dv_kms
        delta_k *= np.sqrt( P_kms / self.dv_kms )
        delta = np.fft.irfft(delta_k, axis=1) / self.dv_kms

        return delta, var_delta

    def get_lya_skewers(self, n_spectra=10, new_seed=None):
        """Return Ns Lyman alpha skewers (wavelength, flux). 
        
          If new_seed is set, it will reset random generator with it."""
        if new_seed:
            self.gen = np.random.default_rng(new_seed)
        # get redshift for all cells in the skewer    
        z = self.get_redshifts()
        delta, var_delta = self.get_gaussian_fields(n_spectra)
        #var_delta = np.var(delta)
        density = self.get_density(var_delta, z, delta)
        tau = get_tau(z,density)
        flux = np.exp(-tau)
        wave = 1215.67*(1+z)
        return wave, flux
