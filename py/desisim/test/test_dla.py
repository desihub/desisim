"""Tests of the DLA absorption (desisim.dla), in particular the higher-order Lyman lines (dlaplus; always on in quickquasars)."""
import unittest
import numpy as np
from desisim.dla import dla_spec, insert_dlas, voigt_tau


class TestDLA(unittest.TestCase):

    def setUp(self):
        self.z_dla = 3.0
        self.wave = np.linspace(3600., 5500., 4000)
        self.dlas = [dict(z=self.z_dla, N=21.0, dlaid=0)]

    def test_default_has_only_lya(self):
        """Without dlaplus the absorption is that of Lya only: the Lyb region is untouched."""
        t = dla_spec(self.wave, self.dlas)
        lyb = 1025.72 * (1. + self.z_dla)
        self.assertTrue(np.all((t >= 0.) & (t <= 1.)))
        self.assertGreater(float(t[np.argmin(np.abs(self.wave - lyb))]), 0.99)

    def test_dlaplus_adds_higher_order_lines(self):
        """With dlaplus the DLA also absorbs at Lyb and never absorbs less than the default."""
        t0 = dla_spec(self.wave, self.dlas)
        t1 = dla_spec(self.wave, self.dlas, dlaplus=True)
        lyb = 1025.72 * (1. + self.z_dla)
        self.assertLess(float(t1[np.argmin(np.abs(self.wave - lyb))]), 0.5)
        self.assertTrue(np.all(t1 <= t0 + 1e-12))

    def test_default_unchanged(self):
        """Without dlaplus the output is bit-identical to the original Lya-only dla_spec."""
        par = [21.0, self.z_dla, 30*1e5, 1215.6700 / 1e8, 0.4164, 626500000.0]
        expected = np.exp(-1.0 * voigt_tau(self.wave / 1e8, par))
        self.assertTrue(np.array_equal(dla_spec(self.wave, self.dlas), expected))

    def test_lyman_continuum_lls(self):
        """For an LLS the Lyman continuum opacity is sigma_0 N (lambda / 911.75 A)^3 below the
        Lyman limit, and the untabulated lines just redward of the limit are a flat
        quasi-continuum sigma_0 N, so there is no transmission spike at the limit."""
        z, logN = 3.5, 17.5
        wave = np.linspace(3600., 5500., 20000)
        t = dla_spec(wave, [dict(z=z, N=logN, dlaid=0)], dlaplus=True)
        with np.errstate(divide='ignore'): # t = 0 in the saturated Lya core
            tau = -np.log(t)
        wave_ll = 911.7536 * (1. + z)
        far = (wave > 800. * (1. + z)) & (wave < 900. * (1. + z))
        expected = 6.304e-18 * 10**logN * (wave[far] / wave_ll)**3
        self.assertTrue(np.allclose(tau[far], expected, rtol=1e-4))
        # between the Lyman limit and the last tabulated line (913.83 A): quasi-continuum
        # sigma_0 N (~2 here), plus the wings of the tabulated lines
        sigma_n = 6.304e-18 * 10**logN
        quasi = (wave > 911.9 * (1. + z)) & (wave < 913.6 * (1. + z))
        self.assertTrue(np.all(tau[quasi] >= sigma_n * (1. - 1e-6)))
        self.assertTrue(np.all(tau[quasi] < 1.1 * sigma_n))
        # no spike: around the Lyman limit, up to the quasi-continuum cutoff (913.73 A), the
        # transmission does not exceed its value just below the limit (blueward of 911 A it
        # rises slowly as lambda^3; redward of the cutoff there are the gaps between lines)
        edge = (wave > 911.0 * (1. + z)) & (wave < 913.7 * (1. + z))
        i_ll = np.argmin(np.abs(wave - 911.7 * (1. + z)))
        self.assertTrue(np.all(t[edge] <= 1.01 * t[i_ll]))

    def test_dlaplus_lya_wing_outside_range(self):
        """The Lya damping wing is included with dlaplus even when the Lya centre is bluer than
        the spectrum, as in the default."""
        dlas = [dict(z=1.9, N=21.0, dlaid=0)] # Lya at 3525 A < 3600 A
        t0 = dla_spec(self.wave, dlas)
        t1 = dla_spec(self.wave, dlas, dlaplus=True)
        self.assertLess(float(t0[0]), 0.999)
        self.assertLessEqual(float(t1[0]), float(t0[0]) + 1e-12)

    def test_insert_dlas_accepts_dlaplus(self):
        rstate = np.random.RandomState(1)
        dlas, t = insert_dlas(self.wave, 3.0, rstate=rstate, dlaplus=True)
        self.assertEqual(len(t), len(self.wave))
        self.assertTrue(np.all((t >= 0.) & (t <= 1.)))


if __name__ == '__main__':
    unittest.main()
