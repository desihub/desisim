"""Tests of the DLA absorption (desisim.dla), in particular the optional higher-order Lyman lines (dlaplus)."""
import unittest
import numpy as np
from desisim.dla import dla_spec, insert_dlas


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

    def test_insert_dlas_accepts_dlaplus(self):
        rstate = np.random.RandomState(1)
        dlas, t = insert_dlas(self.wave, 3.0, rstate=rstate, dlaplus=True)
        self.assertEqual(len(t), len(self.wave))
        self.assertTrue(np.all((t >= 0.) & (t <= 1.)))


if __name__ == '__main__':
    unittest.main()
