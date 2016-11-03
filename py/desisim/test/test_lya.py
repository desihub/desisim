import unittest
from pkg_resources import resource_filename

import numpy as np
try:
    import fitsio
    missing_fitsio = False
except ImportError:
    missing_fitsio = True

from desisim import lya_spectra

class TestLya(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.infile = resource_filename('desisim', 'test/data/simpleLyaSpec.fits.gz')
        if not missing_fitsio:
            fx = fitsio.FITS(cls.infile)
            cls.nspec = len(fx) - 1
            fx.close()
        
    @unittest.skipIf(missing_fitsio, 'fitsio not installed; skipping lya_spectra tests')
    def test_read_lya(self):
        flux, wave, meta = lya_spectra.get_spectra(self.infile)
        self.assertEqual(flux.shape[0], self.nspec)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), self.nspec)

        nqso = 3
        flux, wave, meta = lya_spectra.get_spectra(self.infile, nqso=nqso)
        self.assertEqual(flux.shape[0], nqso)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), nqso)

        flux, wave, meta = lya_spectra.get_spectra(self.infile, nqso=nqso, first=2)
        self.assertEqual(flux.shape[0], nqso)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), nqso)

    @unittest.skipIf(missing_fitsio, 'fitsio not installed; skipping lya_spectra tests')
    def test_read_lya_seed(self):
        flux1a, wave1a, meta1a = lya_spectra.get_spectra(self.infile, nqso=3, seed=1)
        flux1b, wave1b, meta1b = lya_spectra.get_spectra(self.infile, nqso=3, seed=1)
        flux2, wave2, meta2 = lya_spectra.get_spectra(self.infile, nqso=3, seed=2)
        self.assertTrue(np.all(flux1a == flux1b))
        self.assertTrue(np.any(flux1a != flux2))
                
if __name__ == '__main__':
    unittest.main()
