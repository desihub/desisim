import unittest
from importlib import resources

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
        cls.infile = str(resources.files('desisim').joinpath('test', 'data', 'simpleLyaSpec.fits.gz'))
        if not missing_fitsio:
            fx = fitsio.FITS(cls.infile)
            cls.nspec = len(fx) - 1
            fx.close()
        cls.wavemin = 3550
        cls.wavemax = 8000
        cls.dwave = 2.0
        cls.wave = np.arange(cls.wavemin, cls.wavemax+cls.dwave/2, cls.dwave)
        cls.nspec = 5
        cls.templateid = [3, 10, 500]
        cls.seed = 12311423
        #cls.seed = np.random.randint(2**31)
        cls.rand = np.random.RandomState(cls.seed)
            
    @unittest.skipIf(missing_fitsio, 'fitsio not installed; skipping lya_spectra tests')
    def test_read_lya(self):
        flux, wave, meta, objmeta = lya_spectra.get_spectra(self.infile, wave=self.wave, seed=self.seed)
        self.assertEqual(flux.shape[0], self.nspec)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), self.nspec)
        self.assertEqual(len(objmeta), self.nspec)
        templateid = [0,1,2]
        nqso = len(templateid)

        flux, wave, meta, objmeta = lya_spectra.get_spectra(self.infile, templateid=templateid,
                                                   wave=self.wave, seed=self.seed)
        self.assertEqual(flux.shape[0], nqso)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), nqso)
        self.assertEqual(len(objmeta), nqso)

    @unittest.skipIf(missing_fitsio, 'fitsio not installed; skipping lya_spectra tests')
    def test_read_lya_seed(self):
        flux1a, wave1a, meta1a, objmeta1a = lya_spectra.get_spectra(self.infile, wave=self.wave, nqso=3, seed=1)
        flux1b, wave1b, meta1b, objmeta1b = lya_spectra.get_spectra(self.infile, wave=self.wave, nqso=3, seed=1)
        flux2, wave2, meta2, objmeta2 = lya_spectra.get_spectra(self.infile, wave=self.wave, nqso=3, seed=2)
        self.assertTrue(np.all(flux1a == flux1b))
        self.assertTrue(np.any(flux1a != flux2))

    @unittest.skipIf(missing_fitsio, 'fitsio not installed; skipping lya_spectra tests')
    def test_insert_dla(self):
        flux, wave, meta, objmeta, dla_meta = lya_spectra.get_spectra(
            self.infile, wave=self.wave, seed=self.seed, add_dlas=True)
        self.assertEqual(flux.shape[0], self.nspec)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), self.nspec)
        self.assertEqual(len(objmeta), self.nspec)
        self.assertGreater(len(dla_meta), 0)
        self.assertIn('NHI', dla_meta.keys())
        templateid = [0,1,2]
        nqso = len(templateid)

        flux, wave, meta, objmeta = lya_spectra.get_spectra(self.infile, templateid=templateid,
                                                            wave=self.wave, seed=self.seed)
        self.assertEqual(flux.shape[0], nqso)
        self.assertEqual(wave.shape[0], flux.shape[1])
        self.assertEqual(len(meta), nqso)
        self.assertEqual(len(objmeta), nqso)

        #flux, wave, meta = lya_spectra.get_spectra(self.infile, nqso=nqso, first=2)

if __name__ == '__main__':
    unittest.main()
