from __future__ import division

import unittest
import numpy as np
from desisim.templates import ELG, LRG, QSO, STAR

desimodel_data_available = True
try:
    foo = os.environ['DESIMODEL']
except KeyError:
    desimodel_data_available = False

desi_templates_available = True
try:
    foo = os.environ['DESI_ROOT']
except KeyError:
    desi_templates_available = False

desi_basis_templates_available = True
try:
    foo = os.environ['DESI_BASIS_TEMPLATES']
except KeyError:
    desi_basis_templates_available = False

class TestTemplates(unittest.TestCase):

    def setUp(self):
        self.wavemin = 5000
        self.wavemax = 8000
        self.dwave = 2.0
        self.wave = np.arange(self.wavemin, self.wavemax+self.dwave/2, self.dwave)
        self.nspec = 5

    def _check_output_size(self, flux, wave, meta):
        self.assertEqual(len(meta), self.nspec)
        self.assertEqual(len(wave), len(self.wave))
        self.assertEqual(flux.shape, (self.nspec, len(self.wave)))

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_simple(self):
        '''Confirm that creating templates works at all'''
        for T in [ELG, LRG, QSO, STAR]:
            template_factory = T(wave=self.wave)
            flux, wave, meta = template_factory.make_templates(self.nspec)
            self._check_output_size(flux, wave, meta)

        #- Can also specify minwave, maxwave, dwave
        elg = ELG(self.wavemin, self.wavemax, self.dwave)
        flux, wave, meta = elg.make_templates(self.nspec)
        self._check_output_size(flux, wave, meta)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_OII(self):
        '''Confirm that ELG [OII] flux matches meta table description'''
        wave = np.arange(5000, 9800.1, 0.2)
        flux, ww, meta = ELG(wave=wave).make_templates(
            nmodel=20, nocolorcuts=True, nocontinuum=True,
            logvdisp_meansig=[np.log10(75),0.0])

        for i in range(len(meta)):
            z = meta['REDSHIFT'][i]
            ii = (3722*(1+z) < wave) & (wave < 3736*(1+z))
            OIIflux = np.sum(flux[i,ii]*np.gradient(wave[ii]))
            self.assertAlmostEqual(OIIflux, meta['OIIFLUX'][i], 2)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_stars(self):
        '''Test options specific to star templates'''
        star = STAR(wave=self.wave)
        flux, wave, meta = star.make_templates(self.nspec)
        self._check_output_size(flux, wave, meta)
        self.assertTrue('LOGG' in meta.dtype.names)
        self.assertTrue('TEFF' in meta.dtype.names)
        self.assertTrue('FEH' in meta.dtype.names)

        star = STAR(wave=self.wave, FSTD=True)
        flux, wave, meta = star.make_templates(self.nspec)
        self._check_output_size(flux, wave, meta)
        self.assertTrue('LOGG' in meta.dtype.names)
        self.assertTrue('TEFF' in meta.dtype.names)
        self.assertTrue('FEH' in meta.dtype.names)

        star = STAR(wave=self.wave, WD=True)
        flux, wave, meta = star.make_templates(self.nspec)
        self._check_output_size(flux, wave, meta)
        self.assertTrue('LOGG' in meta.dtype.names)
        self.assertTrue('TEFF' in meta.dtype.names)
        self.assertTrue('FEH' not in meta.dtype.names)  #- note: *no* FEH

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_random_seed(self):
        '''Test that random seed works to get the same results back'''
        for T in [ELG, LRG, QSO, STAR]:
            template_factory = T(wave=self.wave)
            flux1, wave1, meta1 = template_factory.make_templates(self.nspec, seed=1)
            flux2, wave2, meta2 = template_factory.make_templates(self.nspec, seed=1)
            flux3, wave3, meta3 = template_factory.make_templates(self.nspec, seed=2)
            self.assertTrue(np.all(flux1==flux2))
            self.assertTrue(np.any(flux1!=flux3))
            self.assertTrue(np.all(wave1==wave2))
            for col in meta1.dtype.names:
                #- QSO currently has NaN; catch that
                if ((T != QSO) and (T != STAR)) or ((col != 'W1MAG') and (col != 'W2MAG')):
                    self.assertTrue(np.all(meta1[col] == meta2[col]))

    @unittest.expectedFailure
    def test_missing_wise_mags(self):
        '''QSO and stellar templates don't have WISE mags.  Flag that'''
        qso = QSO(wave=self.wave)
        flux, wave, meta = qso.make_templates(self.nspec)
        self.assertTrue(not np.any(np.isnan(meta['W1MAG'])))
        self.assertTrue(not np.any(np.isnan(meta['W2MAG'])))

        star = STAR(wave=self.wave)
        flux, wave, meta = star.make_templates(self.nspec)
        self.assertTrue(not np.any(np.isnan(meta['W1MAG'])))
        self.assertTrue(not np.any(np.isnan(meta['W2MAG'])))

if __name__ == '__main__':
    unittest.main()
