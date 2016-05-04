from __future__ import division

import os
import unittest
import numpy as np
from desisim.templates import ELG, LRG, QSO, BGS, STAR, FSTD, MWS_STAR

desimodel_data_available = 'DESIMODEL' in os.environ
desi_templates_available = 'DESI_ROOT' in os.environ
desi_basis_templates_available = 'DESI_BASIS_TEMPLATES' in os.environ

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
        for T in [ELG, LRG, QSO, BGS, STAR, FSTD, MWS_STAR]:
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
    def test_HBETA(self):
        '''Confirm that BGS H-beta flux matches meta table description'''
        wave = np.arange(5000, 7000.1, 0.2)
        flux, ww, meta = BGS(wave=wave).make_templates(zrange=[0.1,0.4],
            nmodel=20, nocolorcuts=True, nocontinuum=True,
            logvdisp_meansig=[np.log10(75),0.0])

        for i in range(len(meta)):
            z = meta['REDSHIFT'][i]
            ii = (4854*(1+z) < wave) & (wave < 4868*(1+z))
            hbetaflux = np.sum(flux[i,ii]*np.gradient(wave[ii]))
            self.assertAlmostEqual(hbetaflux, meta['HBETAFLUX'][i], 2)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_stars(self):
        '''Test options specific to star templates'''
        star = STAR(wave=self.wave)
        flux, wave, meta = star.make_templates(self.nspec)
        self._check_output_size(flux, wave, meta)
        self.assertTrue('LOGG' in meta.dtype.names)
        self.assertTrue('TEFF' in meta.dtype.names)
        self.assertTrue('FEH' in meta.dtype.names)

        fstd = FSTD(wave=self.wave)
        flux, wave, meta = fstd.make_templates(self.nspec)
        self._check_output_size(flux, wave, meta)
        self.assertTrue('LOGG' in meta.dtype.names)
        self.assertTrue('TEFF' in meta.dtype.names)
        self.assertTrue('FEH' in meta.dtype.names)

        mwsstar = MWS_STAR(wave=self.wave)
        flux, wave, meta = mwsstar.make_templates(self.nspec)
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
        for T in [ELG, LRG, QSO, BGS, STAR, FSTD, MWS_STAR]:
            Tx = T(wave=self.wave)
            flux1, wave1, meta1 = Tx.make_templates(self.nspec, seed=1)
            flux2, wave2, meta2 = Tx.make_templates(self.nspec, seed=1)
            flux3, wave3, meta3 = Tx.make_templates(self.nspec, seed=2)
            self.assertTrue(np.all(flux1==flux2))
            self.assertTrue(np.any(flux1!=flux3))
            self.assertTrue(np.all(wave1==wave2))
            for col in meta1.dtype.names:
                #- QSO currently has NaN; catch that
                if ((T != QSO) and (T != STAR)) or ((col != 'W1MAG') and (col != 'W2MAG')):
                    self.assertTrue(np.all(meta1[col] == meta2[col]),
                        'metadata {} inconsistent for objtype {}'.format(col, Tx.objtype))

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_sne(self):
        '''Test options for adding in SNeIa spectra'''
        for T in [ELG, LRG, BGS]:
            template_factory = T(wave=self.wave, add_SNeIa=True)
            flux, wave, meta = template_factory.make_templates(self.nspec, sne_rfluxratiorange=(0.5,0.7))
            #import pdb ; pdb.set_trace()
            self._check_output_size(flux, wave, meta)
            self.assertTrue('SNE_TEMPLATEID' in meta.dtype.names)
            self.assertTrue('SNE_RFLUXRATIO' in meta.dtype.names)
            self.assertTrue('SNE_EPOCH' in meta.dtype.names)

    @unittest.expectedFailure
    def test_missing_wise_mags(self):
        '''QSO and WD templates don't have WISE mags.  Flag that'''
        qso = QSO(wave=self.wave)
        flux, wave, meta = qso.make_templates(self.nspec)
        self.assertTrue(not np.any(meta['W1MAG']==99))
        self.assertTrue(not np.any(meta['W2MAG']==99))

        wd = STAR(wave=self.wave, WD=True)
        flux, wave, meta = star.make_templates(self.nspec)
        self.assertTrue(not np.any(meta['W1MAG']==99))
        self.assertTrue(not np.any(meta['W2MAG']==99))

if __name__ == '__main__':
    unittest.main()
