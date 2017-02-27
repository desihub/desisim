from __future__ import division

import os
import unittest
import numpy as np
from astropy.table import Table, Column
from desisim.templates import ELG, LRG, QSO, BGS, STAR, FSTD, MWS_STAR, WD

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
        self.seed = np.random.randint(2**32)
        self.rand = np.random.RandomState(self.seed)

    def _check_output_size(self, flux, wave, meta):
        self.assertEqual(len(meta), self.nspec)
        self.assertEqual(len(wave), len(self.wave))
        self.assertEqual(flux.shape, (self.nspec, len(self.wave)))

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_simple(self):
        '''Confirm that creating templates works at all'''
        print('In function test_simple, seed = {}'.format(self.seed))
        for T in [ELG, LRG, QSO, BGS, STAR, FSTD, MWS_STAR, WD]:
            template_factory = T(wave=self.wave)
            flux, wave, meta = template_factory.make_templates(self.nspec, seed=self.seed)
            self._check_output_size(flux, wave, meta)

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_restframe(self):
        '''Confirm restframe template creation for a galaxy and a star'''
        print('In function test_simple, seed = {}'.format(self.seed))
        for T in [ELG, MWS_STAR]:
            template_factory = T(wave=self.wave)
            flux, wave, meta = template_factory.make_templates(self.nspec, seed=self.seed, restframe=True)
            self.assertEqual(len(wave), len(template_factory.basewave))

    def test_input_wave(self):
        '''Confirm that we can specify the wavelength array.'''
        print('In function test_input_wave, seed = {}'.format(self.seed))
        lrg = LRG(minwave=self.wavemin, maxwave=self.wavemax, cdelt=self.dwave)
        flux, wave, meta = lrg.make_templates(self.nspec, seed=self.seed)
        self._check_output_size(flux, wave, meta)
    
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_random_seed(self):
        '''Test that random seed works to get the same results back'''
        print('In function test_input_random_seed, seed = {}'.format(self.seed))
        for T in [ELG, QSO, MWS_STAR]:
            Tx = T(wave=self.wave)
            flux1, wave1, meta1 = Tx.make_templates(self.nspec, seed=1)
            flux2, wave2, meta2 = Tx.make_templates(self.nspec, seed=1)
            flux3, wave3, meta3 = Tx.make_templates(self.nspec, seed=2)
            self.assertTrue(np.all(flux1==flux2))
            self.assertTrue(np.any(flux1!=flux3))
            self.assertTrue(np.all(wave1==wave2))
    
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_OII(self):
        '''Confirm that ELG [OII] flux matches meta table description'''
        print('In function test_OII, seed = {}'.format(self.seed))
        wave = np.arange(5000, 9800.1, 0.2)
        flux, ww, meta = ELG(wave=wave).make_templates(seed=self.seed,
            nmodel=10, zrange=(0.6, 1.6),
            logvdisp_meansig = [np.log10(75), 0.0],
            nocolorcuts=True, nocontinuum=True)
    
        for i in range(len(meta)):
            z = meta['REDSHIFT'][i]
            ii = (3722*(1+z) < wave) & (wave < 3736*(1+z))
            OIIflux = np.sum(flux[i,ii]*np.gradient(wave[ii]))
            self.assertAlmostEqual(OIIflux, meta['OIIFLUX'][i], 2)
    
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_HBETA(self):
        '''Confirm that BGS H-beta flux matches meta table description'''
        print('In function test_HBETA, seed = {}'.format(self.seed))
        wave = np.arange(5000, 7000.1, 0.2)
        # Need to choose just the star-forming galaxies.
        from desisim.io import read_basis_templates
        baseflux, basewave, basemeta = read_basis_templates(objtype='BGS')
        keep = np.where(basemeta['HBETA_LIMIT'] == 0)[0]
        bgs = BGS(wave=wave, basewave=basewave, baseflux=baseflux[keep, :],
                  basemeta=basemeta[keep])
        flux, ww, meta = bgs.make_templates(seed=self.seed,
            nmodel=10, zrange=(0.05, 0.4),
            logvdisp_meansig=[np.log10(75),0.0], 
            nocolorcuts=True, nocontinuum=True)
    
        for i in range(len(meta)):
            z = meta['REDSHIFT'][i]
            ii = (4854*(1+z) < wave) & (wave < 4868*(1+z))
            hbetaflux = np.sum(flux[i,ii]*np.gradient(wave[ii]))
            self.assertAlmostEqual(hbetaflux, meta['HBETAFLUX'][i], 2)
    
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_input_redshift(self):
        '''Test that we can input the redshift for a representative galaxy and star class.'''
        print('In function test_input_redshift, seed = {}'.format(self.seed))
        zrange = np.array([(0.5, 1.0), (0.5, 4.0), (-0.003, 0.003)])
        for zminmax, T in zip(zrange, [LRG, QSO, STAR]):
            redshift = np.random.uniform(zminmax[0], zminmax[1], self.nspec)
            Tx = T(wave=self.wave)
            flux, wave, meta = Tx.make_templates(self.nspec, redshift=redshift, seed=self.seed)
            self.assertTrue(np.allclose(redshift, meta['REDSHIFT']))
    
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_wd_subtype(self):
        '''Test option of specifying the white dwarf subtype.'''
        print('In function test_wd_subtype, seed = {}'.format(self.seed))
        wd = WD(wave=self.wave, subtype='DA')
        flux, wave, meta = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
        import pdb ; pdb.set_trace()
        self._check_output_size(flux, wave, meta)
        np.all(meta['SUBTYPE'] == 'DA')

        wd = WD(wave=self.wave, subtype='DB')
        flux, wave, meta = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
        np.all(meta['SUBTYPE'] == 'DB')
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_input_meta(self):
        '''Test that input meta table option works.'''
        print('In function test_input_meta, seed = {}'.format(self.seed))
        for T in [LRG, QSO, BGS, STAR, WD]:
            Tx = T(wave=self.wave)
            flux1, wave1, meta1 = Tx.make_templates(self.nspec, seed=self.seed)
            flux2, wave2, meta2 = Tx.make_templates(input_meta=meta1)
            badkeys = list()
            for key in meta1.colnames:
                if key in ('DECAM_FLUX', 'WISE_FLUX', 'OIIFLUX', 'HBETAFLUX'):
                    #- not sure why the tolerances aren't closer
                    if not np.allclose(meta1[key], meta2[key], atol=5e-5):
                        print(meta1['OBJTYPE'][0], key, meta1[key], meta2[key])
                        badkeys.append(key)
                else:
                    if not np.all(meta1[key] == meta2[key]):
                        badkeys.append(key)

            self.assertEqual(len(badkeys), 0, 'mismatch for spectral type {} in keys {}'.format(meta1['OBJTYPE'][0], badkeys))
            self.assertTrue(np.allclose(flux1, flux2))
            self.assertTrue(np.all(wave1 == wave2))

    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_star_properties(self):
        '''Test that input data table option works.'''
        print('In function test_star_properties, seed = {}'.format(self.seed))
        star_properties = Table()
        star_properties.add_column(Column(name='REDSHIFT', length=self.nspec, dtype='f4'))
        star_properties.add_column(Column(name='MAG', length=self.nspec, dtype='f4'))
        star_properties.add_column(Column(name='TEFF', length=self.nspec, dtype='f4'))
        star_properties.add_column(Column(name='LOGG', length=self.nspec, dtype='f4'))
        star_properties.add_column(Column(name='FEH', length=self.nspec, dtype='f4'))
        star_properties['REDSHIFT'] = self.rand.uniform(-5E-4, 5E-4, self.nspec)
        star_properties['MAG'] = self.rand.uniform(16, 19, self.nspec)
        star_properties['TEFF'] = self.rand.uniform(4000, 10000, self.nspec)
        star_properties['LOGG'] = self.rand.uniform(0.5, 5.0, self.nspec)
        star_properties['FEH'] = self.rand.uniform(-2.0, 0.0, self.nspec)
        for T in [STAR]:
            Tx = T(wave=self.wave)
            flux, wave, meta = Tx.make_templates(star_properties=star_properties, seed=self.seed)
            #import pdb ; pdb.set_trace()
            badkeys = list()
            for key in meta.colnames:
                if key in star_properties.colnames:
                    if not np.allclose(meta[key], star_properties[key]):
                        badkeys.append(key)
            self.assertEqual(len(badkeys), 0, 'mismatch for spectral type {} in keys {}'.format(meta['OBJTYPE'][0], badkeys))

if __name__ == '__main__':
    unittest.main()
