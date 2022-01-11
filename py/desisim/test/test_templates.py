from __future__ import division

import os
import unittest
import numpy as np
from astropy.table import Table, Column
from desisim.templates import ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO
from desisim import lya_mock_p1d as lyamock

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

    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_simple_south(self):
    #    '''Confirm that creating templates works at all'''
    #    #print('In function test_simple_south, seed = {}'.format(self.seed))
    #    for T in [ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO]:
    #        template_factory = T(wave=self.wave)
    #        flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=self.seed, south=True)
    #        self._check_output_size(flux, wave, meta)
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_simple_north(self):
    #    '''Confirm that creating templates works at all'''
    #    #print('In function test_simple_north, seed = {}'.format(self.seed))
    #    for T in [ELG, LRG, QSO, BGS, STAR, STD, MWS_STAR, WD, SIMQSO]:
    #        template_factory = T(wave=self.wave)
    #        flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=self.seed, south=False)
    #        self._check_output_size(flux, wave, meta)
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_restframe(self):
    #    '''Confirm restframe template creation for a galaxy and a star'''
    #    #print('In function test_simple, seed = {}'.format(self.seed))
    #    for T in [ELG, MWS_STAR]:
    #        template_factory = T(wave=self.wave)
    #        flux, wave, meta, _ = template_factory.make_templates(self.nspec, seed=self.seed, restframe=True)
    #        self.assertEqual(len(wave), len(template_factory.basewave))
    #
    #def test_input_wave(self):
    #    '''Confirm that we can specify the wavelength array.'''
    #    #print('In function test_input_wave, seed = {}'.format(self.seed))
    #    lrg = LRG(minwave=self.wavemin, maxwave=self.wavemax, cdelt=self.dwave)
    #    flux, wave, meta, _ = lrg.make_templates(self.nspec, seed=self.seed)
    #    self._check_output_size(flux, wave, meta)
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_random_seed(self):
    #    '''Test that random seed works to get the same results back'''
    #    #print('In function test_input_random_seed, seed = {}'.format(self.seed))
    #    for T in [ELG, QSO, MWS_STAR, SIMQSO]:
    #        Tx = T(wave=self.wave)
    #        flux1, wave1, meta1, _ = Tx.make_templates(self.nspec, seed=1)
    #        flux2, wave2, meta2, _ = Tx.make_templates(self.nspec, seed=1)
    #        flux3, wave3, meta3, _ = Tx.make_templates(self.nspec, seed=2)
    #        self.assertTrue(np.all(flux1==flux2))
    #        self.assertTrue(np.any(flux1!=flux3))
    #        self.assertTrue(np.all(wave1==wave2))
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_OII(self):
    #    '''Confirm that ELG [OII] flux matches meta table description'''
    #    #print('In function test_OII, seed = {}'.format(self.seed))
    #    wave = np.arange(5000, 9800.1, 0.2)
    #    flux, ww, meta, objmeta = ELG(wave=wave).make_templates(
    #        seed=self.seed, nmodel=10, zrange=(0.6, 1.6), 
    #        vdisprange=(75.0, 75.0),
    #        nocolorcuts=True, nocontinuum=True)
    #
    #    for i in range(len(meta)):
    #        z = meta['REDSHIFT'][i]
    #        ii = (3722*(1+z) < wave) & (wave < 3736*(1+z))
    #        OIIflux = 1e-17 * np.sum(flux[i,ii] * np.gradient(wave[ii]))
    #        self.assertAlmostEqual(OIIflux, objmeta['OIIFLUX'][i], 2)
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_HBETA(self):
    #    '''Confirm that BGS H-beta flux matches meta table description'''
    #    #print('In function test_HBETA, seed = {}'.format(self.seed))
    #    wave = np.arange(5000, 7000.1, 0.2)
    #    # Need to choose just the star-forming galaxies.
    #    from desisim.io import read_basis_templates
    #    baseflux, basewave, basemeta = read_basis_templates(objtype='BGS')
    #    keep = np.where(basemeta['HBETA_LIMIT'] == 0)[0]
    #    bgs = BGS(wave=wave, basewave=basewave, baseflux=baseflux[keep, :],
    #              basemeta=basemeta[keep])
    #    flux, ww, meta, objmeta = bgs.make_templates(seed=self.seed,
    #        nmodel=10, zrange=(0.05, 0.4),
    #        vdisprange=(75.0, 75.0),
    #        nocolorcuts=True, nocontinuum=True)
    #
    #    for i in range(len(meta)):
    #        z = meta['REDSHIFT'][i]
    #        ii = (4854*(1+z) < wave) & (wave < 4868*(1+z))
    #        hbetaflux = 1e-17 * np.sum(flux[i,ii] * np.gradient(wave[ii]))
    #        self.assertAlmostEqual(hbetaflux, objmeta['HBETAFLUX'][i], 2)
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_input_redshift(self):
    #    '''Test that we can input the redshift for a representative galaxy and star class.'''
    #    #print('In function test_input_redshift, seed = {}'.format(self.seed))
    #    zrange = np.array([(0.5, 1.0), (0.5, 4.0), (-0.003, 0.003)])
    #    for zminmax, T in zip(zrange, [LRG, QSO, STAR, SIMQSO]):
    #        redshift = np.random.uniform(zminmax[0], zminmax[1], self.nspec)
    #        Tx = T(wave=self.wave)
    #        flux, wave, meta, _ = Tx.make_templates(self.nspec, redshift=redshift, seed=self.seed)
    #        self.assertTrue(np.allclose(redshift, meta['REDSHIFT']))
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_wd_subtype(self):
    #    '''Test option of specifying the white dwarf subtype.'''
    #    #print('In function test_wd_subtype, seed = {}'.format(self.seed))
    #    wd = WD(wave=self.wave, subtype='DA')
    #    flux, wave, meta, _ = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
    #    self._check_output_size(flux, wave, meta)
    #    np.all(meta['SUBTYPE'] == 'DA')
    #
    #    wd = WD(wave=self.wave, subtype='DB')
    #    flux, wave, meta, _ = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
    #    np.all(meta['SUBTYPE'] == 'DB')
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #@unittest.expectedFailure
    #def test_wd_subtype_failure(self):
    #    '''Test a known failure of specifying the white dwarf subtype.'''
    #    #print('In function test_wd_subtype_failure, seed = {}'.format(self.seed))
    #    wd = WD(wave=self.wave, subtype='DA')
    #    flux1, wave1, meta1, _ = wd.make_templates(self.nspec, seed=self.seed, nocolorcuts=True)
    #    meta1['SUBTYPE'][0] = 'DB'
    #    flux2, wave2, meta2, _ = wd.make_templates(input_meta=meta1)
        
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    def test_input_meta(self):
        '''Test that input meta table option works.'''
        #print('In function test_input_meta, seed = {}'.format(self.seed))
        for T in [ELG, LRG, BGS, QSO, STAR, MWS_STAR, WD]:
            print('Working on {} templates'.format(T.__name__))
            Tx = T(wave=self.wave)
            flux1, wave1, meta1, objmeta1 = Tx.make_templates(self.nspec, seed=self.seed)
            flux2, wave2, meta2, objmeta2 = Tx.make_templates(input_meta=meta1, input_objmeta=objmeta1)

            badkeys = list()
            for key in meta1.colnames:
                if key in ('REDSHIFT', 'MAG', 'SEED', 'FLUX_G',
                           'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'):
                    #- not sure why the tolerances aren't closer
                    if not np.allclose(meta1[key], meta2[key], rtol=1e-4):
                        #print(meta1['OBJTYPE'][0], key, meta1[key], meta2[key])
                        badkeys.append(key)
                        print(key, meta1[key][0], meta2[key][0])
                else:
                    if not np.all(meta1[key] == meta2[key]):
                        badkeys.append(key)

            self.assertEqual(len(badkeys), 0, 'mismatch for spectral type {} in keys {}'.format(meta1['OBJTYPE'][0], badkeys))
            #self.assertTrue(np.all(np.isclose(flux1, flux2, atol=1e-3)))
            self.assertTrue(np.allclose(flux1, flux2, rtol=1e-4))
            self.assertTrue(np.all(wave1 == wave2))

    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_star_properties(self):
    #    '''Test that input data table option works.'''
    #    #print('In function test_star_properties, seed = {}'.format(self.seed))
    #    star_properties = Table()
    #    star_properties.add_column(Column(name='REDSHIFT', length=self.nspec, dtype='f4'))
    #    star_properties.add_column(Column(name='MAG', length=self.nspec, dtype='f4'))
    #    star_properties.add_column(Column(name='MAGFILTER', length=self.nspec, dtype='U15'))
    #    star_properties.add_column(Column(name='TEFF', length=self.nspec, dtype='f4'))
    #    star_properties.add_column(Column(name='LOGG', length=self.nspec, dtype='f4'))
    #    star_properties.add_column(Column(name='FEH', length=self.nspec, dtype='f4'))
    #    star_properties['REDSHIFT'] = self.rand.uniform(-5E-4, 5E-4, self.nspec)
    #    star_properties['MAG'] = self.rand.uniform(16, 19, self.nspec)
    #    star_properties['MAGFILTER'][:] = 'decam2014-r'
    #    star_properties['TEFF'] = self.rand.uniform(4000, 10000, self.nspec)
    #    star_properties['LOGG'] = self.rand.uniform(0.5, 5.0, self.nspec)
    #    star_properties['FEH'] = self.rand.uniform(-2.0, 0.0, self.nspec)
    #    for T in [STAR]:
    #        Tx = T(wave=self.wave)
    #        flux, wave, meta, objmeta = Tx.make_templates(star_properties=star_properties, seed=self.seed)
    #        badkeys = list()
    #        for key in ('REDSHIFT', 'MAG'):
    #            if not np.allclose(meta[key], star_properties[key]):
    #                badkeys.append(key)
    #        for key in ('TEFF', 'LOGG', 'FEH'):
    #            if not np.allclose(objmeta[key], star_properties[key]):
    #                badkeys.append(key)
    #        self.assertEqual(len(badkeys), 0, 'mismatch for spectral type {} in keys {}'.format(meta['OBJTYPE'][0], badkeys))
    #
    #def test_lyamock_seed(self):
    #    '''Test that random seed works to get the same results back'''
    #    #print('In function test_lyamock_seed, seed = {}'.format(self.seed))
    #    mock = lyamock.MockMaker()
    #    wave1, flux1 = mock.get_lya_skewers(self.nspec, new_seed=1)
    #    wave2, flux2 = mock.get_lya_skewers(self.nspec, new_seed=1)
    #    wave3, flux3 = mock.get_lya_skewers(self.nspec, new_seed=2)
    #    self.assertTrue(np.all(flux1==flux2))
    #    self.assertTrue(np.any(flux1!=flux3))
    #    self.assertTrue(np.all(wave1==wave2))
    #
    #@unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES was not detected.')
    #def test_meta(self):
    #    '''Test the metadata tables have the columns we expect'''
    #    #print('In function test_meta, seed = {}'.format(self.seed))
    #    for T in [ELG, LRG, BGS, STAR, STD, MWS_STAR, WD, QSO]:
    #        template_factory = T(wave=self.wave)
    #        flux, wave, meta, objmeta = template_factory.make_templates(self.nspec, seed=self.seed)
    #
    #        self.assertTrue(np.all(np.in1d(['TARGETID', 'OBJTYPE', 'SUBTYPE', 'TEMPLATEID', 'SEED',
    #                                        'REDSHIFT', 'MAG', 'MAGFILTER', 'FLUX_G', 'FLUX_R',
    #                                        'FLUX_Z', 'FLUX_W1', 'FLUX_W2'],
    #                                        meta.colnames)))
    #
    #        if ( isinstance(template_factory, ELG) or isinstance(template_factory, LRG) or
    #             isinstance(template_factory, BGS) ):
    #            self.assertTrue(np.all(np.in1d(['TARGETID', 'OIIFLUX', 'HBETAFLUX', 'EWOII', 'EWHBETA',
    #                                            'D4000', 'VDISP', 'OIIDOUBLET', 'OIIIHBETA', 'OIIHBETA',
    #                                            'NIIHBETA', 'SIIHBETA'],
    #                                            objmeta.colnames)))
    #            
    #        if (isinstance(template_factory, STAR) or isinstance(template_factory, STD) or
    #            isinstance(template_factory, MWS_STAR) ):
    #            self.assertTrue(np.all(np.in1d(['TARGETID', 'TEFF', 'LOGG', 'FEH'], objmeta.colnames)))
    #
    #        if isinstance(template_factory, WD):
    #            self.assertTrue(np.all(np.in1d(['TARGETID', 'TEFF', 'LOGG'], objmeta.colnames)))
    #
    #        if isinstance(template_factory, QSO):
    #            self.assertTrue(np.all(np.in1d(['TARGETID', 'PCA_COEFF'], objmeta.colnames)))

if __name__ == '__main__':
    unittest.main()
