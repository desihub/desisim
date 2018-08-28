import unittest, os
from uuid import uuid1
from shutil import rmtree

import numpy as np
from astropy.io import fits
from astropy.table import Table

from desisim import io
from desisim import obs
import desisim.simexp

desimodel_data_available = 'DESIMODEL' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

class TestObs(unittest.TestCase):
    #- Create test subdirectory
    @classmethod
    def setUpClass(cls):
        cls.night = '20150101'
        cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_io')
        cls.origEnv = dict(PIXPROD = None, DESI_SPECTRO_SIM = None)
        cls.testEnv = dict(
            PIXPROD = 'test',
            DESI_SPECTRO_SIM = os.path.join(cls.testDir,'spectro','sim'),
            )
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]

    #- Cleanup files but not directories after each test
    def tearDown(self):
        for expid in range(5):
            for filetype in ['simspec', 'simfibermap']:
                filename = io.findfile('simspec', self.night, expid)
                if os.path.exists(filename):
                    os.remove(filename)

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.testfile):
            os.remove(cls.testfile)
            testpath = os.path.normpath(os.path.dirname(cls.testfile))
            if testpath != '.':
                os.removedirs(testpath)
        for e in cls.origEnv:
            if cls.origEnv[e] is None:
                del os.environ[e]
            else:
                os.environ[e] = cls.origEnv[e]
        if os.path.exists(cls.testDir):
            rmtree(cls.testDir)

    # def new_exposure(program, nspec=5000, night=None, expid=None, tileid=None, \
    #     airmass=1.0, exptime=None):
    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_newexp(self):
        night = self.night
        seed = np.random.randint(2**30)
        #- programs 'bgs' and 'bright' not yet implemented
        for expid, program in enumerate(['arc', 'flat', 'dark', 'mws']):
            sim, fibermap, meta, obsconditions, objmeta = obs.new_exposure(program, nspec=10, night=night, expid=expid, seed=seed)
            simspecfile = io.findfile('simspec', night, expid=expid)
            fibermapfile = io.findfile('simfibermap', night, expid=expid)
            self.assertTrue(os.path.exists(simspecfile))
            self.assertTrue(os.path.exists(fibermapfile))
            simspec = io.read_simspec(simspecfile)
            if program in ('arc', 'flat'):
                self.assertEqual(simspec.flavor, program)
            else:
                self.assertEqual(simspec.flavor, 'science')

            #- Check that photons are in a reasonable range
            self.assertGreater(len(simspec.cameras), 0)
            for camera in simspec.cameras.values():
                maxphot = camera.phot.max()
                self.assertTrue(maxphot > 1, 'suspiciously few {} photons ({}); wrong units?'.format(program, maxphot))
                self.assertTrue(maxphot < 1e6, 'suspiciously many {} photons ({}); wrong units?'.format(program, maxphot))
                if program not in ('arc', 'flat'):
                    self.assertTrue(camera.skyphot.max() > 1, 'suspiciously few sky photons; wrong units?')
                    self.assertTrue(camera.skyphot.max() < 1e6, 'suspiciously many sky photons; wrong units?')

            if program in ('arc', 'flat'):
                self.assertTrue(meta is None)
                self.assertTrue(obsconditions is None)
            else:
                flux, fluxhdr = fits.getdata(simspecfile, 'FLUX', header=True)
                skyflux, skyfluxhdr = fits.getdata(simspecfile, 'SKYFLUX', header=True)
                self.assertTrue(fluxhdr['BUNIT'].startswith('1e-17'))
                self.assertTrue(skyfluxhdr['BUNIT'].startswith('1e-17'))
                for i in range(flux.shape[0]):
                    objtype = simspec.truth['OBJTYPE'][i]
                    maxflux = flux[i].max()
                    maxsky = skyflux[i].max()
                    self.assertTrue(maxsky > 1, 'suspiciously low {} sky flux ({}); wrong units?'.format(objtype, maxsky))
                    self.assertTrue(maxsky < 1e5, 'suspiciously high {} sky flux ({}); wrong units?'.format(objtype, maxsky))
                    if objtype != 'SKY':
                        ### print('---> {} maxflux {}'.format(objtype, maxflux))
                        self.assertTrue(maxflux > 0.01, 'suspiciously low {} flux ({}) using seed {}; wrong units?'.format(objtype, maxflux, seed))
                        self.assertTrue(maxflux < 1e5, 'suspiciously high {} flux ({}) using seed {}; wrong units?'.format(objtype, maxflux, seed))
                    else:
                        self.assertTrue(np.all(flux[i] == 0.0))

            os.remove(simspecfile)
            os.remove(fibermapfile)

        #- confirm that night and expid are optional
        results = obs.new_exposure('arc', nspec=2)

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_newexp_sky(self):
        "Test different levels of sky brightness"
        night = self.night
        #- programs 'bgs' and 'bright' not yet implemented
        sim_dark, fmap_dark, meta_dark, obscond_dark, objmeta_dark = obs.new_exposure('dark', nspec=10, night=night, expid=0, exptime=1000)
        dark = sim_dark.simulated.copy()
        sim_mws, fmap_mws, meta_mws, obscond_mws, objmeta_mws = obs.new_exposure('mws', nspec=10, night=night, expid=1, exptime=1000)
        mws = sim_dark.simulated.copy()
        for channel in ['b', 'r', 'z']:
            sky_mws = mws['num_sky_electrons_'+channel]
            sky_dark = dark['num_sky_electrons_'+channel]
            nonzero = (sky_mws != 0.0)
            self.assertTrue(np.all(sky_mws[nonzero] > sky_dark[nonzero]))

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_update_obslog(self):
        #- These shouldn't fail, but we don't really have verification
        #- code that they did anything correct.
        expid, dateobs = obs.update_obslog(expid=1)
        self.assertEqual(expid, 1)
        expid, dateobs = obs.update_obslog(obstype='arc', program='calib', expid=2)
        self.assertEqual(expid, 2)
        expid, dateobs = obs.update_obslog(obstype='science', expid=3, tileid=1)
        expid, dateobs = obs.update_obslog(obstype='science', expid=3,
            tileid=1, ra=0.1, dec=2.3)

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_get_next_tileid(self):
        #- Two tileid request without an observation should be the same
        a = obs.get_next_tileid()
        b = obs.get_next_tileid()
        self.assertEqual(a, b)

        #- But then register the obs, and we should get a different tile
        print('### Updating obslog ###')
        obs.update_obslog(expid=0, tileid=a)
        print('### Getting more tiles ###')
        c = obs.get_next_tileid()        
        self.assertNotEqual(a, c)
        
        #- different programs should be different tiles
        a = obs.get_next_tileid(program='dark')
        b = obs.get_next_tileid(program='gray')
        c = obs.get_next_tileid(program='bright')
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        
        #- program is case insensitive
        a = obs.get_next_tileid(program='GRAY')
        b = obs.get_next_tileid(program='gray')
        self.assertEqual(a, b)

    def test_specter_objtype(self):
        self.assertEqual(obs.specter_objtype('MWS_STAR'), 'STAR')
        self.assertEqual(obs.specter_objtype(['MWS_STAR',])[0], 'STAR')
        a = np.array(['STAR', 'MWS_STAR', 'QSO_BAD', 'STD', 'QSO', 'ELG'])
        b = np.array(['STAR', 'STAR', 'STAR', 'STAR', 'QSO', 'ELG'])
        self.assertTrue(np.all(obs.specter_objtype(a) == b))

    def test_get_next_expid(self):
        a = obs.get_next_expid()
        b = obs.get_next_expid()
        c = obs.get_next_expid()
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        
    def test_testslit_fibermap(self):
        #- Should have one fiber per bundle = 10*20 = 200
        fm = desisim.simexp.testslit_fibermap()
        self.assertTrue(len(fm) == 200)     #- 10 spectro * 20 bundles
        self.assertTrue(len(set(fm['FIBER'])) == 200)   #- unique fibers
        for i in range(10):
            self.assertIn(i*500, fm['FIBER'])
            self.assertIn(i*500+499, fm['FIBER'])
    

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
