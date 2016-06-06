import unittest, os
from uuid import uuid1
from shutil import rmtree

import numpy as np
from astropy.io import fits

from desisim import io
from desisim import obs

desimodel_data_available = 'DESIMODEL' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

class TestObs(unittest.TestCase):
    #- Create test subdirectory
    @classmethod
    def setUpClass(cls):
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

    # def new_exposure(flavor, nspec=5000, night=None, expid=None, tileid=None, \
    #     airmass=1.0, exptime=None):
    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_newexp(self):
        night = '20150101'
        #- flavors 'bgs' and 'bright' not yet implemented
        for expid, flavor in enumerate(['arc', 'flat', 'dark', 'mws']):
            fibermap, true = obs.new_exposure(flavor, nspec=10, night=night, expid=expid)
            simspecfile = io.findfile('simspec', night, expid=expid)
            self.assertTrue(os.path.exists(simspecfile))
            simspec = io.read_simspec(simspecfile)
            self.assertEqual(simspec.flavor, flavor)

            #- Check that photons are in a reasonable range
            for channel in ('b', 'r', 'z'):
                maxphot = simspec.phot[channel].max()
                self.assertTrue(maxphot > 1, 'suspiciously few {} photons ({}); wrong units?'.format(flavor, maxphot))
                self.assertTrue(maxphot < 1e6, 'suspiciously many {} photons ({}); wrong units?'.format(flavor, maxphot))
                if flavor not in ('arc', 'flat'):
                    self.assertTrue(simspec.skyphot[channel].max() > 1, 'suspiciously few sky photons; wrong units?')
                    self.assertTrue(simspec.skyphot[channel].max() < 1e6, 'suspiciously many sky photons; wrong units?')

            if flavor not in ('arc', 'flat'):
                fx = fits.open(simspecfile)
                self.assertTrue(fx['FLUX'].header['BUNIT'].startswith('1e-17 '))
                self.assertTrue(fx['SKYFLUX'].header['BUNIT'].startswith('1e-17 '))
                flux = fx['FLUX'].data
                skyflux = fx['SKYFLUX'].data
                for i in range(flux.shape[0]):
                    objtype = simspec.metadata['OBJTYPE'][i]
                    maxflux = flux[i].max()
                    maxsky = skyflux[i].max()
                    self.assertTrue(maxsky > 1, 'suspiciously low {} sky flux ({}); wrong units?'.format(objtype, maxsky))
                    self.assertTrue(maxsky < 1e5, 'suspiciously high {} sky flux ({}); wrong units?'.format(objtype, maxsky))
                    if objtype != 'SKY':
                        self.assertTrue(maxflux > 0.1, 'suspiciously low {} flux ({}); wrong units?'.format(objtype, maxflux))
                        self.assertTrue(maxflux < 1e5, 'suspiciously high {} flux ({}); wrong units?'.format(objtype, maxflux))
                    else:
                        self.assertTrue(np.all(flux[i] == 0.0))

                fx.close()

        #- confirm that night and expid are optional
        fibermap, true = obs.new_exposure('arc', nspec=2)

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_newexp_sky(self):
        "Test different levels of sky brightness"
        night = '20150101'
        #- flavors 'bgs' and 'bright' not yet implemented
        fibermap, truth_dark = obs.new_exposure('dark', nspec=10, night=night, expid=0)
        fibermap, truth_mws  = obs.new_exposure('mws', nspec=10, night=night, expid=1)
        for channel in ['B', 'R', 'Z']:
            key = 'SKYPHOT_'+channel
            self.assertTrue(np.all(truth_mws[key] > truth_dark[key]))
        

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_update_obslog(self):
        #- These shouldn't fail, but we don't really have verification
        #- code that they did anything correct.
        expid, dateobs = obs.update_obslog(expid=1)
        self.assertEqual(expid, 1)
        expid, dateobs = obs.update_obslog(obstype='arc', expid=2)
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
        obs.update_obslog(expid=0, tileid=a)
        c = obs.get_next_tileid()
        self.assertNotEqual(a, c)

    def test_specter_objtype(self):
        self.assertEqual(obs.specter_objtype('MWS_STAR'), 'STAR')
        self.assertEqual(obs.specter_objtype(['MWS_STAR',])[0], 'STAR')
        a = np.array(['STAR', 'MWS_STAR', 'QSO_BAD', 'FSTD', 'QSO', 'ELG'])
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
        fm = obs.testslit_fibermap()
        self.assertTrue(len(fm) == 200)     #- 10 spectro * 20 bundles
        self.assertTrue(len(set(fm['FIBER'])) == 200)   #- unique fibers
        for i in range(10):
            self.assertIn(i*500, fm['FIBER'])
            self.assertIn(i*500+499, fm['FIBER'])
    

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
