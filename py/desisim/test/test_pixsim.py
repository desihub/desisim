import unittest, os
from uuid import uuid1
from shutil import rmtree

import numpy as np

import desimodel.io

from desisim import io
from desisim import obs
from desisim import pixsim

class TestPixsim(unittest.TestCase):
    #- Create test subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_io')
        cls.origEnv = dict(
            PIXPROD = None,
            DESI_SPECTRO_SIM = None,
            DESI_SPECTRO_DATA = None,
        )
        cls.testEnv = dict(
            PIXPROD = 'test',
            DESI_SPECTRO_SIM = os.path.join(cls.testDir,'spectro','sim'),
            DESI_SPECTRO_DATA = os.path.join(cls.testDir,'spectro','sim', 'test'),
            )
        for e in cls.origEnv:
            if e in os.environ:
                cls.origEnv[e] = os.environ[e]
            os.environ[e] = cls.testEnv[e]
            
        cls.cosmics = os.getenv('DESI_ROOT') + \
            '/spectro/templates/cosmics/v0.2/cosmics-bias-r.fits'

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

    # def simulate(night, expid, camera, nspec=None, verbose=False, ncpu=None,
    #     trimxy=False, cosmics=None):
    def test_pixsim(self):
        night = '20150105'
        expid = 123
        camera = 'r0'
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        pixsim.simulate(night, expid, camera, nspec=3, trimxy=True)
        
        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        self.assertTrue(os.path.exists(io.findfile('simpix', night, expid, camera)))
        self.assertTrue(os.path.exists(io.findfile('pix', night, expid, camera)))

    def test_pixsim_cosmics(self):
        night = '20150105'
        expid = 124
        camera = 'r0'
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        pixsim.simulate(night, expid, camera, nspec=3, trimxy=True, cosmics=self.cosmics)

        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        self.assertTrue(os.path.exists(io.findfile('simpix', night, expid, camera)))
        self.assertTrue(os.path.exists(io.findfile('pix', night, expid, camera)))
        
    def test_project(self):
        psf = desimodel.io.load_psf('z')
        wave = np.arange(8000, 8010)
        phot = np.ones((2, len(wave)))
        specmin = 12
        args = psf, wave, phot, specmin
        xyrange, pix = pixsim._project(args)

        print("\nINFO This is supposed to raise a ValueError; don't panic")
        with self.assertRaises(ValueError):
            phot = np.ones((2,3,4))
            args = psf, wave, phot, specmin
            xyrange, pix = pixsim._project(args)
        
        
#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
        