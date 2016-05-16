import unittest, os
from uuid import uuid1
from shutil import rmtree

import numpy as np

import desimodel.io
import desispec.io

from desisim import io
from desisim import obs
from desisim import pixsim

desi_templates_available = 'DESI_ROOT' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

class TestPixsim(unittest.TestCase):
    #- Create test subdirectory
    @classmethod
    def setUpClass(cls):
        global desi_templates_available
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
        if desi_templates_available:
            cls.cosmics = (os.environ['DESI_ROOT'] +
                '/spectro/templates/cosmics/v0.2/cosmics-bias-r.fits')
        else:
            cls.cosmics = None

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

    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_pixsim(self):
        night = '20150105'
        expid = 123
        camera = 'r0'
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        pixsim.simulate_frame(night, expid, camera, nspec=3)

        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        self.assertTrue(os.path.exists(io.findfile('simpix', night, expid, camera)))
        self.assertTrue(os.path.exists(io.findfile('pix', night, expid, camera)))

    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_pixsim_waveminmax(self):
        night = '20150105'
        expid = 123
        camera = 'r0'
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        pixsim.simulate_frame(night, expid, camera, nspec=3,
            wavemin=6000, wavemax=6100)

        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        self.assertTrue(os.path.exists(io.findfile('simpix', night, expid, camera)))
        self.assertTrue(os.path.exists(io.findfile('pix', night, expid, camera)))

    @unittest.skipUnless(desi_templates_available, 'The DESI templates directory ($DESI_ROOT/spectro/templates) was not detected.')
    def test_pixsim_cosmics(self):
        night = '20150105'
        expid = 124
        camera = 'r0'
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        pixsim.simulate_frame(night, expid, camera, nspec=3, cosmics=self.cosmics)

        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        self.assertTrue(os.path.exists(io.findfile('simpix', night, expid, camera)))
        self.assertTrue(os.path.exists(io.findfile('pix', night, expid, camera)))

    def test_simulate(self):
        import desispec.image
        night = '20150105'
        expid = 124
        camera = 'r0'
        nspec = 3
        obs.new_exposure('arc', night=night, expid=expid, nspec=nspec)
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        psf = desimodel.io.load_psf(camera[0])
        
        image, rawpix, truepix = pixsim.simulate(camera, simspec, psf, nspec=nspec)

        self.assertTrue(isinstance(image, desispec.image.Image))
        self.assertTrue(isinstance(rawpix, np.ndarray))
        self.assertTrue(isinstance(truepix, np.ndarray))
        self.assertEqual(image.pix.shape, truepix.shape)
        self.assertEqual(image.pix.shape[0], rawpix.shape[0])
        self.assertLess(image.pix.shape[1], rawpix.shape[1])  #- raw has overscan

    def test_project(self):
        psf = desimodel.io.load_psf('z')
        wave = np.arange(8000, 8010)
        phot = np.ones((2, len(wave)))
        specmin = 12
        args = psf, wave, phot, specmin
        xyrange, pix = pixsim._project(args)

        with self.assertRaises(ValueError):
            phot = np.ones((2,3,4))
            args = psf, wave, phot, specmin
            os.environ['UNITTEST_SILENT'] = 'TRUE'
            xyrange, pix = pixsim._project(args)
            del os.environ['UNITTEST_SILENT']

    def test_parse(self):
        night = '20151223'
        expid = 1
        opts = ['--psf', 'blat.fits', '--night', night, '--expid', expid]
        args = pixsim.parse(opts)
        self.assertEqual(args.psf, 'blat.fits')
        self.assertEqual(args.night, night)
        self.assertEqual(args.expid, expid)
        
        with self.assertRaises(ValueError):
            pixsim.parse([])

    def test_expand_args(self):
        night = '20151223'
        expid = 1
        opts = ['--psf', 'blat.fits', '--night', night, '--expid', expid]
        args = pixsim.parse(opts)
        self.assertEqual(args.rawfile, desispec.io.findfile('raw', night, expid))

        opts = ['--night', night, '--expid', expid]
        args = pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0','r0','z0'])

        opts = ['--night', night, '--expid', expid, '--spectrographs', '0,1',
            '--arms', 'b,z']
        args = pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0', 'b1', 'z0', 'z1'])

        opts = ['--cameras', 'b0', '--night', night, '--expid', expid]
        args = pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0'])

        opts = ['--cameras', 'b0,r1', '--night', night, '--expid', expid]
        args = pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0','r1'])

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
