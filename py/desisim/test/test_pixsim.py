import unittest, os
from uuid import uuid1
from shutil import rmtree

import numpy as np
from astropy.io import fits

import desimodel.io
import desispec.io

from desisim import io
from desisim import obs
from desisim import pixsim
import desisim.scripts.pixsim

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

    def setUp(self):
        self.night = '20150105'
        self.expid = 124

    def tearDown(self):
        rawfile = desispec.io.findfile('raw', self.night, self.expid)
        if os.path.exists(rawfile):
            os.remove(rawfile)
        fibermap = desispec.io.findfile('fibermap', self.night, self.expid)
        if os.path.exists(fibermap):
            os.remove(fibermap)
        simspecfile = io.findfile('simspec', self.night, self.expid)
        if os.path.exists(simspecfile):
            os.remove(simspecfile)
        for camera in ('b0', 'r0', 'z0'):
            pixfile = desispec.io.findfile('pix', self.night, self.expid, camera=camera)
            if os.path.exists(pixfile):
                os.remove(pixfile)
            simpixfile = io.findfile('simpix', self.night, self.expid, camera=camera)
            if os.path.exists(simpixfile):
                os.remove(simpixfile)
        

    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_pixsim(self):
        night = self.night
        expid = self.expid
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
        night = self.night
        expid = self.expid
        camera = 'r0'
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        pixsim.simulate_frame(night, expid, camera, nspec=3, cosmics=self.cosmics)

        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        self.assertTrue(os.path.exists(io.findfile('simpix', night, expid, camera)))
        self.assertTrue(os.path.exists(io.findfile('pix', night, expid, camera)))

    def test_simulate(self):
        import desispec.image
        night = self.night
        expid = self.expid
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

    #- Travis tests hang when writing coverage when both test_main1 and
    #- test_main2 are called.  Commenting out the simpler one for now.
    # def test_main1(self):
    #     night = self.night
    #     expid = self.expid
    #     camera = 'r0'
    #     nspec = 3
    #     obs.new_exposure('arc', night=night, expid=expid, nspec=nspec)
    #     
    #     #- run pixsim
    #     opts = ['--night', night, '--expid', expid, '--nspec', nspec]
    #     pixsim.main(opts)
    #     
    #     #- verify outputs
    #     simpixfile = io.findfile('simpix', night, expid)
    #     self.assertTrue(os.path.exists(simpixfile))
    #     rawfile = desispec.io.findfile('raw', night, expid)
    #     self.assertTrue(os.path.exists(rawfile))
    #     fx = fits.open(rawfile)
    #     
    #     self.assertTrue('B0' in fx)
    #     self.assertTrue('R0' in fx)
    #     self.assertTrue('Z0' in fx)
    #     fx.close()
    #     
    #     #- cleanup as we go
    #     os.remove(simpixfile)
    #     os.remove(rawfile)

    def test_main_override(self):
        night = self.night
        expid = self.expid
        camera = 'r0'
        nspec = 3
        obs.new_exposure('arc', night=night, expid=expid, nspec=nspec)

        #- derive night from simspec input while overriding expid
        simspecfile = io.findfile('simspec', night, expid)
        altrawfile = desispec.io.findfile('raw', night, expid) + '.blat'
        opts = [
            '--simspec', simspecfile,
            '--expid', expid+1,
            '--rawfile', altrawfile,
            '--cameras', 'b0,r0',
            '--preproc',
            '--wavemin', 5000, '--wavemax', 7000.0,
            ]
        desisim.scripts.pixsim.main(opts)
        simpixfile = io.findfile('simpix', night, expid+1)
        self.assertTrue(os.path.exists(simpixfile))
        self.assertTrue(os.path.exists(altrawfile))
        fx = fits.open(altrawfile)
        self.assertTrue('B0' in fx)
        self.assertTrue('R0' in fx)
        self.assertTrue('Z0' not in fx)
        fx.close()
        
        #- cleanup as we go
        os.remove(simpixfile)
        os.remove(altrawfile)
        
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
        night = self.night
        expid = self.expid
        opts = ['--psf', 'blat.fits', '--night', night, '--expid', expid]
        opts += ['--spectrographs', '0,3']
        args = desisim.scripts.pixsim.parse(opts)
        self.assertEqual(args.psf, 'blat.fits')
        self.assertEqual(args.night, night)
        self.assertEqual(args.expid, expid)
        self.assertEqual(args.spectrographs, [0,3])
        self.assertEqual(args.cameras, ['b0', 'b3', 'r0', 'r3', 'z0', 'z3'])
        
        with self.assertRaises(ValueError):
            desisim.scripts.pixsim.parse([])

    def test_expand_args(self):
        night = self.night
        expid = self.expid

        opts = ['--night', night, '--expid', expid, '--spectrographs', '0']
        args = desisim.scripts.pixsim.parse(opts)
        self.assertEqual(args.rawfile, desispec.io.findfile('raw', night, expid))
        self.assertEqual(args.cameras, ['b0','r0','z0'])

        opts = ['--night', night, '--expid', expid, '--spectrographs', '0,1',
            '--arms', 'b,z']
        args = desisim.scripts.pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0', 'b1', 'z0', 'z1'])

        opts = ['--cameras', 'b0', '--night', night, '--expid', expid]
        args = desisim.scripts.pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0'])

        opts = ['--cameras', 'b0,r1', '--night', night, '--expid', expid]
        args = desisim.scripts.pixsim.parse(opts)
        self.assertEqual(args.cameras, ['b0','r1'])

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
