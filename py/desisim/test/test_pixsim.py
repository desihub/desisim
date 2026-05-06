import unittest, os, sys
import tempfile

import numpy as np
from astropy.io import fits

import desimodel.io
import desispec.io

from desisim import io
from desisim import obs
from desisim import pixsim
import desisim.scripts.pixsim

from desiutil.log import get_logger
log = get_logger()

desi_templates_available = 'DESI_ROOT' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

class TestPixsim(unittest.TestCase):
    #- Class-wide constants that do not write output
    @classmethod
    def setUpClass(cls):
        if desi_templates_available:
            cls.cosmics = (os.environ['DESI_ROOT'] +
                '/spectro/templates/cosmics/v0.3/cosmics-bias-r.fits')
        else:
            cls.cosmics = None

        #- to save memory while testing
        cls.ccdshape = (2000,2000)

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory(prefix='desi_test_pixsim-')
        self.testdir = self._tempdir.name
        pixprod = 'test-{}'.format(os.path.basename(self.testdir))
        self.origEnv = dict(
            PIXPROD = None,
            SPECPROD = None,
            DESI_SPECTRO_SIM = None,
            DESI_SPECTRO_DATA = None,
            DESI_SPECTRO_REDUX = None,
        )
        self.testEnv = dict(
            PIXPROD = pixprod,
            SPECPROD = pixprod,
            DESI_SPECTRO_SIM = os.path.join(self.testdir,'spectro','sim'),
            DESI_SPECTRO_DATA = os.path.join(self.testdir,'spectro','sim', pixprod),
            DESI_SPECTRO_REDUX = os.path.join(self.testdir,'spectro','redux'),
            )
        for e in self.origEnv:
            if e in os.environ:
                self.origEnv[e] = os.environ[e]
            os.environ[e] = self.testEnv[e]

        self.night = '20150105'
        self.expid = 124
        for expid in (self.expid, self.expid+1):
            pixfile = desispec.io.findfile('preproc', self.night, expid, camera='b0')
            pixdir = os.path.dirname(pixfile)
            if not os.path.isdir(pixdir):
                os.makedirs(pixdir)

    def tearDown(self):
        for e in self.origEnv:
            if self.origEnv[e] is None:
                os.environ.pop(e, None)
            else:
                os.environ[e] = self.origEnv[e]
        self._tempdir.cleanup()


    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_pixsim(self):
        night = self.night
        expid = self.expid
        cameras = ['r0']
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        self.assertTrue(os.path.exists(io.findfile('simspec', night, expid)))

        simspecfile = io.findfile('simspec', night, expid)
        rawfile = desispec.io.findfile('desi', night, expid)
        simpixfile = io.findfile('simpix', night, expid)

        self.assertFalse(os.path.exists(simpixfile))
        self.assertFalse(os.path.exists(rawfile))

        pixsim.simulate_exposure(simspecfile, rawfile, cameras,
            ccdshape=self.ccdshape,
            addcosmics=False, simpixfile=simpixfile)

        self.assertTrue(os.path.exists(simpixfile))
        self.assertTrue(os.path.exists(rawfile))

    @unittest.skipUnless(desi_templates_available, 'The DESI templates directory ($DESI_ROOT/spectro/templates) was not detected.')
    def test_pixsim_cosmics(self):
        night = self.night
        expid = self.expid
        cameras = ['r0']
        obs.new_exposure('arc', night=night, expid=expid, nspec=3)
        simspecfile = io.findfile('simspec', night, expid)
        rawfile = desispec.io.findfile('desi', night, expid)
        simpixfile = io.findfile('simpix', night, expid, cameras)

        self.assertFalse(os.path.exists(simpixfile))
        self.assertFalse(os.path.exists(rawfile))

        pixsim.simulate_exposure(simspecfile, rawfile, cameras,
                addcosmics=True, ccdshape=self.ccdshape)

        self.assertTrue(os.path.exists(rawfile))

        #- No simpixfile option, shouldn't exist
        self.assertFalse(os.path.exists(simpixfile))

    def test_simulate(self):
        import desispec.image
        night = self.night
        expid = self.expid
        camera = 'r0'
        nspec = 3
        obs.new_exposure('arc', night=night, expid=expid, nspec=nspec)
        simspec = io.read_simspec(io.findfile('simspec', night, expid))
        psf = desimodel.io.load_psf(camera[0])
        psf.npix_y, psf.npix_x = self.ccdshape

        image, rawpix, truepix = pixsim.simulate(camera, simspec, psf,
            nspec=nspec, preproc=False)

        self.assertTrue(isinstance(image, desispec.image.Image))
        self.assertTrue(isinstance(rawpix, np.ndarray))
        self.assertTrue(isinstance(truepix, np.ndarray))
        self.assertEqual(image.pix.shape, truepix.shape)
        self.assertEqual(image.pix.shape[0], rawpix.shape[0])
        self.assertLess(image.pix.shape[1], rawpix.shape[1])  #- raw has overscan

    def test_get_nodes_per_exp(self):
        # nodes_per_comm_exp = get_nodes_per_exp(nnodes, nexposures, ncameras)

        self.assertEqual(pixsim.get_nodes_per_exp(6,2,30), 6)
        self.assertEqual(pixsim.get_nodes_per_exp(30,2,30), 30)
        self.assertEqual(pixsim.get_nodes_per_exp(9,3,21), 3)
        self.assertEqual(pixsim.get_nodes_per_exp(17,3,17), 17)
        self.assertEqual(pixsim.get_nodes_per_exp(12,12,6), 6)

        #- Now prints warning but isn't an error
        # with self.assertRaises(ValueError):
        #     pixsim.get_nodes_per_exp(34,3,17)   #- 3*17 % 34 != 0

        #- TODO: add more failure cases

    #- Travis tests hang when writing coverage when both test_main* were
    #- called, though the tests work on other systems.
    #- Disabling multiprocessing also "fixed" this for unknown reasons.
    @unittest.skipIf(False, 'Skip test that is causing coverage tests to hang.')
    def test_main_defaults(self):
        night = self.night
        expid = self.expid
        camera = 'r0'
        nspec = 3
        ncpu = 3
        obs.new_exposure('arc', night=night, expid=expid, nspec=nspec)

        #- run pixsim
        simspec = io.findfile('simspec', night, expid)
        simpixfile = io.findfile('simpix', night, expid)
        rawfile = desispec.io.findfile('raw', night, expid)
        opts = ['--simspec', simspec,'--simpixfile', simpixfile, '--rawfile', rawfile]

        if ncpu is not None:
            opts.extend( ['--ncpu', ncpu] )

        log.debug('testing pixsim.main({})'.format(opts))
        pixsimargs = desisim.scripts.pixsim.parse(opts)
        desisim.scripts.pixsim.main(pixsimargs)

        #- verify outputs
        self.assertTrue(os.path.exists(simpixfile))
        self.assertTrue(os.path.exists(rawfile))
        fx = fits.open(rawfile)

        self.assertTrue('B0' in fx)
        self.assertTrue('R0' in fx)
        self.assertTrue('Z0' in fx)
        fx.close()

        #- cleanup as we go
        os.remove(simpixfile)
        os.remove(rawfile)


    def test_main_override(self):
        night = self.night
        expid = self.expid
        camera = 'r0'
        nspec = 3
        ncpu = 3
        obs.new_exposure('arc', night=night, expid=expid, nspec=nspec)

        #- derive night from simspec input while overriding expid
        #- Include wavelengths covering z, but only ask for b and r
        simspecfile = io.findfile('simspec', night, expid)
        altexpid = expid+1
        altrawfile = desispec.io.findfile('raw', night, altexpid) + '.blat'
        opts = [
            '--simspec', simspecfile,
            '--keywords', f'EXPID={expid}',
            '--rawfile', altrawfile,
            '--cameras', 'b0,r0',
            '--wavemin', 5500, '--wavemax', 7000.0,
            '--ccd_npix_x', 2000,
            ]
        if ncpu is not None:
            opts.extend( ['--ncpu', ncpu] )

        dirname = os.path.dirname(altrawfile)
        if not os.path.isdir(dirname) :
            os.makedirs(dirname)

        log.debug('testing pixsim.main({})'.format(opts))
        pixsimargs = desisim.scripts.pixsim.parse(opts)
        desisim.scripts.pixsim.main(pixsimargs)

        self.assertTrue(os.path.exists(altrawfile))
        fx = fits.open(altrawfile)
        self.assertTrue('B0' in fx)
        self.assertTrue('R0' in fx)
        self.assertTrue('Z0' not in fx)
        fx.close()

        #- cleanup as we go
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

        simspec = io.findfile('simspec', night, expid)
        simpixfile = io.findfile('simpix', night, expid)
        rawfile = desispec.io.findfile('raw', night, expid)
        opts = ['--simspec', simspec,'--simpixfile', simpixfile, '--rawfile', rawfile]
        opts.extend(['--cameras', 'b0,r1'])
        args = desisim.scripts.pixsim.parse(opts)
        self.assertEqual(args.rawfile, rawfile)
        self.assertEqual(args.simspec, simspec)
        self.assertEqual(args.cameras, ['b0','r1'])

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
