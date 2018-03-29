import unittest, os
from uuid import uuid1
from shutil import rmtree

import numpy as np

import desisim
from desisim import io
from astropy.io import fits

desimodel_data_available = 'DESIMODEL' in os.environ
desi_templates_available = 'DESI_ROOT' in os.environ
desi_basis_templates_available = 'DESI_BASIS_TEMPLATES' in os.environ

class TestIO(unittest.TestCase):

    #- Create unique test filename in a subdirectory
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

    def test_simdir(self):
        x = io.simdir()
        self.assertTrue(x is not None)
        night = '20150101'
        x = io.simdir(night)
        self.assertTrue(x.endswith(night))
        x = io.simdir(night, mkdir=True)
        self.assertTrue(os.path.exists(x))

    def test_findfile(self):
        night = '20150102'
        expid = 3
        camera = 'z3'
        filepath = io.findfile('simspec', night, expid)
        filepath = io.findfile('simpix', night, expid, camera)
        outdir = '/blat/foo/bar'
        filepath = io.findfile('simpix', night, expid, camera, outdir=outdir, mkdir=False)
        self.assertTrue(filepath.startswith(outdir))

        with self.assertRaises(ValueError):
            io.findfile('blat', night, expid, camera)  #- bad filetype

    def test_write_simpix(self):
        outfile = io.findfile('simpix', '20150104', 5, 'z3')
        pix = np.random.uniform( (5,10) )
        meta = dict(BLAT='foo', BAR='biz')
        io.write_simpix(outfile, pix, camera='b0', meta=meta)
        image, header = fits.getdata(outfile, 'B0', header=True)
        self.assertEqual(image.dtype.itemsize, 4)
        self.assertTrue(np.all(pix.astype(np.float32) == image))
        for key in meta:
            self.assertTrue(meta[key] == header[key])

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_get_tile_radec(self):
        ra, dec = io.get_tile_radec(0)
        ra, dec = io.get_tile_radec(1)
        ra, dec = io.get_tile_radec(2)
        ra, dec = io.get_tile_radec(-1)
        self.assertTrue( (ra,dec) == (0.0, 0.0) )
        with self.assertRaises(ValueError):
            ra, dec = io.get_tile_radec('blat')

    def test_resize(self):
        for origsize in [(4,5), (4,4), (7,5)]:
            image = np.random.uniform(size=(4,5))
            for ny in [3,4,5,7,9,15]:
                for nx in [3,5,7,19]:
                    shape = (ny, nx)
                    x = io._resize(image, shape)
                    self.assertEqual(x.shape, shape)
                
        #- sub- and super- selection should remain centered on original image
        image = np.random.uniform(size=(4,5))
        tmp = io._resize(image, (4,3))
        self.assertTrue(np.all(tmp == image[:, 1:-1]))
        tmp = io._resize(image, (4,7))
        self.assertTrue(np.all(tmp[:,1:-1] == image))

    #- read_cosmics(filename, expid=1, shape=None, jitter=True):
    @unittest.skipUnless(desi_templates_available, 'The DESI templates directory ($DESI_ROOT/spectro/templates) was not detected.')
    def test_read_cosmics(self):
        #- hardcoded cosmics version
        infile = (os.environ['DESI_ROOT'] +
            '/spectro/templates/cosmics/v0.2/cosmics-bias-r.fits')

        shape = (10, 11)
        c1 = io.read_cosmics(infile, expid=0, shape=shape, jitter=True)
        self.assertEqual(c1.pix.shape, shape)
        c2 = io.read_cosmics(infile, expid=0, shape=shape, jitter=False)
        self.assertEqual(c2.pix.shape, shape)
        #- A different exposure should return different cosmics
        c3 = io.read_cosmics(infile, expid=1, shape=shape, jitter=False)
        self.assertTrue(np.any(c2.pix != c3.pix))

    #- read_templates(wave, objtype, nspec=None, seed=1, infile=None):
    @unittest.skipUnless(desi_basis_templates_available, '$DESI_BASIS_TEMPLATES not set')
    def test_read_templates(self):
        wave = np.arange(7000, 7020)
        nspec = 3
        for objtype in ['ELG', 'LRG', 'QSO', 'STAR', 'WD']:
            flux, wave1, meta = io.read_basis_templates(objtype, outwave=wave, nspec=3)
            ntemplates, nwave = flux.shape
            self.assertEqual(nwave, len(wave))
            self.assertEqual(ntemplates, nspec)
            self.assertEqual(len(meta), nspec)

    def test_parse_filename(self):
        prefix, camera, expid = io._parse_filename('/blat/foo/simspec-00000002.fits')
        self.assertEqual(prefix, 'simspec')
        self.assertEqual(camera, None)
        self.assertEqual(expid, 2)
        prefix, camera, expid = io._parse_filename('/blat/foo/preproc-r2-00000003.fits')
        self.assertEqual(prefix, 'preproc')
        self.assertEqual(camera, 'r2')
        self.assertEqual(expid, 3)

    #- TODO
    #- simspec_io
    #- read_cosmics



#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
