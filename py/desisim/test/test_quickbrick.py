from __future__ import absolute_import, division, print_function

import unittest
from uuid import uuid1
from shutil import rmtree
import os

import numpy as np
from astropy.io import fits
from desisim.scripts import quickbrick
import desispec.io

class TestQuickBrick(unittest.TestCase):
    
    #- Pick random test directory
    def setUp(self):
        self.testdir = 'test-{uuid}'.format(uuid=uuid1())

    #- Cleanup test files if they exist
    def tearDown(self):
        if os.path.exists(self.testdir):
            rmtree(self.testdir)
            
    #- Run basic test of quickbrick and check its outputs
    @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickbrick(self):
        brickname = 'test1'
        nspec = 3
        cmd = "quickbrick -b {} --objtype ELG -n {} --outdir {}".format(brickname, nspec, self.testdir)
        args = quickbrick.parse(cmd.split()[1:])
        results = quickbrick.main(args)
        #- Do the bricks exist?
        for channel in ['z', 'r', 'b']:
            brickfile = '{}/brick-{}-{}.fits'.format(self.testdir, channel, brickname)
            self.assertTrue(os.path.exists(brickfile))
            
        #- Check one of them for correct dimensionality of contents
        brickfile = '{}/brick-b-{}.fits'.format(self.testdir, brickname)
        with fits.open(brickfile) as fx:
            self.assertEqual(fx['FLUX'].shape[0], nspec)
            self.assertEqual(fx['FLUX'].shape, fx['IVAR'].shape)
            self.assertEqual(fx['FLUX'].shape[1], fx['WAVELENGTH'].shape[0])
            self.assertEqual(fx['FLUX'].shape[0], fx['RESOLUTION'].shape[0])
            self.assertEqual(fx['FLUX'].shape[1], fx['RESOLUTION'].shape[2])
            self.assertEqual(len(fx['FIBERMAP'].data), nspec)
            self.assertEqual(fx['_TRUEFLUX'].shape, fx['FLUX'].shape)
            self.assertTrue(np.all(fx['_TRUTH'].data['OBJTYPE'] == 'ELG'))
        
        #- Make sure that desispec I/O functions can read it
        frame = desispec.io.read_frame(brickfile)
        self.assertEqual(frame.flux.shape[0], nspec)
        brick = desispec.io.Brick(brickfile)
        self.assertEqual(brick.get_num_spectra(), nspec)
        brick.close()

    #- Test quickbrick with a bunch of options
    @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickbrick_options(self):
        brickname = 'test2'
        nspec = 3
        cmd = "quickbrick -b {} --objtype BGS -n {} --outdir {}".format(brickname, nspec, self.testdir)
        cmd = cmd + " --airmass 1.5 --verbose --zrange-bgs 0.1 0.2"
        cmd = cmd + " --moon-phase 0.1 --moon-angle 30 --moon-zenith 20"
        args = quickbrick.parse(cmd.split()[1:])
        results = quickbrick.main(args)
        brickfile = '{}/brick-b-{}.fits'.format(self.testdir, brickname)
        truth = fits.getdata(brickfile, '_TRUTH')        
        z = truth['TRUEZ']
        self.assertTrue(np.all((0.1 <= z) & (z <= 0.2)))

    #- Ensure that using --seed results in reproducible spectra
    @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickbrick_seed(self):
        nspec = 2
        cmd = "quickbrick --seed 1 -b test1a --objtype BRIGHT_MIX -n {} --outdir {}".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        cmd = "quickbrick --seed 1 -b test1b --objtype BRIGHT_MIX -n {} --outdir {}".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        cmd = "quickbrick --seed 2 -b test2 --objtype BRIGHT_MIX -n {} --outdir {}".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        
        f1a = desispec.io.read_frame('{}/brick-b-test1a.fits'.format(self.testdir))
        f1b = desispec.io.read_frame('{}/brick-b-test1b.fits'.format(self.testdir))
        f2 = desispec.io.read_frame('{}/brick-b-test2.fits'.format(self.testdir))
        self.assertTrue(np.all(f1a.flux == f1b.flux))  #- same seed
        self.assertTrue(np.all(f1a.ivar == f1b.ivar))
        self.assertTrue(np.any(f1a.flux != f2.flux))   #- different seed

        t1a = fits.getdata('{}/brick-b-test1a.fits'.format(self.testdir), '_TRUTH')
        t1b = fits.getdata('{}/brick-b-test1b.fits'.format(self.testdir), '_TRUTH')
        t2 = fits.getdata('{}/brick-b-test2.fits'.format(self.testdir), '_TRUTH')
        self.assertTrue(np.all(t1a['TRUE_OBJTYPE'] == t1b['TRUE_OBJTYPE']))
        self.assertTrue(np.all(t1b['TRUEZ'] == t1b['TRUEZ']))
        self.assertTrue(np.any(t1a['TRUEZ'] != t2['TRUEZ']))

    #- Test that brighter moon makes noisier spectra
    @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickbrick_moon(self):
        nspec = 2
        cmd = "quickbrick --seed 1 -b brightmoon --objtype BRIGHT_MIX -n {} --outdir {} --moon-phase 0.1".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        cmd = "quickbrick --seed 1 -b crescentmoon --objtype BRIGHT_MIX -n {} --outdir {} --moon-phase 0.9".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        
        brickfile = '{}/brick-b-brightmoon.fits'.format(self.testdir)
        brick1 = desispec.io.read_frame(brickfile)
        brickfile = '{}/brick-b-crescentmoon.fits'.format(self.testdir)
        brick2 = desispec.io.read_frame(brickfile)
        #- brick1 has more moonlight thus larger errors thus smaller ivar
        self.assertLess(np.median(brick1.ivar), np.median(brick2.ivar))

    #- Test that shorter exposures make noisier spectra
    @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickbrick_exptime(self):
        nspec = 2
        cmd = "quickbrick --seed 1 -b test1 --exptime 100 --objtype DARK_MIX -n {} --outdir {}".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        cmd = "quickbrick --seed 1 -b test2 --exptime 1000 --objtype DARK_MIX -n {} --outdir {}".format(nspec, self.testdir)
        quickbrick.main(quickbrick.parse(cmd.split()[1:]))
        
        f1 = desispec.io.read_frame('{}/brick-b-test1.fits'.format(self.testdir))
        f2 = desispec.io.read_frame('{}/brick-b-test2.fits'.format(self.testdir))
        #- test1 has shorter exposure time thus larger errors thus smaller ivar
        self.assertLess(np.median(f1.ivar), np.median(f2.ivar))

if __name__ == '__main__':
    unittest.main()
