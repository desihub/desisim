"""
Run unit tests through quickgen
"""
import unittest, os
from uuid import uuid1
from shutil import rmtree

from astropy.io import fits
import numpy as np
import desispec.io
from desisim import io
from desisim import obs
from desisim.scripts import quickgen
from desiutil.log import get_logger
log = get_logger()

desi_templates_available = 'DESI_ROOT' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

class TestQuickgen(unittest.TestCase):
    def check_env():
        """
        Check required environment variables; raise RuntimeException if missing
        """
        log = logging.get_logger()
        #- template locations
        missing_env = False
        if 'DESI_BASIS_TEMPLATES' not in os.environ:
            log.warning('missing $DESI_BASIS_TEMPLATES needed for simulating spectra'.format(name))
            missing_env = True
    
        if not os.path.isdir(os.getenv('DESI_BASIS_TEMPLATES')):
            log.warning('missing $DESI_BASIS_TEMPLATES directory')
            log.warning('e.g. see NERSC:/project/projectdirs/desi/spectro/templates/basis_templates/v1.0')
            missing_env = True
    
        for name in (
            'DESI_SPECTRO_SIM', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'SPECPROD', 'DESIMODEL'):
            if name not in os.environ:
                log.warning("missing ${0}".format(name))
                missing_env = True
    
        if missing_env:
            log.warning("Why are these needed?")
            log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
            log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
            log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$SPECPROD/")
            log.warning("    Templates are read from $DESI_BASIS_TEMPLATES")
    
        #- Wait until end to raise exception so that we report everything that
        #- is missing before actually failing
        if missing_env:
            log.critical("missing env vars; exiting without running pipeline")
            sys.exit(1)

    #- Create test subdirectory
    @classmethod
    def setUpClass(cls):
        global desi_templates_available
        cls.testfile = 'test-{uuid}/test-{uuid}.fits'.format(uuid=uuid1())
        cls.testDir = os.path.join(os.environ['HOME'],'desi_test_io')
        cls.origEnv = dict(
            PIXPROD = None,
            SPECPROD = None,
            DESI_SPECTRO_SIM = None,
            DESI_SPECTRO_DATA = None,
            DESI_SPECTRO_REDUX = None,
        )
        cls.testEnv = dict(
            PIXPROD = 'test-quickgen',
            SPECPROD = 'test-quickgen',
            DESI_SPECTRO_SIM = os.path.join(cls.testDir,'spectro','sim'),
            DESI_SPECTRO_DATA = os.path.join(cls.testDir,'spectro','sim', 'test-quickgen'),
            DESI_SPECTRO_REDUX = os.path.join(cls.testDir,'spectro','redux'),
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

    def setUp(self):
        self.night = '20150105'
        self.expid = 124

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

    def tearDown(self):
        for expid in [self.expid, 100, 101, 102]:
            fibermap = desispec.io.findfile('fibermap', self.night, expid)
            if os.path.exists(fibermap):
                os.remove(fibermap)
            simspecfile = io.findfile('simspec', self.night, expid)
            if os.path.exists(simspecfile):
                os.remove(simspecfile)
            for camera in ('b0', 'r0', 'z0'):
                framefile = desispec.io.findfile('frame', self.night, expid, camera=camera)
                if os.path.exists(framefile):
                    os.remove(framefile)
                cframefile = desispec.io.findfile('cframe', self.night, expid, camera=camera)
                if os.path.exists(cframefile):
                    os.remove(cframefile)
                skyfile = desispec.io.findfile('sky', self.night, expid, camera=camera)
                if os.path.exists(skyfile):
                    os.remove(skyfile)
                fluxcalibfile = desispec.io.findfile('calib', self.night, expid, camera=camera)
                if os.path.exists(fluxcalibfile):
                    os.remove(fluxcalibfile)

    def test_parse(self):
        night = self.night
        expid = self.expid
        obs.new_exposure('dark', night=night, expid=expid, nspec=4)

        simspec = io.findfile('simspec', self.night, self.expid)
        fibermap = desispec.io.findfile('fibermap', self.night, self.expid)
        opts = ['--simspec', simspec, '--fibermap', fibermap]
        opts += ['--spectrograph', '3', '--config', 'desi']
        args = quickgen.parse(opts)
        self.assertEqual(args.simspec, simspec)
        self.assertEqual(args.fibermap, fibermap)
        self.assertEqual(args.spectrograph, 3)
        self.assertEqual(args.config, 'desi')

        with self.assertRaises(ValueError):
            quickgen.parse([])

    def test_expand_args(self):
        night = self.night
        expid = self.expid
        obs.new_exposure('arc', night=night, expid=expid, nspec=4)

        simspec = io.findfile('simspec', self.night, self.expid)
        fibermap = desispec.io.findfile('fibermap', self.night, self.expid)
        opts = ['--simspec', simspec, '--fibermap', fibermap]
        args = quickgen.parse(opts)
        self.assertEqual(args.simspec, simspec)
        self.assertEqual(args.fibermap, fibermap)

    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    #- Run basic test of quickgen and check its outputs using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_simspec(self):
        night = self.night
        expid = self.expid
        camera = 'r0'
        # flavors = ['flat','dark','gray','bright','bgs','mws','elg','lrg','qso','arc']
        flavors = ['arc', 'flat', 'dark', 'bright']
        for i in range(len(flavors)):
            flavor = flavors[i]
            obs.new_exposure(flavor, night=night, expid=expid, nspec=2)
    
            #- output to same directory as input
            os.environ['DESI_SPECTRO_REDUX'] = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'))
    
            #- run quickgen
            simspec = io.findfile('simspec', night, expid)
            fibermap = io.findfile('simfibermap', night, expid)
            
            self.assertTrue(os.path.exists(simspec))
            self.assertTrue(os.path.exists(fibermap))

            if flavor == 'flat':
                try:
                    cmd = "quickgen --simspec {} --fibermap {}".format(simspec,fibermap)
                    quickgen.main(quickgen.parse(cmd.split()[1:]))
                except SystemExit:
                    pass

                #- verify flat outputs
                fiberflatfile = desispec.io.findfile('fiberflat', night, expid, camera)
                self.assertTrue(os.path.exists(fiberflatfile))

            elif flavor == 'arc':
                try:
                    cmd = "quickgen --simspec {} --fibermap {}".format(simspec,fibermap)
                    quickgen.main(quickgen.parse(cmd.split()[1:]))
                except SystemExit:
                    pass

                #- verify arc outputs
                framefile = desispec.io.findfile('frame', night, expid, camera)
                self.assertTrue(os.path.exists(framefile))

            else:
                cmd = "quickgen --simspec {} --fibermap {}".format(simspec,fibermap)
                quickgen.main(quickgen.parse(cmd.split()[1:]))
    
                #- verify outputs
                framefile = desispec.io.findfile('frame', night, expid, camera)
                self.assertTrue(os.path.exists(framefile))
                cframefile = desispec.io.findfile('cframe', night, expid, camera)
                self.assertTrue(os.path.exists(cframefile))
                skyfile = desispec.io.findfile('sky', night, expid, camera)
                self.assertTrue(os.path.exists(skyfile))
                fluxcalibfile = desispec.io.findfile('calib', night, expid, camera)
                self.assertTrue(os.path.exists(fluxcalibfile))

            os.remove(simspec)
            os.remove(fibermap)

    #- Run basic test of quickgen and check its outputs for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_brick(self):
        brickname = 'test1'
        nspec = 3
        cmd = "quickgen --brickname {} --objtype ELG -n {} --outdir {}".format(brickname, nspec, self.testDir)
        args = quickgen.parse(cmd.split()[1:])
        results = quickgen.main(args)
        #- Do the bricks exist?
        for channel in ['z', 'r', 'b']:
            brickfile = '{}/brick-{}-{}.fits'.format(self.testDir, channel, brickname)
            self.assertTrue(os.path.exists(brickfile))

        #- Check one of them for correct dimensionality of contents
        brickfile = '{}/brick-b-{}.fits'.format(self.testDir, brickname)
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

    #- Test quickgen with a bunch of options for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_options_brick(self):
        brickname = 'test2'
        nspec = 7
        seed = np.random.randint(2**30)
        cmd = "quickbrick --brickname {} --objtype BGS -n {} --outdir {} --seed {}".format(brickname, nspec, self.testDir, seed)
        cmd = cmd + " --airmass 1.5 --verbose --zrange-bgs 0.1 0.2"
        cmd = cmd + " --moon-phase 0.1 --moon-angle 30 --moon-zenith 20"
        args = quickgen.parse(cmd.split()[1:])
        results = quickgen.main(args)
        brickfile = '{}/brick-b-{}.fits'.format(self.testDir, brickname)
        truth = fits.getdata(brickfile, '_TRUTH')
        z = truth['TRUEZ']
        self.assertTrue(np.all((0.1 <= z) & (z <= 0.2)))

        #- Basic empirical sanity check of median(error) vs. median(flux)
        #- These coefficients are taken from 50 quickbrick BGS targets
        #- The main point is to trigger if something significantly changed
        #- in a new version, not to validate that this is actually the right
        #- error vs. flux model.
        coeff = dict()
        coeff['b'] = [6.379e-08, -2.824e-05, 1.952e-02, 8.051e+00]
        coeff['r'] = [6.683e-08, -4.537e-05, 2.230e-02, 3.767e+00]
        coeff['z'] = [1.358e-07, -7.210e-05, 2.392e-02, 2.090e+00]
        for channel in ['b', 'r', 'z']:
            brickfile = '{}/brick-{}-{}.fits'.format(self.testDir, channel, brickname)
            brick = desispec.io.read_frame(brickfile)
            medflux = np.median(brick.flux, axis=1)
            mederr = np.median(1/np.sqrt(brick.ivar), axis=1)
            ii = (medflux<250)
            if np.count_nonzero(ii) < 3:
                print(cmd)
                print('Too few points for stddev; skipping comparison')
            else:
                std = np.std(mederr[ii] - np.polyval(coeff[channel], medflux[ii]))
                if std > 0.15:
                    print(cmd)
                    self.assertLess(std, 0.2, 'noise model failed for channel {} seed {}'.format(channel, seed))

    #- Ensure that using --seed results in reproducible spectra using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_seed_simspec(self):
        night=self.night
        camera = 'r0'
        expid0 = 100
        expid1 = 101
        expid2 = 102

        # generate exposures seed 1 & 2
        obs.new_exposure('dark',night=night,expid=expid0,nspec=1,seed=1)
        simspec0 = io.findfile('simspec', night, expid0)
        fibermap0 = desispec.io.findfile('fibermap', night, expid0)
        obs.new_exposure('dark',night=night,expid=expid1,nspec=1,seed=1)
        simspec1 = io.findfile('simspec', night, expid1)
        fibermap1 = desispec.io.findfile('fibermap', night, expid1)
        obs.new_exposure('dark',night=night,expid=expid2,nspec=1,seed=2)
        simspec2 = io.findfile('simspec', night, expid2)
        fibermap2 = desispec.io.findfile('fibermap', night, expid2)

        # generate quickgen output for each exposure
        cmd = "quickgen --simspec {} --fibermap {} --seed 1".format(simspec0,fibermap0)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {} --seed 1".format(simspec1,fibermap1)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {} --seed 2".format(simspec2,fibermap2)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        cframe0=desispec.io.findfile("cframe",night,expid0,camera)
        cframe1=desispec.io.findfile("cframe",night,expid1,camera)
        cframe2=desispec.io.findfile("cframe",night,expid2,camera)
        cf0=desispec.io.read_frame(cframe0)
        cf1=desispec.io.read_frame(cframe1)
        cf2=desispec.io.read_frame(cframe2)

        self.assertTrue(np.all(cf0.flux == cf1.flux))   #- same seed
        self.assertTrue(np.all(cf0.ivar == cf1.ivar))
        self.assertTrue(np.any(cf0.flux != cf2.flux))   #- different seed

        s0=fits.open(simspec0)
        s1=fits.open(simspec1)
        s2=fits.open(simspec2)

        self.assertEqual(s0['METADATA'].data['OBJTYPE'][0], s1['METADATA'].data['OBJTYPE'][0])
        self.assertEqual(s0['METADATA'].data['REDSHIFT'][0], s1['METADATA'].data['REDSHIFT'][0])
        self.assertNotEqual(s0['METADATA'].data['REDSHIFT'][0], s2['METADATA'].data['REDSHIFT'][0])

    #- Ensure that using --seed results in reproducible spectra for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_seed_brick(self):
        cmd = "quickgen --seed 1 --brickname test1a --objtype BRIGHT_MIX -n 1 --outdir {}".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --seed 1 --brickname test1b --objtype BRIGHT_MIX -n 1 --outdir {}".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --seed 2 -b test2 --objtype BRIGHT_MIX -n 1 --outdir {}".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        f1a = desispec.io.read_frame('{}/brick-b-test1a.fits'.format(self.testDir))
        f1b = desispec.io.read_frame('{}/brick-b-test1b.fits'.format(self.testDir))
        f2 = desispec.io.read_frame('{}/brick-b-test2.fits'.format(self.testDir))
        self.assertTrue(np.all(f1a.flux == f1b.flux))  #- same seed
        self.assertTrue(np.all(f1a.ivar == f1b.ivar))
        self.assertTrue(np.any(f1a.flux != f2.flux))   #- different seed

        t1a = fits.getdata('{}/brick-b-test1a.fits'.format(self.testDir), '_TRUTH')
        t1b = fits.getdata('{}/brick-b-test1b.fits'.format(self.testDir), '_TRUTH')
        t2 = fits.getdata('{}/brick-b-test2.fits'.format(self.testDir), '_TRUTH')
        self.assertTrue(np.all(t1a['TRUE_OBJTYPE'] == t1b['TRUE_OBJTYPE']))
        self.assertTrue(np.all(t1b['TRUEZ'] == t1b['TRUEZ']))
        self.assertTrue(np.any(t1a['TRUEZ'] != t2['TRUEZ']))

    #- Test that higher airmass makes noisier spectra using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_airmass_simspec(self):
        night=self.night
        camera = 'r0'
        expid0 = 100
        expid1 = 101

        # generate exposures of varying airmass
        obs.new_exposure('dark',night=night,expid=expid0,nspec=1,airmass=1.5,seed=1)
        simspec0 = io.findfile('simspec', night, expid0)
        fibermap0 = desispec.io.findfile('fibermap', night, expid0)
        obs.new_exposure('dark',night=night,expid=expid1,nspec=1,airmass=1.0,seed=1)
        simspec1 = io.findfile('simspec', night, expid1)
        fibermap1 = desispec.io.findfile('fibermap', night, expid1)

        # generate quickgen output for each airmass
        cmd = "quickgen --simspec {} --fibermap {}".format(simspec0,fibermap0)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {}".format(simspec1,fibermap1)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        cframe0=desispec.io.findfile("cframe",night,expid0,camera)
        cframe1=desispec.io.findfile("cframe",night,expid1,camera)
        cf0=desispec.io.read_frame(cframe0)
        cf1=desispec.io.read_frame(cframe1)
        self.assertLess(np.median(cf0.ivar),np.median(cf1.ivar))

    #- Test that higher airmass makes noisier spectra for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_airmass_brick(self):
        nspec = 2
        cmd = "quickgen --brickname test1 --airmass 1.5 --objtype DARK_MIX -n {} --outdir {}".format(nspec, self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --brickname test2 --airmass 1.0 --objtype DARK_MIX -n {} --outdir {}".format(nspec, self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        f1 = desispec.io.read_frame('{}/brick-b-test1.fits'.format(self.testDir))
        f2 = desispec.io.read_frame('{}/brick-b-test2.fits'.format(self.testDir))
        #- test1 has shorter exposure time thus larger errors thus smaller ivar
        self.assertLess(np.median(f1.ivar), np.median(f2.ivar))

    #- Test that shorter exposures make noisier spectra using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_exptime_simspec(self):
        night = self.night
        camera = 'r0'
        expid0 = 100
        expid1 = 101

        # generate exposures of varying exposure times
        obs.new_exposure('dark',exptime=100,night=night,expid=expid0,nspec=1,seed=1)
        simspec0 = io.findfile('simspec', night, expid0)
        fibermap0 = desispec.io.findfile('fibermap', night, expid0)
        obs.new_exposure('dark',exptime=1000,night=night,expid=expid1,nspec=1,seed=1)
        simspec1 = io.findfile('simspec', night, expid1)
        fibermap1 = desispec.io.findfile('fibermap', night, expid1)

        # generate quickgen output for each exposure time
        cmd = "quickgen --simspec {} --fibermap {}".format(simspec0,fibermap0)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {}".format(simspec1,fibermap1)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        cframe0=desispec.io.findfile("cframe",night,expid0,camera)
        cframe1=desispec.io.findfile("cframe",night,expid1,camera)
        cf0=desispec.io.read_frame(cframe0)
        cf1=desispec.io.read_frame(cframe1)
        self.assertLess(np.median(cf0.ivar),np.median(cf1.ivar))

    #- Test that shorter exposures make noisier spectra for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_exptime_brick(self):
        cmd = "quickgen --brickname test1 --exptime 100 --objtype DARK_MIX -n 1 --outdir {}".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --brickname test2 --exptime 1000 --objtype DARK_MIX -n 1 --outdir {}".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        f1 = desispec.io.read_frame('{}/brick-b-test1.fits'.format(self.testDir))
        f2 = desispec.io.read_frame('{}/brick-b-test2.fits'.format(self.testDir))
        #- test1 has shorter exposure time thus larger errors thus smaller ivar
        self.assertLess(np.median(f1.ivar), np.median(f2.ivar))

    #- Test that brighter moon makes noisier spectra using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_moonphase_simspec(self):
        night = self.night
        camera = 'r0'
        expid0 = 100
        expid1 = 101

        # generate exposures
        obs.new_exposure('bgs',night=night,expid=expid0,nspec=1,seed=1)
        simspec0 = io.findfile('simspec', night, expid0)
        fibermap0 = desispec.io.findfile('fibermap', night, expid0)
        obs.new_exposure('bgs',night=night,expid=expid1,nspec=1,seed=1)
        simspec1 = io.findfile('simspec', night, expid1)
        fibermap1 = desispec.io.findfile('fibermap', night, expid1)

        # generate quickgen output for each moon phase
        cmd = "quickgen --simspec {} --fibermap {} --moon-phase 0.0".format(simspec0,fibermap0)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {} --moon-phase 1.0".format(simspec1,fibermap1)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        cframe0=desispec.io.findfile("cframe",night,expid0,camera)
        cframe1=desispec.io.findfile("cframe",night,expid1,camera)
        cf0=desispec.io.read_frame(cframe0)
        cf1=desispec.io.read_frame(cframe1)
        self.assertLess(np.median(cf0.ivar),np.median(cf1.ivar))

    #- Test that brighter moon makes noisier spectra for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_moonphase_brick(self):
        cmd = "quickgen --brickname brightmoon --objtype BRIGHT_MIX -n 1 --outdir {} --moon-phase 0.0".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --brickname crescentmoon --objtype BRIGHT_MIX -n 1 --outdir {} --moon-phase 1.0".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        brick1 = desispec.io.read_frame('{}/brick-b-brightmoon.fits'.format(self.testDir))
        brick2 = desispec.io.read_frame('{}/brick-b-crescentmoon.fits'.format(self.testDir))
        #- brick1 has more moonlight thus larger errors thus smaller ivar
        self.assertLess(np.median(brick1.ivar), np.median(brick2.ivar))

    #- Test that smaller moon angle makes noisier spectra using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_moonangle_simspec(self):
        night = self.night
        camera = 'r0'
        expid0 = 100
        expid1 = 101

        # generate exposures
        obs.new_exposure('bgs',night=night,expid=expid0,nspec=1,seed=1)
        simspec0 = io.findfile('simspec', night, expid0)
        fibermap0 = desispec.io.findfile('fibermap', night, expid0)
        obs.new_exposure('bgs',night=night,expid=expid1,nspec=1,seed=1)
        simspec1 = io.findfile('simspec', night, expid1)
        fibermap1 = desispec.io.findfile('fibermap', night, expid1)

        # generate quickgen output for each moon phase
        cmd = "quickgen --simspec {} --fibermap {} --moon-angle 0".format(simspec0,fibermap0)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {} --moon-angle 180".format(simspec1,fibermap1)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        cframe0=desispec.io.findfile("cframe",night,expid0,camera)
        cframe1=desispec.io.findfile("cframe",night,expid1,camera)
        cf0=desispec.io.read_frame(cframe0)
        cf1=desispec.io.read_frame(cframe1)
        self.assertLess(np.median(cf0.ivar),np.median(cf1.ivar))

    #- Test that smaller moon angle makes noisier spectra for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_moonangle_brick(self):
        cmd = "quickgen --brickname angle0 --objtype BRIGHT_MIX -n 1 --outdir {} --moon-angle 0".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --brickname angle180 --objtype BRIGHT_MIX -n 1 --outdir {} --moon-angle 180".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        brick1 = desispec.io.read_frame('{}/brick-b-angle0.fits'.format(self.testDir))
        brick2 = desispec.io.read_frame('{}/brick-b-angle180.fits'.format(self.testDir))
        #- brick1 has more moonlight thus larger errors thus smaller ivar
        self.assertLess(np.median(brick1.ivar), np.median(brick2.ivar))

    #- Test that smaller moon zenith makes noisier spectra using simspec as input
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickgen/specsim test on Travis')
    def test_quickgen_moonzenith_simspec(self):
        night = self.night
        camera = 'r0'
        expid0 = 100
        expid1 = 101

        # generate exposures
        obs.new_exposure('bgs',night=night,expid=expid0,nspec=1,seed=1)
        simspec0 = io.findfile('simspec', night, expid0)
        fibermap0 = desispec.io.findfile('fibermap', night, expid0)
        obs.new_exposure('bgs',night=night,expid=expid1,nspec=1,seed=1)
        simspec1 = io.findfile('simspec', night, expid1)
        fibermap1 = desispec.io.findfile('fibermap', night, expid1)

        # generate quickgen output for each moon phase
        cmd = "quickgen --simspec {} --fibermap {} --moon-zenith 0".format(simspec0,fibermap0)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --simspec {} --fibermap {} --moon-zenith 90".format(simspec1,fibermap1)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        cframe0=desispec.io.findfile("cframe",night,expid0,camera)
        cframe1=desispec.io.findfile("cframe",night,expid1,camera)
        cf0=desispec.io.read_frame(cframe0)
        cf1=desispec.io.read_frame(cframe1)
        self.assertLess(np.median(cf0.ivar),np.median(cf1.ivar))

    #- Test that smaller moon zenith makes noisier spectra for bricks
    ### @unittest.skipIf('TRAVIS_JOB_ID' in os.environ, 'Skipping memory hungry quickbrick/specsim test on Travis')
    def test_quickgen_moonzenith_brick(self):
        cmd = "quickgen --brickname zenith0 --objtype BRIGHT_MIX -n 1 --outdir {} --moon-zenith 0".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))
        cmd = "quickgen --brickname zenith90 --objtype BRIGHT_MIX -n 1 --outdir {} --moon-zenith 90".format(self.testDir)
        quickgen.main(quickgen.parse(cmd.split()[1:]))

        brick1 = desispec.io.read_frame('{}/brick-b-zenith0.fits'.format(self.testDir))
        brick2 = desispec.io.read_frame('{}/brick-b-zenith90.fits'.format(self.testDir))
        #- brick1 has more moonlight thus larger errors thus smaller ivar
        self.assertLess(np.median(brick1.ivar), np.median(brick2.ivar))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
