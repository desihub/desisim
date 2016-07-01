"""
Run unit tests through quickgen
"""
import unittest, os
from uuid import uuid1
from shutil import rmtree

import desispec.io
from desisim import io
from desisim import obs
import desisim.scripts.quickgen
from desispec.log import get_logger
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
            'DESI_SPECTRO_SIM', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'PRODNAME', 'DESIMODEL'):
            if name not in os.environ:
                log.warning("missing ${0}".format(name))
                missing_env = True
    
        if missing_env:
            log.warning("Why are these needed?")
            log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
            log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
            log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$PRODNAME/")
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
        fibermap = desispec.io.findfile('fibermap', self.night, self.expid)
        if os.path.exists(fibermap):
            os.remove(fibermap)
        simspecfile = io.findfile('simspec', self.night, self.expid)
        if os.path.exists(simspecfile):
            os.remove(simspecfile)
        for camera in ('b0', 'r0', 'z0'):
            framefile = desispec.io.findfile('frame', self.night, self.expid, camera=camera)
            if os.path.exists(framefile):
                os.remove(framefile)
            cframefile = desispec.io.findfile('cframe', self.night, self.expid, camera=camera)
            if os.path.exists(cframefile):
                os.remove(cframefile)
            skyfile = desispec.io.findfile('sky', self.night, self.expid, camera=camera)
            if os.path.exists(skyfile):
                os.remove(skyfile)
            fluxcalibfile = desispec.io.findfile('calib', self.night, self.expid, camera=camera)
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
        args = desisim.scripts.quickgen.parse(opts)
        self.assertEqual(args.simspec, simspec)
        self.assertEqual(args.fibermap, fibermap)
        self.assertEqual(args.spectrograph, 3)
        self.assertEqual(args.config, 'desi')

        with self.assertRaises(ValueError):
            desisim.scripts.quickgen.parse([])

    def test_expand_args(self):
        night = self.night
        expid = self.expid
        obs.new_exposure('arc', night=night, expid=expid, nspec=4)

        simspec = io.findfile('simspec', self.night, self.expid)
        fibermap = desispec.io.findfile('fibermap', self.night, self.expid)
        opts = ['--simspec', simspec, '--fibermap', fibermap]
        args = desisim.scripts.quickgen.parse(opts)
        self.assertEqual(args.simspec, simspec)
        self.assertEqual(args.fibermap, fibermap)

    @unittest.skipUnless(desi_root_available, '$DESI_ROOT not set')
    def test_quickgen(self):
        night = self.night
        expid = self.expid
        camera = 'r0'
        flavors = ['flat','dark','gray','bright','bgs','mws','elg','lrg','qso']#,'arc']
        for i in range(len(flavors)):
            flavor = flavors[i]
            obs.new_exposure(flavor, night=night, expid=expid, nspec=4)
    
            #- output to same directory as input
            os.environ['DESI_SPECTRO_REDUX'] = os.path.join(os.getenv('DESI_SPECTRO_SIM'), os.getenv('PIXPROD'))
    
            #- run quickgen
            simspec = io.findfile('simspec', self.night, self.expid)
            fibermap = desispec.io.findfile('fibermap', self.night, self.expid)
            opts = ['--simspec', simspec, '--fibermap', fibermap]
            log.debug('testing quickgen({})'.format(opts))

            if flavor == 'flat':
                try:
                    desisim.scripts.quickgen.main(opts)
                except SystemExit:
                    pass

                #- verify flat outputs
                fiberflatfile = desispec.io.findfile('fiberflat', night, expid, camera)
                self.assertTrue(os.path.exists(fiberflatfile))

                #- cleanup flat outputs
                os.remove(fiberflatfile)

            elif flavor == 'arc':
                try:
                    desisim.scripts.quickgen.main(opts)
                except SystemExit:
                    pass

                #- verify arc outputs
                framefile = desispec.io.findfile('frame', night, expid, camera)
                self.assertTrue(os.path.exists(framefile))

                #- cleanup arc outputs
                os.remove(framefile)

            else:
                desisim.scripts.quickgen.main(opts)
    
                #- verify outputs
                framefile = desispec.io.findfile('frame', night, expid, camera)
                self.assertTrue(os.path.exists(framefile))
                cframefile = desispec.io.findfile('cframe', night, expid, camera)
                self.assertTrue(os.path.exists(cframefile))
                skyfile = desispec.io.findfile('sky', night, expid, camera)
                self.assertTrue(os.path.exists(skyfile))
                fluxcalibfile = desispec.io.findfile('calib', night, expid, camera)
                self.assertTrue(os.path.exists(fluxcalibfile))
    
                #- cleanup outputs
                os.remove(framefile)
                os.remove(cframefile)
                os.remove(skyfile)
                os.remove(fluxcalibfile)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
