from pkg_resources import resource_filename
import unittest, os, shutil, tempfile, subprocess
import numpy as np
from desisim.scripts import quickquasars
import desispec.io
from astropy.io import fits


class Testquickquasars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['DUST_DIR']=resource_filename('desisim','test/data/')
        cls.origdir = os.getcwd()
        cls.testdir =tempfile.mkdtemp()
        os.chdir(cls.testdir)
        cls.infile = resource_filename('desisim','test/data/Lv5-transmission-16-1531.fits')       #London mock
        cls.infile2 = resource_filename('desisim','test/data/transmission-16-1747.fits.gz')    #Saclay mock
        cls.outspec1 = os.path.join(cls.testdir, 'spectra-16-1531.fits')
        cls.outzbest = os.path.join(cls.testdir, 'zbest-16-1531.fits')
        cls.outspec1_s=os.path.join(cls.testdir, 'spectra-16-1747.fits')
        cls.outzbest_s = os.path.join(cls.testdir, 'zbest-16-1747.fits')

    @classmethod
    def setUp(self):
        pass

    def tearDown(self):
        #- Remove output files but not input files
        for filename in [self.outspec1, self.outspec1_s,self.outzbest,self.outzbest_s]:
            if os.path.exists(filename):
                os.remove(filename)

    def _check_spectra_match(self, sp1, sp2, invert=False):
        '''
        Check if two spectra objects match.

        If invert=True, check that wavelengths match but not flux.
        '''
        for x in ['b', 'r', 'z']:
            self.assertTrue(np.allclose(sp1.wave[x], sp2.wave[x]))
            if invert:
                self.assertFalse(np.allclose(sp1.flux[x], sp2.flux[x]))
            else:
                self.assertTrue(np.allclose(sp1.flux[x], sp2.flux[x]))
                self.assertTrue(np.allclose(sp1.ivar[x], sp2.ivar[x]))
            

    def test_quickquasars(self):
        cmd = 'quickquasars -i {} -o {} --exptime 4000 --nmax 5 --overwrite --seed 1'.format(self.infile, self.outspec1)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1))
      

        cmd = 'quickquasars -i {} -o {} --exptime 4000 --bbflux --nmax 5 --overwrite --seed 1 --extintion-Rv '.format(self.infile, self.outspec1)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1))

        cmd = 'quickquasars -i {} -o {} --exptime 4000 --zbest --nmax 5 --overwrite --seed 1'.format(self.infile, self.outspec1)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1))
        self.assertTrue(os.path.exists(self.outzbest))
   

        cmd = 'quickquasars -i {} -o {} --exptime 4000 --zbest --nmax 5 --overwrite --seed 1'.format(self.infile2, self.outspec1_s)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1_s))
        self.assertTrue(os.path.exists(self.outzbest_s))



if __name__ == '__main__':
    unittest.main()

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desisim.test.test_quickquasars
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
