import unittest, os, shutil, tempfile, subprocess
import numpy as np
from desisim.scripts import quickspectra
import desispec.io
from astropy.io import fits

class TestQuickSpectra(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origdir = os.getcwd()
        cls.testdir = tempfile.mkdtemp()
        os.chdir(cls.testdir)
        cls.inspec_txt = os.path.join(cls.testdir, 'inspec.txt')
        cls.inspec_fits = os.path.join(cls.testdir, 'inspec.fits')
        cls.outspec1 = os.path.join(cls.testdir, 'outspec1.fits')
        cls.outspec2 = os.path.join(cls.testdir, 'outspec2.fits')

        #- Create matching inputs in both text and FITS format
        wave = np.arange(3600, 9800, 1)
        flux = np.random.uniform(1,2, size=(2, len(wave)))
        with open(cls.inspec_txt, 'w') as fx:
            for i in range(len(wave)):
                fx.write('{} {} {}\n'.format(wave[i], flux[0,i], flux[1,i]))

        hx = fits.HDUList()
        hx.append(fits.PrimaryHDU(None))
        hx.append(fits.ImageHDU(wave, name='WAVELENGTH'))
        hx.append(fits.ImageHDU(flux, name='FLUX'))
        hx.writeto(cls.inspec_fits, overwrite=True)

    @classmethod
    def tearDownClass(cls):
        #- Remove all test input and output files
        os.chdir(cls.origdir)
        if os.path.exists(cls.testdir):
            shutil.rmtree(cls.testdir)

    def setUp(self):
        pass

    def tearDown(self):
        #- Remove output files but not input files
        for filename in [self.outspec1, self.outspec2]:
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
                self.assertFalse(np.allclose(sp1.ivar[x], sp2.ivar[x]))
            else:
                self.assertTrue(np.allclose(sp1.flux[x], sp2.flux[x]))
                self.assertTrue(np.allclose(sp1.ivar[x], sp2.ivar[x]))

    def test_quickspectra(self):
        cmd = 'quickspectra -i {} -o {} --seed 1'.format(self.inspec_txt, self.outspec1)
        opts = quickspectra.parse(cmd.split()[1:])
        quickspectra.main(opts)
        self.assertTrue(os.path.exists(self.outspec1))

        cmd = 'quickspectra -i {} -o {} --seed 1'.format(self.inspec_fits, self.outspec2)
        opts = quickspectra.parse(cmd.split()[1:])
        quickspectra.main(opts)
        self.assertTrue(os.path.exists(self.outspec2))

        sp1 = desispec.io.read_spectra(self.outspec1)
        sp2 = desispec.io.read_spectra(self.outspec2)
        self._check_spectra_match(sp1, sp2)

        #- Test skyerr option
        cmd = 'quickspectra -i {} -o {} --seed 1 --skyerr 0.1'.format(
            self.inspec_fits, self.outspec2)
        opts = quickspectra.parse(cmd.split()[1:])
        quickspectra.main(opts)
        self.assertTrue(os.path.exists(self.outspec2))
        sp2 = desispec.io.read_spectra(self.outspec2)
        self.assertGreater(np.std(sp2.flux['r']), np.std(sp1.flux['r']))

        #- Different seed should result in different spectra
        cmd = 'quickspectra -i {} -o {} --seed 2'.format(self.inspec_fits, self.outspec2)
        opts = quickspectra.parse(cmd.split()[1:])
        quickspectra.main(opts)
        self.assertTrue(os.path.exists(self.outspec2))
        sp2 = desispec.io.read_spectra(self.outspec2)
        self._check_spectra_match(sp1, sp2, invert=True)

        #- FITS output
        cmd = 'quickspectra -i {} -o {} --seed 1'.format(self.inspec_fits, self.outspec2)
        opts = quickspectra.parse(cmd.split()[1:])
        quickspectra.main(opts)
        self.assertTrue(os.path.exists(self.outspec2))
        sp2 = desispec.io.read_spectra(self.outspec2)
        self._check_spectra_match(sp1, sp2)

        #- Changing moon parameters should change spectra
        cmd = 'quickspectra -i {} -o {} --seed 1'.format(self.inspec_fits, self.outspec2)
        cmd += ' --moonfrac 0.9 --moonalt 80 --moonsep 10'
        opts = quickspectra.parse(cmd.split()[1:])
        quickspectra.main(opts)
        self.assertTrue(os.path.exists(self.outspec2))
        sp2 = desispec.io.read_spectra(self.outspec2)
        self._check_spectra_match(sp1, sp2, invert=True)

if __name__ == '__main__':
    unittest.main()

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m desisim.test.test_quickspectra
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
