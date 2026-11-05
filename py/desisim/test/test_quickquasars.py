from importlib import resources
import unittest, os, shutil, tempfile, subprocess
import numpy as np
from desisim.scripts import quickquasars
import desispec.io
from astropy.io import fits
from astropy.table import Table


class Testquickquasars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['DUST_DIR']=str(resources.files('desisim').joinpath('test', 'data'))
        cls.origdir = os.getcwd()
        cls.testdir =tempfile.mkdtemp()
        os.chdir(cls.testdir)
        # cls.infile  = str(resources.files('desisim').joinpath('test','data','transmission-16-1630.fits.gz'))   #London mock
        cls.infile  = str(resources.files('desisim').joinpath('test','data','Lv5-transmission-16-1531.fits'))  #London mock
        cls.infile2 = str(resources.files('desisim').joinpath('test','data','transmission-16-1747.fits.gz'))   #Saclay mock
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
            

    def _seed_catalog(self, infile, n=5):
        """Seed catalog (as made by gen_qso_catalog) with the first n quasars of a transmission file."""
        meta = Table.read(infile, hdu='METADATA')[:n]
        n = len(meta)
        catalog = Table()
        catalog['MOCKID'] = meta['MOCKID']
        catalog['FLUX_R'] = np.full(n, 10**((22.5-21.)/2.5))
        catalog['EXPTIME'] = np.full(n, 4000.)
        filename = os.path.join(self.testdir, 'seed_zcat_{}.fits'.format(os.path.basename(infile).split('.')[0]))
        catalog.write(filename, overwrite=True)
        return filename

    def test_mem_per_worker_validation(self):
        """--mem-per-worker must be a finite number larger than zero"""
        base = ['--infile', self.infile, '--outfile', self.outspec1]
        for bad in ['0', '-1', 'nan', 'inf']:
            with self.assertRaises(SystemExit, msg=bad):
                quickquasars.parse(base + ['--mem-per-worker', bad])
        args = quickquasars.parse(base + ['--mem-per-worker', '8'])
        self.assertEqual(args.mem_per_worker, 8.0)
        with self.assertRaises(ValueError):
            quickquasars._memory_limited_nproc(4, 0)

    def test_dla_and_metals_options(self):
        """DLAs (random, with and without the higher-order lines) and metals run together, and change the spectra."""
        base = 'quickquasars -i {} -o {} --from-catalog {} --overwrite --seed 1'.format(self.infile, self.outspec1, self._seed_catalog(self.infile))
        quickquasars.main(quickquasars.parse(base.split()[1:]))
        sp0 = desispec.io.read_spectra(self.outspec1)
        quickquasars.main(quickquasars.parse((base + ' --dla random --metals LYB LY3 LY4 LY5 SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --metal-strengths 0.1901 0.0697 0.0335 0.0187 1.8e-03 3.66e-03 1.83e-03 1.89e-03').split()[1:]))
        sp1 = desispec.io.read_spectra(self.outspec1)
        quickquasars.main(quickquasars.parse((base + ' --dla random --dlaplus --metals LYB LY3 LY4 LY5 SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --metal-strengths 0.1901 0.0697 0.0335 0.0187 1.8e-03 3.66e-03 1.83e-03 1.89e-03').split()[1:]))
        sp2 = desispec.io.read_spectra(self.outspec1)
        # same seed, so the noise is identical; the forest and the DLAs only affect the blue arm here
        self.assertFalse(np.allclose(sp0.flux['b'], sp1.flux['b']))     # DLAs and metals change the flux
        for x in ['b', 'r', 'z']:
            self.assertTrue(np.all(sp2.flux[x] - sp1.flux[x] <= 1e-6))     # the higher-order lines can only remove flux
        self.assertLess(np.min(sp2.flux['b'] - sp1.flux['b']), -1e-2)    # and they do remove some

    def test_no_lya_transmission(self):
        """--no-lya-transmission gives F=1 for the Lya forest; metals and DLAs can still be added on their own."""
        base = 'quickquasars -i {} -o {} --from-catalog {} --overwrite --seed 1'.format(self.infile, self.outspec1, self._seed_catalog(self.infile))
        metals = '--metals LYB LY3 LY4 LY5 SiII(1260) SiIII(1207) SiII(1193) SiII(1190) --metal-strengths 0.1901 0.0697 0.0335 0.0187 1.8e-03 3.66e-03 1.83e-03 1.89e-03'
        spectra = {}
        for name, extra in (('forest', ''), ('nolya', ' --no-lya-transmission'),
                            ('metals', ' --no-lya-transmission ' + metals), ('dla', ' --no-lya-transmission --dla random')):
            quickquasars.main(quickquasars.parse((base + extra).split()[1:]))
            spectra[name] = desispec.io.read_spectra(self.outspec1)
        self.assertFalse(np.allclose(spectra['forest'].flux['b'], spectra['nolya'].flux['b']))   # the Lya forest is gone
        self.assertFalse(np.allclose(spectra['nolya'].flux['b'], spectra['metals'].flux['b']))   # metal-only spectra
        self.assertFalse(np.allclose(spectra['nolya'].flux['b'], spectra['dla'].flux['b']))      # DLA-only spectra

    def test_specsim_config(self):
        """--specsim-config: the default is unchanged and a yaml pointing to other throughput files is used."""
        import re
        import specsim
        from desimodel.io import findfile
        cam_files = {x: findfile('throughput/thru-{}_y1measured.fits'.format(x)) for x in 'brz'}
        if not all(os.path.exists(f) for f in cam_files.values()):
            self.skipTest('the DESIMODEL throughput files are not available')
        # a copy of desi.yaml (the name must contain "desi") whose throughput files are scaled by 0.5
        yaml_text = open(os.path.join(os.path.dirname(specsim.__file__), 'data', 'config', 'desi.yaml')).read()
        for x, f in cam_files.items():
            tab = Table.read(f, hdu='THROUGHPUT')
            tab['throughput'] = 0.5 * tab['throughput']
            half = os.path.join(self.testdir, 'thru-{}_half.fits'.format(x))
            tab.write(half, overwrite=True, format='fits')
            fits.setval(half, 'EXTNAME', value='THROUGHPUT', ext=1)
            yaml_text = re.sub(r'path: throughput/thru-{}_y1measured.fits'.format(x), 'path: ' + half, yaml_text)
        yaml_file = os.path.join(self.testdir, 'desi_half_throughput.yaml')
        open(yaml_file, 'w').write(yaml_text)

        catalog = self._seed_catalog(self.infile)
        base = 'quickquasars -i {} -o {} --from-catalog {} --overwrite --seed 1'.format(self.infile, self.outspec1, catalog)
        spectra = {}
        for name, extra in (('default', ''), ('desi', ' --specsim-config desi'), ('half', ' --specsim-config ' + yaml_file)):
            quickquasars.main(quickquasars.parse((base + extra).split()[1:]))
            spectra[name] = desispec.io.read_spectra(self.outspec1)
        self._check_spectra_match(spectra['default'], spectra['desi'])
        ok = (spectra['default'].ivar['b'] > 0) & (spectra['half'].ivar['b'] > 0)
        self.assertLess(np.median(spectra['half'].ivar['b'][ok] / spectra['default'].ivar['b'][ok]), 0.8)    # a lower throughput gives a noisier spectrum

    def test_quickquasars(self):
        catalog = self._seed_catalog(self.infile)
        cmd = 'quickquasars -i {} -o {} --from-catalog {} --overwrite --seed 1'.format(self.infile, self.outspec1, catalog)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1))
        self.assertTrue(os.path.exists(self.outzbest))

        cmd = 'quickquasars -i {} -o {} --from-catalog {} --overwrite --seed 1 --save-continuum'.format(self.infile, self.outspec1, catalog)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1))

        catalog_s = self._seed_catalog(self.infile2)
        cmd = 'quickquasars -i {} -o {} --from-catalog {} --overwrite --seed 1'.format(self.infile2, self.outspec1_s, catalog_s)
        opts = quickquasars.parse(cmd.split()[1:])
        quickquasars.main(opts)
        self.assertTrue(os.path.exists(self.outspec1_s))
        self.assertTrue(os.path.exists(self.outzbest_s))

if __name__ == '__main__':
    unittest.main()
