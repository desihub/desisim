import os
import numpy as np
import unittest
from astropy.table import Table, Column
from astropy.io import fits
from desisim.quickcat import quickcat

class TestQuickCat(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tilefiles = ['tile-{:05d}.fits'.format(i) for i in range(4)]
        
        n = 20
        targets = Table()
        targets['DESI_TARGET'] = 2**np.random.randint(0,3,size=n)
        cls.targets = targets
        
        truth = Table()
        truth['TARGETID'] = np.random.randint(0,2**60, size=n)
        truth['TRUEZ'] = np.random.uniform(0, 1.5, size=n)
        truth['TRUETYPE'] = np.zeros(n, dtype=(str, 10))
        truth['GMAG'] = np.random.uniform(18.0, 24.0, size=n)
        ii = (targets['DESI_TARGET'] == 1); truth['TRUETYPE'][ii] = 'GALAXY'
        ii = (targets['DESI_TARGET'] == 2); truth['TRUETYPE'][ii] = 'GALAXY'
        ii = (targets['DESI_TARGET'] == 4); truth['TRUETYPE'][ii] = 'QSO'
        cls.truth = truth
        
        fiberassign = truth['TARGETID',]
        fiberassign['RA'] = np.random.uniform(0,5, size=n)
        fiberassign['DEC'] = np.random.uniform(0,5, size=n)
        fiberassign.meta['EXTNAME'] = 'FIBER_ASSIGNMENTS'
        for i, filename in enumerate(cls.tilefiles):
            fiberassign[i*5:(i+1)*5].write(filename)
            hdulist = fits.open(filename, mode='update')
            hdr = hdulist[1].header
            hdr.set('TILEID', i)
            hdulist.close()

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        for filename in cls.tilefiles:
            if os.path.exists(filename):
                os.remove(filename)
            
    def test_quickcat(self):
        #- First round of obs: perfect input z -> output z
        zcat1 = quickcat(self.tilefiles[0:2], self.targets, truth=self.truth, perfect=True)
        self.assertTrue(np.all(zcat1['TARGETID'] == self.truth['TARGETID'][0:10]))
        self.assertTrue(np.all(zcat1['Z'] == self.truth['TRUEZ'][0:10]))
        self.assertTrue(np.all(zcat1['ZWARN'] == 0))
        
        #- Now observe with random redshift errors
        zcat2 = quickcat(self.tilefiles[2:4], self.targets, truth=self.truth, zcat=zcat1)
        self.assertTrue(np.all(zcat2['TARGETID'] == self.truth['TARGETID']))
        self.assertTrue(np.all(zcat1['Z'] != self.truth['TRUEZ']))
                
if __name__ == '__main__':
    unittest.main()
