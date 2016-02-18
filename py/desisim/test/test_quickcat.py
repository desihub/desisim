import os
import numpy as np
import unittest
from astropy.table import Table, Column

from desisim.quickcat import quickcat

class TestQuickCat(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tilefiles = ['tile-{:05d}.fits'.format(i) for i in range(4)]
        
        truth = Table()
        truth['TARGETID'] = np.random.randint(0,2**60, size=20)
        truth['TYPE'] = np.random.randint(0,8, size=20)
        truth['Z'] = np.random.uniform(0, 1.5, size=20)
        cls.truth = truth
        
        fiberassign = truth['TARGETID',]
        fiberassign['RA'] = np.random.uniform(0,5, size=20)
        fiberassign['DEC'] = np.random.uniform(0,5, size=20)
        fiberassign.meta['EXTNAME'] = 'FIBER_ASSIGNMENTS'
        for i, filename in enumerate(cls.tilefiles):
            fiberassign[i*5:(i+1)*5].write(filename)
    
    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        for filename in cls.tilefiles:
            if os.path.exists(filename):
                os.remove(filename)
            
    def test_quickcat(self):
        #- First round of obs: perfect input z -> output z
        zcat1 = quickcat(self.tilefiles[0:2], truth=self.truth, perfect=True)
        self.assertTrue(np.all(zcat1['TARGETID'] == self.truth['TARGETID'][0:10]))
        self.assertTrue(np.all(zcat1['Z'] == self.truth['Z'][0:10]))
        self.assertTrue(np.all(zcat1['ZWARN'] == 0))
        
        #- Now observe with random redshift errors
        zcat2 = quickcat(self.tilefiles[2:4], truth=self.truth, zcat=zcat1)
        self.assertTrue(np.all(zcat2['TARGETID'] == self.truth['TARGETID']))
        self.assertTrue(np.all(zcat1['Z'] != self.truth['Z']))
                
if __name__ == '__main__':
    unittest.main()
