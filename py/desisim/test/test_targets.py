import unittest, os

import numpy as np

import desisim.targets

class TestObs(unittest.TestCase):

    def test_sample_nz(self):
        n = 5
        for objtype in ['LRG', 'ELG', 'QSO', 'STAR', 'STD']:
            z = desisim.targets.sample_nz(objtype, n)
            self.assertEqual(len(z), n)
    
    def test_get_targets(self):
        n = 5
        for flavor in ['DARK', 'BRIGHT', 'LRG', 'ELG', 'QSO', 'BGS', 'MWS']:
            fibermap, truth = desisim.targets.get_targets(n, flavor)
            fibermap, truth = desisim.targets.get_targets(n, flavor.lower())
            
    def test_sample_objtype(self):
        for flavor in ['DARK', 'GRAY', 'BRIGHT', 'LRG', 'ELG', 'QSO', 'BGS', 'MWS']:
            for n in [5,10,50]:
                for i in range(10):
                    truetype, targettype = desisim.targets.sample_objtype(n, flavor)

    def test_parallel(self):
        import multiprocessing as mp
        nproc = mp.cpu_count() // 2
        for n in [nproc, 5*nproc, 5*nproc+3]:
            fibermap, truth = desisim.targets.get_targets_parallel(n, 'DARK')
            self.assertEqual(n, len(fibermap))
            self.assertEqual(n, len(set(fibermap['FIBER'])))  #- unique FIBER
            self.assertTrue(np.all(fibermap['SPECTROID'] == fibermap['FIBER']//500))
            for key in truth.keys():
                if key not in ('UNITS', 'WAVE'):
                    self.assertEqual(n, truth[key].shape[0])

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
