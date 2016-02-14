import unittest, os

import numpy as np

import desisim.targets

desimodel_data_available = 'DESIMODEL' in os.environ

class TestObs(unittest.TestCase):

    @unittest.skipUnless(desimodel_data_available, 'The desimodel data/ directory was not detected.')
    def test_sample_nz(self):
        for objtype in ['LRG', 'ELG', 'QSO', 'STAR', 'STD']:
            n = desisim.targets.sample_nz(objtype, 5)

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
