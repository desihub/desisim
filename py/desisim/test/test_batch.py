import os
import unittest

from desisim.batch import calc_nodes
from desisim.batch.pixsim import batch_newexp, batch_pixsim

class TestBlat(unittest.TestCase):
    
    def setUp(self):
        self.batchfile = 'batch-d4ae52ada252.sh'

    def tearDown(self):
        if os.path.exists(self.batchfile):
            os.remove(self.batchfile)
            
    def test_calc_nodes(self):
        self.assertEqual(calc_nodes(10, 1.5, 10), 4)
        self.assertEqual(calc_nodes(10, 1.5, 20), 4)
        self.assertEqual(calc_nodes(120, 1.5, 20), 10)
        self.assertEqual(calc_nodes(9, 10, 20), 5)
        self.assertEqual(calc_nodes(10, 10, 20), 5)
        self.assertEqual(calc_nodes(11, 10, 20), 6)
    
    def test_batch_newexp(self):
        flavors = ['arc', 'flat', 'bright', 'gray', 'dark']
        expids = range(len(flavors))
        night = '20101020'
        if os.path.exists(self.batchfile):
            os.remove(self.batchfile)
        batch_newexp(self.batchfile, flavors, nspec=5000, night=night,
            expids=expids, nodes=None)
        self.assertTrue(os.path.exists(self.batchfile))

        os.remove(self.batchfile)
        batch_newexp(self.batchfile, flavors, nspec=5000, night=night,
            expids=expids)
        self.assertTrue(os.path.exists(self.batchfile))

    def test_batch_pixsim(self):
        flavors = ['arc', 'flat', 'bright', 'gray', 'dark']
        expids = range(len(flavors))
        night = '20101020'
        if os.path.exists(self.batchfile):
            os.remove(self.batchfile)
        batch_pixsim(self.batchfile, flavors, nspec=5000, night=night,
            expids=expids, nodes=None)
        self.assertTrue(os.path.exists(self.batchfile))

        os.remove(self.batchfile)
        batch_pixsim(self.batchfile, flavors, nspec=5000, night=night,
            expids=expids)
        self.assertTrue(os.path.exists(self.batchfile))
                
if __name__ == '__main__':
    unittest.main()
