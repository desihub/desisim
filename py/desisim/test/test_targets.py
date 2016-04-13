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
        for n in [5, 23, 15*nproc+7]:
            fibermap, truth = desisim.targets.get_targets_parallel(n, 'DARK')
            self.assertEqual(n, len(fibermap))
            #- unique FIBER and TARGETID
            self.assertEqual(n, len(set(fibermap['FIBER'])))
            self.assertEqual(n, len(set(fibermap['TARGETID'])))
            self.assertTrue(np.all(fibermap['SPECTROID'] == fibermap['FIBER']//500))
            for key in truth.keys():
                if key not in ('UNITS', 'WAVE'):
                    self.assertEqual(n, truth[key].shape[0])

    def test_random(self):
        for nspec in (15, 30):
            fibermap1, truth1 = desisim.targets.get_targets_parallel(nspec, 'DARK', seed=nspec+1)
            fibermap2a, truth2a = desisim.targets.get_targets_parallel(nspec, 'DARK', seed=nspec+2)
            fibermap2b, truth2b = desisim.targets.get_targets_parallel(nspec, 'DARK', seed=nspec+2)

            #- Check that 1 and 2a do not have the same spectra
            notsky = (fibermap1['OBJTYPE'] != 'SKY') & (fibermap2a['OBJTYPE'] != 'SKY')
            self.assertTrue(np.all(truth1['REDSHIFT'][notsky] != truth2a['REDSHIFT'][notsky]))
            self.assertTrue(np.all(fibermap1['TARGETID'] != fibermap2a['TARGETID']))

            #- Check 2a and 2b have the same spectra
            self.assertTrue(np.all(truth2a['REDSHIFT'][notsky] == truth2b['REDSHIFT'][notsky]))
            self.assertTrue(np.all(fibermap2a['TARGETID'] == fibermap2b['TARGETID']))
            self.assertTrue(np.all(truth2a['OIIFLUX'] == truth2b['OIIFLUX']))
            self.assertTrue(np.all(truth2a['FLUX'] == truth2b['FLUX']))

            #- Check for duplicates
            for i in range(nspec-1):
                if truth1['OBJTYPE'][i] != 'SKY':
                    for j in range(i+1, nspec):
                        self.assertFalse(np.all(truth1['FLUX'][i] == truth1['FLUX'][j]),
                            'Spectra {} and {} are identical'.format(i,j))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
