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
            fibermap, targets = desisim.targets.get_targets(n, flavor)
            fibermap, targets = desisim.targets.get_targets(n, flavor.lower())
            
    def test_sample_objtype(self):
        for flavor in ['DARK', 'GRAY', 'BRIGHT', 'LRG', 'ELG', 'QSO', 'BGS', 'MWS']:
            for n in [5,10,50]:
                for i in range(10):
                    truetype, targettype = desisim.targets.sample_objtype(n, flavor)

    def test_parallel(self):
        import multiprocessing as mp
        nproc = mp.cpu_count() // 2
        for n in [5, 23, 15*nproc+7]:
            fibermap, (flux, wave, meta) = desisim.targets.get_targets_parallel(n, 'DARK')
            self.assertEqual(n, len(fibermap))
            #- unique FIBER and TARGETID
            self.assertEqual(n, len(set(fibermap['FIBER'])))
            self.assertEqual(n, len(set(fibermap['TARGETID'])))
            self.assertTrue(np.all(fibermap['SPECTROID'] == fibermap['FIBER']//500))
            self.assertEqual(flux.shape[0], n)
            self.assertEqual(flux.shape[1], wave.shape[0])
            self.assertEqual(len(meta), n)

    def test_parallel_radec(self):
        '''Ensure that parallel generated ra,dec are unique'''
        nspec = 60
        fibermap, (flux, wave, meta) = desisim.targets.get_targets_parallel(nspec, 'SKY')
        nra = len(set(fibermap['RA_TARGET']))
        ndec = len(set(fibermap['DEC_TARGET']))
        self.assertEqual(nra, nspec)
        self.assertEqual(ndec, nspec)

    def test_random(self):
        for nspec in (5):
            fibermap1, (flux1, wave1, meta1) = desisim.targets.get_targets_parallel(nspec, 'DARK', seed=nspec+1)
            fibermap2a, (flux2a, wave2a, meta2a) = desisim.targets.get_targets_parallel(nspec, 'DARK', seed=nspec+2)
            fibermap2b, (flux2b, wave2b, meta2b) = desisim.targets.get_targets_parallel(nspec, 'DARK', seed=nspec+2)

            #- Check that 1 and 2a do not have the same spectra
            notsky = (fibermap1['OBJTYPE'] != 'SKY') & (fibermap2a['OBJTYPE'] != 'SKY')
            self.assertTrue(np.all(meta1['REDSHIFT'][notsky] != meta2a['REDSHIFT'][notsky]))
            self.assertTrue(np.all(fibermap1['TARGETID'] != fibermap2a['TARGETID']))

            #- Check 2a and 2b have the same spectra
            self.assertTrue(np.all(meta2a['REDSHIFT'][notsky] == meta2b['REDSHIFT'][notsky]))
            self.assertTrue(np.all(fibermap2a['TARGETID'] == fibermap2b['TARGETID']))
            self.assertTrue(np.all(meta2a['OIIFLUX'] == meta2b['OIIFLUX']))
            self.assertTrue(np.all(flux2a == flux2b))

            #- Check for duplicates
            for i in range(nspec-1):
                if np.any(flux1[i] != 0):  #- skip sky with flux=0
                    for j in range(i+1, nspec):
                        identical = np.all(flux1[i] == flux2a[j])
                        self.assertFalse(identical, 'Spectra {} and {} are identical'.format(i,j))

#- This runs all test* functions in any TestCase class in this file
if __name__ == '__main__':
    unittest.main()
