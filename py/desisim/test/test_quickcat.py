import os
import numpy as np
import unittest
from astropy.table import Table, Column
from astropy.io import fits
from desisim.quickcat import quickcat
from desitarget import desi_mask, bgs_mask, mws_mask

class TestQuickCat(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.ntiles = 4
        cls.tilefiles = ['tile-{:05d}.fits'.format(i) for i in range(cls.ntiles)]

        cls.nspec = n = 100
        targets = Table()
        targets['TARGETID'] = np.random.randint(0,2**60, size=n)
        targets['DESI_TARGET'] = 2**np.random.randint(0,3,size=n)
        targets['BGS_TARGET'] = np.zeros(n, dtype=int)
        targets['MWS_TARGET'] = np.zeros(n, dtype=int)
        cls.targets = targets

        #- Make a few of them BGS and MWS
        isBGS = np.random.randint(n, size=3)
        isMWS = np.random.randint(n, size=3)
        targets['DESI_TARGET'][isBGS] = desi_mask.BGS_ANY
        targets['BGS_TARGET'][isBGS] = bgs_mask.BGS_BRIGHT
        targets['DESI_TARGET'][isMWS] = desi_mask.MWS_ANY
        targets['MWS_TARGET'][isMWS] = mws_mask.MWS_MAIN

        truth = Table()
        truth['TARGETID'] = targets['TARGETID']
        truth['TRUEZ'] = np.random.uniform(0, 1.5, size=n)
        truth['TRUETYPE'] = np.zeros(n, dtype=(str, 10))
        truth['GMAG'] = np.random.uniform(18.0, 24.0, size=n)
        ii = (targets['DESI_TARGET'] & desi_mask.mask('LRG|ELG|BGS_ANY')) != 0
        truth['TRUETYPE'][ii] = 'GALAXY'
        ii = (targets['DESI_TARGET'] == desi_mask.QSO)
        truth['TRUETYPE'][ii] = 'QSO'
        starmask = desi_mask.mask('MWS_ANY|STD_FSTAR|STD_WD|STD_BRIGHT')
        ii = (targets['DESI_TARGET'] & starmask) != 0
        truth['TRUETYPE'][ii] = 'STAR'


        #- Add some fake [OII] fluxes for the ELGS
        isELG = (targets['DESI_TARGET'] & desi_mask.ELG) != 0
        nELG = np.count_nonzero(isELG)
        truth['OIIFLUX'] = np.zeros(n, dtype=float)
        truth['OIIFLUX'][isELG] = np.random.normal(8e-17, 2e-17, size=nELG).clip(0)

        cls.truth = truth

        fiberassign = truth['TARGETID',]
        fiberassign['RA'] = np.random.uniform(0,5, size=n)
        fiberassign['DEC'] = np.random.uniform(0,5, size=n)
        fiberassign.meta['EXTNAME'] = 'FIBER_ASSIGNMENTS'
        nx = cls.nspec // cls.ntiles
        for i, filename in enumerate(cls.tilefiles):
            fiberassign[i*nx:(i+1)*nx].write(filename)
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

        nx = self.nspec // self.ntiles
        self.assertTrue(np.all(zcat1['TARGETID'] == self.truth['TARGETID'][0:2*nx]))
        self.assertTrue(np.all(zcat1['Z'] == self.truth['TRUEZ'][0:2*nx]))
        self.assertTrue(np.all(zcat1['ZWARN'] == 0))

        #- Now observe with random redshift errors
        zcat2 = quickcat(self.tilefiles[0:2], self.targets, truth=self.truth, perfect=False)
        self.assertTrue(np.all(zcat2['TARGETID'] == self.truth['TARGETID'][0:2*nx]))
        self.assertTrue(np.all(zcat2['Z'] != self.truth['TRUEZ'][0:2*nx]))
        self.assertTrue(np.any(zcat2['ZWARN'] != 0))

        #- And add a second round of observations
        zcat3 = quickcat(self.tilefiles[2:4], self.targets, truth=self.truth, zcat=zcat1)
        self.assertTrue(np.all(zcat3['TARGETID'] == self.truth['TARGETID']))
        self.assertTrue(np.all(zcat3['Z'] != self.truth['TRUEZ']))

                
if __name__ == '__main__':
    unittest.main()
