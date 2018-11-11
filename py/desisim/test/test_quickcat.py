import os
import numpy as np
import unittest
from astropy.table import Table, Column
from astropy.io import fits
from desisim.quickcat import quickcat
from desitarget.targetmask import desi_mask, bgs_mask, mws_mask
import desimodel.io

class TestQuickCat(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.ntiles = 4
        tiles = desimodel.io.load_tiles()
        cls.tileids = tiles['TILEID'][0:cls.ntiles]
        cls.tilefiles = ['tile-{:05d}.fits'.format(i) for i in cls.tileids]
        cls.tilefiles_multiobs = ['multitile-{:05d}.fits'.format(i) for i in cls.tileids]

        cls.nspec = n = 1000
        targets = Table()
        targets['TARGETID'] = np.random.randint(0,2**60, size=n)
        targets['DESI_TARGET'] = 2**np.random.randint(0,3,size=n)
        targets['BGS_TARGET'] = np.zeros(n, dtype=int)
        targets['MWS_TARGET'] = np.zeros(n, dtype=int)
        isLRG = (targets['DESI_TARGET'] & desi_mask.LRG) != 0
        isELG = (targets['DESI_TARGET'] & desi_mask.ELG) != 0
        isQSO = (targets['DESI_TARGET'] & desi_mask.QSO) != 0
        cls.targets = targets

        #- Make a few of them BGS and MWS
        iibright = np.random.choice(np.arange(n), size=6, replace=False)
        isBGS = iibright[0:3]
        isMWS = iibright[3:6]
        targets['DESI_TARGET'][isBGS] = desi_mask.BGS_ANY
        targets['BGS_TARGET'][isBGS] = bgs_mask.BGS_BRIGHT
        targets['DESI_TARGET'][isMWS] = desi_mask.MWS_ANY
        try:
            #- desitarget >= 0.25.0
            targets['MWS_TARGET'][isMWS] = mws_mask.MWS_BROAD
        except AttributeError:
            #- desitarget <= 0.24.0
            targets['MWS_TARGET'][isMWS] = mws_mask.MWS_MAIN

        #- Add some fake photometry; no attempt to get colors right
        flux = np.zeros((n, 6))  #- ugrizY; DESI has grz
        flux[isLRG, 1] = np.random.uniform(0, 1.0, np.count_nonzero(isLRG))
        flux[isLRG, 2] = np.random.uniform(0, 5.0, np.count_nonzero(isLRG))
        flux[isLRG, 4] = np.random.uniform(0, 5.0, np.count_nonzero(isLRG))
        flux[isELG, 1] = np.random.uniform(0, 4.0, np.count_nonzero(isELG))
        flux[isELG, 2] = np.random.uniform(0, 4.0, np.count_nonzero(isELG))
        flux[isELG, 4] = np.random.uniform(0, 10.0, np.count_nonzero(isELG))
        flux[isQSO, 1] = np.random.uniform(0, 4.0, np.count_nonzero(isQSO))
        flux[isQSO, 2] = np.random.uniform(0, 4.0, np.count_nonzero(isQSO))
        flux[isQSO, 4] = np.random.uniform(0, 6.0, np.count_nonzero(isQSO))
        # isBGS and isMWS are arrays of indices, not arrays of booleans
        flux[isBGS, 1] = np.random.uniform(10, 600, isBGS.size)
        flux[isBGS, 2] = np.random.uniform(15, 1000, isBGS.size)
        flux[isBGS, 4] = np.random.uniform(10, 1400, isBGS.size)
        flux[isMWS, 1] = np.random.uniform(10, 150, isMWS.size)
        flux[isMWS, 2] = np.random.uniform(15, 350, isMWS.size)
        flux[isMWS, 4] = np.random.uniform(10, 1500, isMWS.size)
        targets['DECAM_FLUX'] = flux

        truth = Table()
        truth['TARGETID'] = targets['TARGETID']
        truth['TRUEZ'] = np.random.uniform(0, 1.5, size=n)
        truth['TRUESPECTYPE'] = np.zeros(n, dtype=(str, 10))
        truth['GMAG'] = np.random.uniform(18.0, 24.0, size=n)
        ii = (targets['DESI_TARGET'] & desi_mask.mask('LRG|ELG|BGS_ANY')) != 0
        truth['TRUESPECTYPE'][ii] = 'GALAXY'
        ii = (targets['DESI_TARGET'] == desi_mask.QSO)
        truth['TRUESPECTYPE'][ii] = 'QSO'
        starmask = desi_mask.mask('MWS_ANY|STD_FAINT|STD_WD|STD_BRIGHT')
        ii = (targets['DESI_TARGET'] & starmask) != 0
        truth['TRUESPECTYPE'][ii] = 'STAR'

        #- Add some fake [OII] fluxes for the ELGs; include some that will fail
        isELG = (targets['DESI_TARGET'] & desi_mask.ELG) != 0
        nELG = np.count_nonzero(isELG)
        truth['OIIFLUX'] = np.zeros(n, dtype=float)
        truth['OIIFLUX'][isELG] = np.random.normal(2e-17, 2e-17, size=nELG).clip(0)

        cls.truth = truth

        fiberassign = truth['TARGETID',]
        fiberassign['RA'] = np.random.uniform(0,5, size=n)
        fiberassign['DEC'] = np.random.uniform(0,5, size=n)
        fiberassign.meta['EXTNAME'] = 'FIBERASSIGN'
        nx = cls.nspec // cls.ntiles
        cls.targets_in_tile = dict()
        for i, filename in enumerate(cls.tilefiles):
            subset = fiberassign[i*nx:(i+1)*nx]
            subset.write(filename)
            cls.targets_in_tile[cls.tileids[i]] = subset['TARGETID']
            hdulist = fits.open(filename, mode='update')
            hdr = hdulist[1].header
            hdr.set('TILEID', cls.tileids[i])
            hdulist.close()

        #- Also create a test of tile files that have multiple observations
        nx = cls.nspec // cls.ntiles
        for i, filename in enumerate(cls.tilefiles_multiobs):
            subset = fiberassign[0:(i+1)*nx]
            subset.write(filename)
            hdulist = fits.open(filename, mode='update')
            hdr = hdulist[1].header
            hdr.set('TILEID', cls.tileids[i])
            hdulist.close()

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        for filename in cls.tilefiles + cls.tilefiles_multiobs:
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
        zcat3 = quickcat(self.tilefiles[2:4], self.targets, truth=self.truth, zcat=zcat2)
        self.assertTrue(np.all(zcat3['TARGETID'] == self.truth['TARGETID']))
        self.assertTrue(np.all(zcat3['Z'] != self.truth['TRUEZ']))
        
        #- successful targets in the first round of observations shouldn't be updated
        ii2 = np.in1d(zcat2['TARGETID'], zcat3['TARGETID']) & (zcat2['ZWARN'] == 0)
        ii3 = np.in1d(zcat3['TARGETID'], zcat2['TARGETID'][ii2])
        self.assertTrue(np.all(zcat2['Z'][ii2] == zcat3['Z'][ii3]))
        
        #- Observe the last tile again
        zcat3copy = zcat3.copy()
        zcat4 = quickcat(self.tilefiles[3:4], self.targets, truth=self.truth, zcat=zcat3)
        self.assertTrue(np.all(zcat3copy == zcat3))  #- original unmodified
        self.assertTrue(np.all(zcat4['TARGETID'] == self.truth['TARGETID']))  #- order preserved
        self.assertTrue(np.all(zcat4['Z'] != self.truth['TRUEZ']))

        #- Check that NUMOBS was incremented
        i3 = np.in1d(zcat3['TARGETID'], self.targets_in_tile[self.tileids[3]])
        i4 = np.in1d(zcat4['TARGETID'], self.targets_in_tile[self.tileids[3]])
        self.assertTrue(np.all(zcat4['NUMOBS'][i4] == zcat3['NUMOBS'][i3]+1))

        #- ZWARN==0 targets should be preserved, while ZWARN!=0 updated
        nx = self.nspec // self.ntiles
        z3 = zcat3[-nx:]
        z4 = zcat4[-nx:]
        ii = (z3['ZWARN'] != 0)
        self.assertTrue(np.all(z3['Z'][ii] != z4['Z'][ii]))
        self.assertTrue(np.all(z3['Z'][~ii] == z4['Z'][~ii]))

    def test_multiobs(self):
        # Earlier targets got more observations so should have higher efficiency
        nx = self.nspec // self.ntiles
        zcat = quickcat(self.tilefiles_multiobs, self.targets, truth=self.truth, perfect=False)
        n1 = np.count_nonzero(zcat['ZWARN'][0:nx] == 0)
        n2 = np.count_nonzero(zcat['ZWARN'][-nx:] == 0)
        self.assertGreater(n1, n2)
        
                
if __name__ == '__main__':
    unittest.main()
