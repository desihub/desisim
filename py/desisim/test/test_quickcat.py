import os
import numpy as np
import unittest
from   astropy.table import Table, Column
from   astropy.io import fits
from   desisim.quickcat import quickcat
from   desitarget.targetmask import desi_mask, bgs_mask, mws_mask
import desimodel.io

class TestQuickCat(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(50)
        cls.ntiles = 4
        tiles = desimodel.io.load_tiles()
        cls.tileids = tiles['TILEID'][0:cls.ntiles]
        cls.tilefiles = ['tile-{:05d}.fits'.format(i) for i in cls.tileids]
        cls.tilefiles_multiobs = ['multitile-{:05d}.fits'.format(i) for i in cls.tileids]

        cls.nspec = n = 5000
        targets = Table()
        targets['TARGETID']    = np.random.randint(0,2**60, size=n)
        targets['RA']          = np.random.uniform(0,360., size=n)
        targets['DEC']         = np.random.uniform(-90.,90., size=n)
        targets['DESI_TARGET'] = 2**np.random.randint(0,3,size=n)
        targets['BGS_TARGET']  = np.zeros(n, dtype=int)
        targets['MWS_TARGET']  = np.zeros(n, dtype=int)
        isLRG = (targets['DESI_TARGET'] & desi_mask.LRG) != 0
        isELG = (targets['DESI_TARGET'] & desi_mask.ELG) != 0
        isQSO = (targets['DESI_TARGET'] & desi_mask.QSO) != 0
        cls.targets = targets

        #- Make a few of them BGS and MWS
        iibright = np.random.choice(np.arange(n), size=6, replace=False)
        isBGS = iibright[0:3]
        isMWS = iibright[3:6]
        targets['DESI_TARGET'][isBGS] = desi_mask.BGS_ANY
        targets['DESI_TARGET'][isMWS] = desi_mask.MWS_ANY
        targets['BGS_TARGET'][isBGS] = bgs_mask.BGS_BRIGHT
        try:
            #- desitarget >= 0.25.0
            targets['MWS_TARGET'][isMWS] = mws_mask.MWS_BROAD
        except AttributeError:
            #- desitarget <= 0.24.0
            targets['MWS_TARGET'][isMWS] = mws_mask.MWS_MAIN

        #- Add some fake photometry; no attempt to get colors right
        #- https://portal.nersc.gov/project/desi/users/adamyers/0.31.1/desitargetQA-dr8-0.31.1/LRG.html
        flux = np.zeros((n, 6))  #- ugrizY; DESI has grz

        #- Assumed nanomaggies.
        flux[isLRG, 1] = np.random.uniform(19., 20., np.count_nonzero(isLRG))
        flux[isLRG, 2] = np.random.uniform(19., 20., np.count_nonzero(isLRG))
        flux[isLRG, 4] = np.random.uniform(19., 20., np.count_nonzero(isLRG))

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

        targets['FLUX_G'] = flux[:,1]
        targets['FLUX_R'] = flux[:,2]
        targets['FLUX_Z'] = flux[:,4]
        
        truth = Table()
        truth['TARGETID'] = targets['TARGETID'].copy()

        truth['FLUX_G']   = targets['FLUX_G'].copy()
        truth['FLUX_R']   = targets['FLUX_R'].copy()
        truth['FLUX_Z']   = targets['FLUX_Z'].copy()
        
        truth['TRUEZ'] = np.random.uniform(0, 1.5, size=n)
        truth['TRUESPECTYPE'] = np.zeros(n, dtype=(str, 10))

        truth['TEMPLATETYPE'] = np.zeros(n, dtype=(str, 10))
        truth['TEMPLATETYPE'][isLRG] = 'LRG'
        truth['TEMPLATETYPE'][isELG] = 'ELG'
        truth['TEMPLATETYPE'][isQSO] = 'QSO'
        truth['TEMPLATETYPE'][isBGS] = 'BGS'
        truth['TEMPLATETYPE'][isMWS] = 'MWS'
        
        truth['GMAG'] = np.random.uniform(18.0, 24.0, size=n)
        ii = (targets['DESI_TARGET'] & desi_mask.mask('LRG|ELG|BGS_ANY')) != 0
        truth['TRUESPECTYPE'][ii] = 'GALAXY'
        ii = (targets['DESI_TARGET'] == desi_mask.QSO)
        truth['TRUESPECTYPE'][ii] = 'QSO'
        starmask = desi_mask.mask('MWS_ANY|STD_FAINT|STD_WD|STD_BRIGHT') 
        ii = ((targets['DESI_TARGET'] & starmask) != 0) 
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
        fiberassign.meta['EXTNAME'] = 'FASSIGN'
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
        
        zcat1.sort(keys='TARGETID')
        nx = self.nspec // self.ntiles
        truth_01 = self.truth[0:2*nx].copy()
        truth_01.sort(keys='TARGETID')
        self.assertTrue(np.all(zcat1['TARGETID'] == truth_01['TARGETID']))
        self.assertTrue(np.all(zcat1['Z'] == truth_01['TRUEZ']))
        self.assertTrue(np.all(zcat1['ZWARN'] == 0))

        #- Now observe with random redshift errors
        zcat2 = quickcat(self.tilefiles[0:2], self.targets, truth=self.truth, perfect=False)
        zcat2_sorted = zcat2.copy()
        zcat2_sorted.sort(keys='TARGETID')
        self.assertTrue(np.all(zcat2_sorted['TARGETID'] == truth_01['TARGETID']))        
        self.assertTrue(np.all(zcat2_sorted['Z'] != truth_01['TRUEZ']))
        self.assertTrue(np.any(zcat2_sorted['ZWARN'] != 0))

        #- And add a second round of observations
        zcat3 = quickcat(self.tilefiles[2:4], self.targets, truth=self.truth, zcat=zcat2, perfect=False)
        zcat3_sorted = zcat3.copy()
        zcat3_sorted.sort(keys='TARGETID')
        truth_sorted = self.truth.copy()
        truth_sorted.sort(keys='TARGETID')
        self.assertTrue(np.all(zcat3_sorted['TARGETID'] == truth_sorted['TARGETID']))
        self.assertTrue(np.all(zcat3_sorted['Z'] != truth_sorted['TRUEZ']))
        
        #- successful targets in the first round of observations shouldn't be updated
        ii2 = np.in1d(zcat2_sorted['TARGETID'], zcat3_sorted['TARGETID']) & (zcat2_sorted['ZWARN'] == 0)
        ii3 = np.in1d(zcat3_sorted['TARGETID'], zcat2_sorted['TARGETID'][ii2])
        ii = zcat2_sorted['Z'][ii2] == zcat3_sorted['Z'][ii3]
        self.assertTrue(np.all(zcat2_sorted['Z'][ii2] == zcat3_sorted['Z'][ii3]))
        
        #- Observe the last tile again
        zcat3copy = zcat3_sorted.copy()
        zcat4 = quickcat(self.tilefiles[3:4], self.targets, truth=self.truth, zcat=zcat3copy)
        zcat4_sorted = zcat4.copy()
        zcat4_sorted.sort(keys='TARGETID')
        self.assertTrue(np.all(zcat3copy == zcat3_sorted))  #- original unmodified
        self.assertTrue(np.all(zcat4_sorted['TARGETID'] == truth_sorted['TARGETID'])) #- all IDS observed
        self.assertTrue(np.all(zcat4_sorted['Z'] != truth_sorted['TRUEZ']))

        #- Check that NUMOBS was incremented
        i3 = np.in1d(zcat3_sorted['TARGETID'], self.targets_in_tile[self.tileids[3]]) # ids observed in the last tile
        i4 = np.in1d(zcat4_sorted['TARGETID'], self.targets_in_tile[self.tileids[3]]) # ids observed in the last tile
        self.assertTrue(np.all(zcat4_sorted['NUMOBS'][i4] == zcat3_sorted['NUMOBS'][i3]+1))

        #- ZWARN==0 targets should be preserved, while ZWARN!=0 updated
        z3 = zcat3_sorted[i3]
        z4 = zcat4_sorted[i4]
        ii = (z3['ZWARN'] != 0)
        self.assertTrue(np.all(z3['Z'][~ii] == z4['Z'][~ii]))
        self.assertTrue(np.all(z3['Z'][ii] != z4['Z'][ii]))


    def test_multiobs(self):
        # Targets with more observations should have a better efficiency
        zcat = quickcat(self.tilefiles_multiobs, self.targets, truth=self.truth, perfect=False)
        
        oneobs = (zcat['NUMOBS'] == 1)
        manyobs = (zcat['NUMOBS'] == np.max(zcat['NUMOBS']))
        goodz = (zcat['ZWARN'] == 0)
        
        p1 = np.count_nonzero(oneobs & goodz) / np.count_nonzero(oneobs)
        p2 = np.count_nonzero(manyobs & goodz) / np.count_nonzero(manyobs)
        self.assertGreater(p2, p1)
                
if __name__ == '__main__':
    unittest.main()
