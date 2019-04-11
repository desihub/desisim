#!/usr/bin/env python
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import simqso.sqgrids  as grids
from simqso.lumfun import QlfEvolParam,PolyEvolParam,DoublePowerLawLF
from simqso.hiforest import IGMTransmissionGrid


##This script basically duplicates simqso_model.py in SIMQSO https://github.com/imcgreer/simqso/blob/master/simqso/sqmodels.py. It defines the emmision lines. This needs the fits table os.environ['DESISIM']+'/py/desisim/data/emlinetrends_Harris2016mod' based on Table 4 of https://iopscience.iop.org/article/10.3847/0004-6256/151/6/155/pdf. Only the Lya line does not correspond to such reference, but to the values originally defined in SIMQSO. Basically the only modified function is EmLineTemplate_modified. But needs all the others definitions to work.



WP11_model = {
 'forest0':{'zrange':(0.0,1.5),
            'logNHrange':(12.0,19.0),
            'gamma':0.2,
            'beta':1.55,
            'B':0.0170,
            'N0':340.,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest1':{'zrange':(1.5,4.6),
            'logNHrange':(12.0,14.5),
            'gamma':2.04,
            'beta':1.50,
            'B':0.0062,
            'N0':102.0,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest2':{'zrange':(1.5,4.6),
            'logNHrange':(14.5,17.5),
            'gamma':2.04,
            'beta':1.80,
            'B':0.0062,
            'N0':4.05,
            'brange':(10.,100.),
            'bsig':24.0},
 'forest3':{'zrange':(1.5,4.6),
            'logNHrange':(17.5,19.0),
            'gamma':2.04,
            'beta':0.90,
            'B':0.0062,
            'N0':0.051,
            'brange':(10.,100.),
            'bsig':24.0},
    'SLLS':{'zrange':(0.0,4.6),
            'logNHrange':(19.0,20.3),
            'N0':0.0660,
            'gamma':1.70,
            'beta':1.40,
            'brange':(10.,100.),
            'bsig':24.0},
     'DLA':{'zrange':(0.0,4.6),
            'logNHrange':(20.3,22.0),
            'N0':0.0440,
            'gamma':1.27,
            'beta':2.00,
            'brange':(10.,100.),
            'bsig':24.0},
}


forestModels = {
		'Worseck&Prochaska2011':WP11_model,
                }


BossDr9_fiducial_continuum = grids.BrokenPowerLawContinuumVar([
                                    grids.GaussianSampler(-1.50,0.3),
                                    grids.GaussianSampler(-0.50,0.3),
                                    grids.GaussianSampler(-0.37,0.3),
                                    grids.GaussianSampler(-1.70,0.3),
                                    grids.GaussianSampler(-1.03,0.3) ],
                                    [1100.,5700.,9730.,22300.])

BossDr9_expDust_cont = grids.BrokenPowerLawContinuumVar([
                                    grids.GaussianSampler(-0.50,0.2),
                                    grids.GaussianSampler(-0.30,0.2),
                                    grids.GaussianSampler(-0.37,0.3),
                                    grids.GaussianSampler(-1.70,0.3),
                                    grids.GaussianSampler(-1.03,0.3) ],
                                    [1100.,5700.,9730.,22300.])

BossDr9_FeScalings = [ (0,1540,0.5),(1540,1680,2.0),(1680,1868,1.6),
                       (1868,2140,1.0),(2140,3500,1.0) ]

def EmLineTemplate_modified(*args,**kwargs):
    kwargs.setdefault('scaleEWs',{
                                  'Lyepsdel':1,
                                  'CIII':0.3,
                                  'NII':1.1,
                                  'LyB/OIVn':0.7,
                                  'LyB/OIVb':1.1,
                                  'CIII*':1.8,
                                  'LyAb':1.1,'LyAn':1.1
                                  #Add more lines if needed.
                                  })
    kwargs['EmissionLineTrendFilename']=os.environ['DESISIM']+'/py/desisim/data/emlinetrends_Harris2016mod'
    return grids.generateBEffEmissionLines(*args,**kwargs)

def model_vars(qsoGrid,wave,nSightLines=0,
                           noforest=False,forestseed=None,verbose=0):

    if not noforest:
        if nSightLines <= 0:
            nSightLines = len(qsoGrid.z)
        subsample = nSightLines < len(qsoGrid.z)
        igmGrid = IGMTransmissionGrid(wave,
                                      forestModels['Worseck&Prochaska2011'],
                                      nSightLines,zmax=qsoGrid.z.max(),
                                      subsample=subsample,seed=forestseed,
                                      verbose=verbose)

    fetempl = grids.VW01FeTemplateGrid(qsoGrid.z,wave,
                                       scales=BossDr9_FeScalings)
    mvars = [ BossDr9_fiducial_continuum,
              EmLineTemplate_modified(qsoGrid.absMag),
              grids.FeTemplateVar(fetempl)]
    if not noforest:
        mvars.append( grids.HIAbsorptionVar(igmGrid,subsample=subsample) )
    return mvars


def model_PLEpivot(**kwargs):
    # the 0.3 makes the PLE and LEDE models align at the pivot redshift
    MStar1450_z0 = -22.92 + 1.486 + 0.3
    k1,k2 = 1.293,-0.268
    c1,c2 = -0.689, -0.809
    logPhiStar_z2_2 = -5.83
    MStar_i_z2_2 = -26.49
    MStar1450_z0_hiz = MStar_i_z2_2 + 1.486 # --> M1450
    logPhiStar = LogPhiStarPLEPivot([c1,logPhiStar_z2_2],z0=2.2,zpivot=2.2)
    MStar = MStarPLEPivot([-2.5*k2,-2.5*k1,MStar1450_z0,c2,MStar1450_z0_hiz],
                          zpivot=2.2,npar1=3,z0_1=0,z0_2=2.2)
    alpha = -1.3
    beta = -3.5
    return DoublePowerLawLF(logPhiStar,MStar,alpha,beta,**kwargs)

class LogPhiStarPLEPivot(PolyEvolParam):
    '''The PLE-Pivot model is PLE (fixed Phi*) below zpivot and
       LEDE (polynomial in log(Phi*) above zpivot.'''
    def __init__(self,*args,**kwargs):
        self.zp = kwargs.pop('zpivot')
        super(LogPhiStarPLEPivot,self).__init__(*args,**kwargs)
    def eval_at_z(self,z,par=None):
        # this fixes Phi* to be the zpivot value at z<zp
        z = np.asarray(z).clip(self.zp,np.inf)
        return super(LogPhiStarPLEPivot,self).eval_at_z(z,par)

class MStarPLEPivot(QlfEvolParam):
    '''The PLE-Pivot model for Mstar encapsulates two evolutionary models,
       one for z<zpivot and one for z>zp.'''
    def __init__(self,*args,**kwargs):
        self.zp = kwargs.pop('zpivot')
        self.n1 = kwargs.pop('npar1')
        self.z0_1 = kwargs.pop('z0_1',0.0)
        self.z0_2 = kwargs.pop('z0_2',0.0)
        super(MStarPLEPivot,self).__init__(*args,**kwargs)
    def eval_at_z(self,z,par=None):
        z = np.asarray(z)
        par = self._extract_par(par)
        return np.choose(z<self.zp,
                         [np.polyval(par[self.n1:],z-self.z0_2),
                          np.polyval(par[:self.n1],z-self.z0_1)])


