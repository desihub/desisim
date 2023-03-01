"""
desisim.survey_release
=============

Functions and methods for mimicking a DESI release footprint and object density with adequate exposure times.

"""
from __future__ import division, print_function
from pkg_resources import resource_filename
import os
import desisim
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table, Column


class survey_release:
    """Class to store useful functions to reproduce a DESI release.
    In terms of footprint, object density and exposures.
    
    Args:
        release (str): DESI's release to be reproduced.
        pixel (int): Input HPXPIXEL to get density and number of observations from.
        nside (int): NSIDE for the sky.
        hpxnest (int): NEST for the sky.
    """
    def __init__(self,release,pixel,nside,hpxnest):
        survey_releases_file=os.path.join(os.path.dirname(desisim.__file__),'data/releases_info.fits')
        self.fname = survey_releases_file
        self.release = release
        self.nside = nside
        self.pixel = pixel
        self.hpxnest = hpxnest
        self.pixels,self.dens,self.numobs_prob = self.get_release_info()
        print('got data from file')
    
    def get_release_info(self):
        print(f'Reading {self.release} information from file {self.fname}')
        hdul = fits.open(self.fname)
        if self.release.upper() not in hdul:
            raise ValueError(f'{self.release} not in an HDU of {self.fname}')
        data = Table(hdul[self.release.upper()].data)
        hdr = hdul[self.release.upper()].header
        if self.nside != hdr['NSIDE']:
            raise ValueError(f'nside from file {self.fname} does not match nside from transmission file')
        if self.hpxnest != hdr['NEST']:
            raise ValueError(f'NEST option from file {self.fname} does not match NEST from transmission file')
            
        pixels = data['HPXPIXEL']
        density = data['MIDZ_DENS','HIGHZ_DENS']
        prob_numobs = data['PROB_NUMOBS']
        return pixels,density,prob_numobs
    
    def density(self):
        return self.dens[self.pixels==self.pixel]
    
    def numobs_probability(self):
        return self.numobs_prob[self.pixels==self.pixel][0]

def reproduce_release(Z,survey_release):
    """ Selects QSOs and assigns exposure times to reproduce a release.
    Args:
        Z (ndarray): Redshift array of objects.
        survey_release: Class object created by survey_release. 
    Returns:
        selection (ndarray): mask to apply to downsample input list
        exptime (ndarray): array of exposure time for each quasar.
    """
    N=len(Z)
    nside = survey_release.nside
    density = survey_release.density()
    
    pixarea = hp.pixelfunc.nside2pixarea(nside, degrees=True)
    selection = np.array([],dtype=int)
    index={'MIDZ':np.where((Z>=1.8)&(Z<2.1))[0],'HIGHZ':np.where(Z>=2.1)[0]}
    if len(density)!=0:
        for whichz in ['MIDZ','HIGHZ']:
            thisN=np.round(density[f'{whichz}_DENS']*pixarea).astype(int)
            if len(index[whichz])==0 or thisN==0:
                continue   
            if len(index[whichz])<thisN:
                thisN=len(index[whichz])
                print(f'Not enough {whichz} quasars in metadata, taking {thisN} available')

            whichIndices=np.random.choice(index[whichz],size=thisN,replace=False)
            print(f'{len(whichIndices)} {whichz} quasars selected out of {len(index[whichz])}')
            selection = np.concatenate((selection,whichIndices))
              
    exptime = np.full(len(Z[selection]),1000)
    if selection.size>0:
        if np.count_nonzero(Z[selection]>2.1)!=0:
            prob_numobs = survey_release.numobs_probability()
            numobs = np.arange(1,len(prob_numobs)+1)
            N_highz= np.count_nonzero(Z[selection]>2.1)
            print(f'Assigning {numobs} exposures with probabilities {prob_numobs} to {N_highz} Z>2.1 qsos')
            exp_highz = 1000*np.random.choice(numobs,size=N_highz,p=prob_numobs)
            exptime[Z[selection]>2.1]=exp_highz
    return selection, exptime
