"""
desisim.bal
===========

Functions and methods for inserting BALs into QSO spectra.

"""
from __future__ import division, print_function

import numpy as np
from astropy.io import fits 
import random 

def readbaltemplates(file):
    """ Read in the BAL templates fits file
    Args: 
	file: name of fits file with BAL templates
    Returns: 
	wave_bal_template (ndarray): 1-D wavelength array in Ang for 
	  all templates in rest frame 
	bal_templates (ndarray): 2-d array with all BAL templates 
    """ 

    hdu = fits.open(file)
    bal_templates = hdu[1].data['TEMP'] 
    nlam = len(hdu[1].data['TEMP'][0])	# number of wavelength elements
    lam1 = hdu[1].header['CRVAL1']
    dlam = hdu[1].header['CDELT1']
    lam2 = lam1 + dlam*nlam
    wave_bal_template = np.arange(lam1, lam2, dlam)
    return wave_bal_template, bal_templates 


def isbal(balprob): 
    """ Randomly determine if a QSO is a BAL 
    Args: 
	balprob (float) : probability that a QSO is a BAL 
	Expect about 12% based on DR12
    Returns: 
	booleal: True if a BAL, otherwise False
    """ 

    # Make sure probability is on the interval [0.,1.]
    if balprob > 1.:
        balprob = 1. 
    if balprob < 0.: 
        balprob = 0.

    if np.random.random() < balprob:
        return True
    return False


def getbaltemplate(qsowave, qsoredshift, balwave, baltemplates):
    """ Get BAL template for a single QSO. 
    Note that the BAL template only covers the SiIV and CIV region
    The first step of this routine copies the CIV region to NV

    Args: 
	qsowave (ndarray): array of observed frame wavelength values
	qsoflux (ndarray) : array of observed frame flux values
	qsoredshift (float): QSO redshift
	balwave (ndarray): array of rest frame wavelength values
	baltemplates (ndarray): 2-D array of rest frame template values
	balwave and baltemplate are in the rest frame
    Returns: 
	template_out (ndarray): BAL template in observed frame
	template_id (int): index of BAL template
    """

    lambda_civ = 1549.48   # Ang
    lambda_nv = 1420.36    # Ang
    min_velocity = -25000. # km/s -- adopted max blueshift for BAL features

    # Determine which template to use 
    template_id = np.random.randint(0, len(baltemplates)) 
    
    # Extend the BAL template in wavelength and copy the CIV template 
    # to the NV region: 
    dbalwave = balwave[1] - balwave[0] # dispersion of bal template
    # approx. lower limit of extended template
    lambda_min_nvbal = lambda_nv*(1. + min_velocity/300000.)  
    # number of elements to add to wavelength array
    nsteps = int((balwave[0] - lambda_min_nvbal)/dbalwave) + 2 
    # update to exact lower limit of extended template
    lambda_min_nvbal = balwave[0] - dbalwave*nsteps 

    # wavelength range for NV
    balwavenv = np.arange(lambda_min_nvbal, balwave[0], dbalwave)
    # new wavelength array for BAL template
    balwaveout = np.concatenate((balwavenv, balwave)) 
    # array indices at the velocity limits of the potential CIV BAL
    civ_ind1 = np.searchsorted(balwave, lambda_civ*(1.+(min_velocity/300000.)), side="left")
    civ_ind2 = np.searchsorted(balwave, lambda_civ, side="left")

    # locate the index of the old first wavelength in the new wavelength array
    bal_ind = np.searchsorted(balwavenv, balwave[0], side="left")
    nv2civratio = lambda_nv/lambda_civ

    # put the NV component in the output template
    baltemplateout = np.interp(balwaveout, balwave[civ_ind1:civ_ind2]*nv2civratio, baltemplates[template_id][civ_ind1:civ_ind2])
    baltemplateout[bal_ind::] = baltemplates[template_id]

    # interpolate the template to match the qso and apply it 
    template_out = np.interp(qsowave, (1.+qsoredshift)*balwaveout, baltemplateout, left=1., right=1.)

    return template_out, template_id

