"""
desisim.bal
===========

Functions and methods for inserting BALs into QSO spectra.

"""
from __future__ import division, print_function

import numpy as np
from astropy.io import fits 
import random 

def readbaltemplates(balfile):
    """ Read in the BAL templates fits file
    Args: 
	balfile: name of fits file with BAL templates
    Returns: 
	wave_bal_template (ndarray): 1-D wavelength array in Ang for 
	  all templates in rest frame 
	bal_templates (ndarray): 2-d array with all BAL templates 
    """ 

    hdu = fits.open(balfile)
    bal_templates = hdu[1].data['TEMP'] 
    nlam = len(hdu[1].data['TEMP'][0])	# number of wavelength elements
    lam1 = hdu[1].header['CRVAL1']
    dlam = hdu[1].header['CDELT1']
    lam2 = lam1 + dlam*nlam
    wave_bal_template = np.arange(lam1, lam2, dlam)
    return wave_bal_template, bal_templates 


def isbal(balprob, balrand): 
    """ Randomly determine if a QSO is a BAL 
    Args: 
	balprob (float) : probability that a QSO is a BAL 
	Expect about 12% based on DR12
        balrand : np.random.RandomState(seed) 
    Returns: 
	booleal: True if a BAL, otherwise False
    """ 

    # Make sure probability is on the interval [0.,1.]
    if balprob > 1.:
        balprob = 1. 
    if balprob < 0.: 
        balprob = 0.
    if balrand.random_sample() < balprob:
        return True
    return False


def getbaltemplate(qsowave, qsoredshift, balwave, baltemplates):
    """ Get BAL template for a single QSO. 

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

    # Determine which template to use 
    template_id = balrand.random_integers(0, len(baltemplates)) 
    
    # interpolate the template to match the qso and apply it 
    template_out = np.interp(qsowave, (1.+qsoredshift)*balwave, baltemplates[template_id], left=1., right=1.)

    return template_out, template_id

