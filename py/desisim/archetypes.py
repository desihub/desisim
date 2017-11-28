"""
desisim.archetypes
==================

Archetype routines for desisim.

"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np

import multiprocessing
nproc = multiprocessing.cpu_count() // 2

from desiutil.log import get_logger
log = get_logger()

def compute_chi2(flux):
    """Compute the chi2 distance matrix.

    Parameters
    ----------
    flux : numpy.ndarray
        Array [Nspec, Npix] of spectra or templates where Nspec is the number of
        spectra and Npix is the number of pixels.

    Returns
    -------
    Tuple of (chi2, amp) where:
        chi2 : numpy.ndarray
            Chi^2 matrix [Nspec, Nspec] between all combinations of spectra.
        amp : numpy.ndarray
            Amplitude matrix [Nspec, Nspec] between all combinations of spectra.

    """
    from SetCoverPy import mathutils    

    nspec, npix = flux.shape
    ferr = np.ones_like(flux)
    
    chi2 = np.zeros((nspec, nspec))
    amp = np.zeros((nspec, nspec))
    for ii in range(nspec):
        if ii % 500 == 0 or ii == 0:
            log.info('Computing chi2 matrix for spectra {}-{} out of {}.'.format(
                ii*500, np.min(((ii+1)*500, nspec)), nspec))
        xx = flux[ii, :].reshape(1, npix)
        xxerr = ferr[ii, :].reshape(1, npix)
        amp1, chi21 = mathutils.quick_amplitude(xx, flux, xxerr, ferr)
        chi2[ii, :] = chi21
        amp[ii, :] = amp1
    
    return chi2, amp

class ArcheTypes(object):
    """Object for generating archetypes and determining their responsibility.

    """
    def __init__(self):
        pass

    def get_archetypes():
        """Get the final set of archetypes (and their responsibility, sorted by D4000)
        by solving the SCP problem.
        
        """
        cost = np.ones(nspec) # uniform cost
        a_matrix = (chi2 <= chi2_thresh) * 1
        gg = setcover.SetCover(a_matrix, cost)
        sol, time = gg.SolveSCP()
        
        iarch = np.nonzero(gg.s)[0]
        iarch = iarch[np.argsort(meta['D4000'][iarch])]
        nnarch = len(iarch)
        resp, respindx = responsibility(iarch, a_matrix)
    
        return iarch, resp, respindx

    def responsibility(iarch, a_matrix):
        """Method to determine the responsibility of each archetype.
    
        In essence, the responsibility is the number of templates described by each
        archetype.
        
        Parameters
        ----------
          iarch : indices of the archetypes
          a_matrix : distance matrix
        
        Returns
        -------
          resp : responsibility of each archetype (number of objects represented by each archetype)
          respindx : list containing the indices of the parent objects represented by each archetype
    
        """
        narch = len(iarch)
        resp = np.zeros(narch).astype('int16')
        respindx = []
        for ii, this in enumerate(iarch):
            respindx.append(np.where(a_matrix[:, this] == 1)[0])
            resp[ii] = np.count_nonzero(a_matrix[:, this])
            
        return resp, np.array(respindx)


