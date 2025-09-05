"""
desisim.archetypes
==================

Archetype routines for desisim.

"""

import os
import numpy as np

from desiutil.log import get_logger
log = get_logger()

def compute_chi2(flux, ferr=None):
    """Compute the chi2 distance matrix.

    Parameters
    ----------
    flux : numpy.ndarray
        Array [Nspec, Npix] of spectra or templates where Nspec is the number of
        spectra and Npix is the number of pixels.
    ferr : numpy.ndarray
        Uncertainty spectra ccorresponding to flux (default None).

    Returns
    -------
    Tuple of (chi2, amp) where:
        chi2 : numpy.ndarray
            Chi^2 matrix [Nspec, Nspec] between all combinations of normalized spectra.
        amp : numpy.ndarray
            Amplitude matrix [Nspec, Nspec] between all combinations of spectra.

    """

    nspec, npix = flux.shape
    chi2 = np.zeros((nspec, nspec), dtype='f4')
    amp = np.zeros((nspec, nspec), dtype='f4')

    flux = flux.copy()
    rescale = np.sqrt(npix/np.sum(flux**2,axis=1))
    flux *= rescale[:,None]

    if ferr is None:
        for ii in range(nspec):
            if ii % 500 == 0:
                log.info('Computing chi2 matrix for spectra {}-{} out of {}.'.format(
                    int(ii/500) * 500, np.min(((ii+1) * int(ii/500), nspec-1)), nspec))
            amp1 = np.sum(flux[ii]*flux,axis=1)/npix
            chi2[ii,:] = npix*(1.-amp1**2)
            amp[ii,:] = amp1
    else:
        from SetCoverPy import mathutils
        ferr = ferr.copy()
        ferr *= rescale[:,None]
        for ii in range(nspec):
            if ii % 500 == 0 or ii == 0:
                log.info('Computing chi2 matrix for spectra {}-{} out of {}.'.format(
                    int(ii/500) * 500, np.min(((ii+1) * int(ii/500), nspec-1)), nspec))
            xx = flux[ii, :].reshape(1, npix)
            xxerr = ferr[ii, :].reshape(1, npix)
            amp[ii,:], chi2[ii,:] = mathutils.quick_amplitude(xx, flux, xxerr, ferr, niter=1)

    amp *= rescale[:,None]/rescale[None,:]
    np.fill_diagonal(chi2,0.)
    np.fill_diagonal(amp,1.)

    return chi2, amp

class ArcheTypes(object):
    """Object for generating archetypes and determining their responsibility.

    Parameters
    ----------
    chi2 : numpy.ndarray
        Chi^2 matrix computed by desisim.archetypes.compute_chi2().
    
    """
    def __init__(self, chi2):

        self.chi2 = chi2

    def get_archetypes(self, chi2_thresh=0.1, responsibility=False):
        """Solve the SCP problem to get the final set of archetypes and, optionally,
        their responsibility.

        Note: We assume that each template has uniform "cost" but a more general
        model in principle could be used / implemented.

        Parameters
        ----------
        chi2 : numpy.ndarray
            Chi^2 matrix computed by archetypes.compute_chi2().
        chi2_thresh : float
            Threshold chi2 value to differentiate "different" templates.
        responsibility : bool
            If True, then compute and return the responsibility of each archetype. 

        Returns
        -------
        If responsibility==True then returns a tuple of (iarch, resp, respindx) where:
            iarch : integer numpy.array
                Indices of the archetypes [N].
            resp : integer numpy.array
                Responsibility of each archetype [N].
            respindx : list of 
                Indices the parent sample each archetype is responsible for [N].

        If responsibility==False then only iarch is returned.

        """
        from SetCoverPy import setcover

        nspec = self.chi2[0].shape
        cost = np.ones(nspec) # uniform cost
        
        a_matrix = (self.chi2 <= chi2_thresh) * 1
        gg = setcover.SetCover(a_matrix, cost)
        sol, time = gg.SolveSCP()
        
        iarch = np.nonzero(gg.s)[0]
        if responsibility:
            resp, respindx = self.responsibility(iarch, a_matrix)
            return iarch, resp, respindx
        else:
            return iarch
  
    def responsibility(self, iarch, a_matrix):
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
            
        return resp, respindx


