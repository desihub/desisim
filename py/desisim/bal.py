"""
desisim.bal
===========

Functions and methods for inserting BALs into QSO spectra.

"""
from __future__ import division, print_function

import numpy as np

class BAL(object):
    """Base class for inserting BALs into (input) QSO spectra."""
    def __init__(self):
        """Read and cache the BAL template set.

        Attributes:
          balflux (numpy.ndarray): Array [nbase,npix] of the rest-frame BAL
            templates.
          balwave (numpy.ndarray): Array [npix] of rest-frame wavelengths
            corresponding to BASEFLUX (Angstrom).
          balmeta (astropy.Table): Table of metadata [nbase] for each template.

        """
        from desisim.io import read_basis_templates
        
        balflux, balwave, balmeta = read_basis_templates(objtype='BAL')
        self.balflux = balflux
        self.balwave = balwave
        self.balmeta = balmeta

    def empty_balmeta(self, qsoredshift=None):
        """Initialize an empty metadata table for BALs."""

        from astropy.table import Table, Column

        if qsoredshift is None:
            nqso = 1
        else:
            nqso = len(np.atleast_1d(qsoredshift))

        balmeta = Table()
        balmeta.add_column(Column(name='BAL_TEMPLATEID', length=nqso, dtype='i4', data=np.zeros(nqso)-1))
        balmeta.add_column(Column(name='Z',length=nqso, dtype='f4', data=np.zeros(nqso)))
        if qsoredshift is not None:
            balmeta['Z'] = qsoredshift

        return balmeta

    def template_balmeta(self,indx):
        """Initialize an empty metadata table for BALs."""

        from astropy.table import Table, Column
        from desisim.io import find_basis_template

        balmeta = self.balmeta.copy()
        balmeta = balmeta[indx]
        nbal = len(balmeta)
        balmeta.add_column(Column(name='BAL_TEMPLATEID', length=nbal, dtype='i4', data=np.zeros(nbal)-1), index=0)
        balmeta.add_column(Column(name='BAL_PROB', length=nbal, dtype='f4', data=np.ones(nbal)), index=0)
        balmeta.add_column(Column(name='Z', length=nbal, dtype='f4', data=np.zeros(nbal)), index=0)
        balmeta.add_column(Column(name='TARGETID', length=nbal, dtype='i4', data=np.zeros(nbal)-1), index=0)
        return balmeta
        
    def insert_bals(self, qsowave, qsoflux, qsoredshift, balprob=0.12,
                    seed=None, verbose=False, qsoid=None):
        """Probabilistically inserts BALs into one or more QSO spectra.

        Args:
            qsowave (numpy.ndarray): observed-frame wavelength array [Angstrom]
            qsoflux (numpy.ndarray): array of observed frame flux values.
            qsoredshift (numpy.array or float): QSO redshift
            balprob (float, optional): Probability that a QSO is a BAL (default
                0.12).  Only used if QSO(balqso=True) at instantiation.
            seed (int, optional): input seed for the random numbers.
            verbose (bool, optional): Be verbose!

        Returns:
            bal_qsoflux (numpy.ndarray): QSO spectrum with the BAL included.
            balmeta (astropy.Table): metadata table for each BAL.

        """
        from desiutil.log import get_logger, DEBUG
        from desispec.interpolation import resample_flux
        from astropy.table import Table


        if verbose:
            log = get_logger(DEBUG)
        else:
            log = get_logger()

        rand = np.random.RandomState(seed)

        if balprob < 0:
            log.warning('Balprob {} is negative; setting to zero.'.format(balprob))
            balprob = 0.0
        if balprob > 1:
            log.warning('Balprob {} cannot exceed unity; setting to 1.0.'.format(balprob))
            balprob = 1.0

        nqso, nwave = qsoflux.shape
        if len(qsoredshift) != nqso:
            log.fatal('Dimensions of qsoflux and qsoredshift do not agree!')
            raise ValueError
        
        if qsowave.ndim == 2: # desisim.QSO(resample=True) returns a 2D wavelength array
            w_nqso, w_nwave = qsowave.shape
            if w_nwave != nwave or w_nqso != nqso:
                log.fatal('Dimensions of qsoflux and qsowave do not agree!')
                raise ValueError
        else:
            if len(qsowave) != nwave:
                log.fatal('Dimensions of qsoflux and qsowave do not agree!')
                raise ValueError
        
        # Determine which QSO spectrum has BAL(s) and then loop on each. 
        hasbal = rand.random_sample(nqso) < balprob
        ihasbal = np.where(hasbal)[0]
        # Should probably return a BAL metadata table, too.
        if len(ihasbal) == 0:
            #Return a fully empy balmeta table
            balmeta=Table(names=('TARGETID','Z','BAL_PROB','BAL_TEMPLATEID'), dtype=('i4', 'f4', 'f4','i4'))
            return qsoflux, balmeta

        balindx = rand.choice( len(self.balmeta), len(ihasbal) )
        #before it was convenient to have the balmeta of size  nqso's and remove non-BALs after. Now I think is easier to return the balmeta for BALs only.
        balmeta = self.template_balmeta(balindx)
        balmeta['Z'] = qsoredshift[ihasbal]
        balmeta['BAL_TEMPLATEID'] = balindx
        balmeta['TARGETID'] = qsoid[ihasbal]

        bal_qsoflux = qsoflux.copy()
        if qsowave.ndim == 2:
            for ii, indx in zip( ihasbal, balindx ):
                thisbalflux = resample_flux(qsowave[ii, :], self.balwave*(1 + qsoredshift[ii]),
                                            self.balflux[indx, :], extrapolate=True)
                bal_qsoflux[ii, :] *= thisbalflux
        else:
            for ii, indx in zip( ihasbal, balindx ):
                thisbalflux = resample_flux(qsowave, self.balwave*(1 + qsoredshift[ii]),
                                            self.balflux[indx, :], extrapolate=True)
                bal_qsoflux[ii, :] *= thisbalflux

        return bal_qsoflux, balmeta
