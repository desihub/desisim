"""
desisim.spec_qa.redshifts
=========================

Module to run high_level QA on a given DESI run

Written by JXP on 3 Sep 2015
"""
from __future__ import print_function, absolute_import, division

import numpy as np
import sys, os, pdb, glob

from astropy.io import fits
from astropy.table import Table, vstack, hstack, MaskedColumn, join

from desiutil.log import get_logger, DEBUG


def elg_flux_lim(z, oii_flux):
    '''Assess which objects pass the ELG flux limit
    Uses DESI document 318 from August 2014

    Parameters
    ----------
    z : ndarray
      ELG redshifts
    oii_flux : ndarray
      [OII] fluxes
    '''
    # Init mask
    mask = np.array([False]*len(z))
    #
    # Flux limits from document
    zmin = 0.6
    zstep = 0.2
    OII_lim = np.array([10., 9., 9., 9., 9])*1e-17
    # Get bin
    zbin = (z-0.6)/zstep
    gd = np.where((zbin>0.) & (zbin<len(OII_lim)) &
        (oii_flux>0.))[0]
    # Fill gd
    OII_match = np.zeros(len(gd))
    for igd in gd:
        OII_match[igd] = OII_lim[int(zbin[igd])]
    # Query
    gd_gd = np.where(oii_flux[gd] > OII_match)[0]
    mask[gd_gd] = True
    # Return
    return mask

def catastrophic_dv(objtype):
    '''Pass back catastrophic velocity limit for given objtype
    From DESI document 318 (August 2014) in docdb

    Parameters
    ----------
    objtype : str
      Object type, e.g. 'ELG', 'LRG'
    '''
    cat_dict = dict(ELG=1000., LRG=1000., QSO_L=1000., QSO_T=1000.)
    #
    return cat_dict[objtype]


def get_sty_otype():
    '''Styles for plots'''
    sty_otype = dict(ELG={'color':'green', 'lbl':'ELG'},
        LRG={'color':'red', 'lbl':'LRG'},
        QSO={'color':'blue', 'lbl':'QSO'},
        QSO_L={'color':'blue', 'lbl':'QSO z>2.1'},
        QSO_T={'color':'cyan', 'lbl':'QSO z<2.1'})
    return sty_otype
