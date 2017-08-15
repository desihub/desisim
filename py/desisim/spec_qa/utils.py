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
    OII_lim = np.array([10., 9., 9., 9., 9])*1e-17
    # Get bin
    #zbin = (z-0.6)/zstep
    zbins = [0.6, 0.8, 1., 1.2, 1.4, 5.6]
    z_i = np.digitize(z, zbins) - 1
    gd_elg = (z_i >= 0) & (oii_flux>0.)
    #gd = np.where((zbin>0.) & (zbin<len(OII_lim)) & (oii_flux>0.))[0]
    # Fill gd
    OII_match = np.ones_like(gd_elg) * 9e99
    for ss in range(len(zbins)-1):
        OII_match[z_i==ss] = OII_lim[ss]
    # Query
    really_gd = (oii_flux > OII_match) & gd_elg
    mask[really_gd] = True
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
    sty_otype = dict(ELG={'color':'green', 'pcolor':'Greens', 'lbl':'ELG'},
        LRG={'color':'red', 'lbl':'LRG', 'pcolor':'Reds'},
                     BGS={'color':'orange', 'lbl':'BGS', 'pcolor':'Oranges'},
        QSO={'color':'blue', 'lbl':'QSO', 'pcolor':'Blues'},
        QSO_L={'color':'blue', 'lbl':'QSO z>2.1', 'pcolor':'Blues'},
        QSO_T={'color':'cyan', 'lbl':'QSO z<2.1', 'pcolor':'GnBu'})
    return sty_otype

def match_otype(tbl, objtype):
    """ Generate a mask for the input objtype
    :param tbl:
    :param objtype: str
    :return: targets: bool mask
    """
    from desitarget import desi_mask
    if objtype in ['BGS']:
        targets = (tbl['DESI_TARGET'] & desi_mask['BGS_ANY']) != 0
    elif objtype in ['MWS']:
        targets = (tbl['DESI_TARGET'] & desi_mask['MWS_ANY']) != 0
        import pdb; pdb.set_trace()
    elif objtype in ['QSO_L']:
        targets = np.where( match_otype(tbl, 'QSO') &
                            (tbl['TRUEZ'] >= 2.1))[0]
    elif objtype in ['QSO_T']:
        targets = np.where( match_otype(tbl, 'QSO') &
                     (tbl['TRUEZ'] < 2.1))[0]
    else:
        targets = (tbl['DESI_TARGET'] & desi_mask[objtype]) != 0
    # Return
    return targets
