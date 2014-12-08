"""
#;+ 
#; NAME:
#; boss_qso_figs
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for making figures with the BOSS QSO fit outpus
#;   02-Dec-2014 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import os

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

from astropy.io import fits

from xastropy.plotting import utils as xputils

try: 
    from xastropy.xutils import xdebug as xdb
except ImportError:
    import pdb as xdb

# ##################### #####################
# ##################### #####################
# Plots the PCA coeff against one another from BOSS DR10 QSOs
def fig_boss_pca_coeff(outfil=None, boss_fil=None):

    # Todo
    #   Include NHI on the label
    # Imports

    # Read FITS table
    if boss_fil is None:
        boss_fil = 'BOSS_DR10Lya_PCA_values.fits.gz'
    hdu = fits.open(boss_fil)
    pca_coeff = hdu[1].data
    #xdb.set_trace()
    
    # Initialize
    #if 'xmnx' not in locals():
    #    xmnx = (17.0, 20.4)
    #ymnx = ((-1.9, 2.3),
    #        (-1.9, 1.4),
    #        (-1.1, 1.4),
    #        (-2.1, 0.9))
                
    ms = 1. # point size
    scl = 100.
            
    allxi = [0,0,0,1,1,2]
    allyi = [1,2,3,2,3,3]

    rngi = ([-0.02, 0.3],
            [-0.2,0.4],
            [-0.2, 0.5],
            [-0.2, 0.4])
    
    # Start the plot
    if outfil != None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(4, 5.5))
    plt.clf()
    gs = gridspec.GridSpec(3, 2)


    # Looping
    for ii in range(6):

        # Axis
        ax = plt.subplot(gs[ii])
        #ax = plt.subplot(gs[ii//2,ii%2])

        '''
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.))
        #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.set_xlim(xmnx)
        ax.set_ylim((-0.5, 0.5))
        '''

        xi = allxi[ii]
        xlbl=str('PCA'+str(xi))
        yi = allyi[ii]
        ylbl=str('PCA'+str(yi))


        # Labels
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)

        # Data

        # Values
        #ax.scatter(pca_coeff[xlbl], pca_coeff[ylbl], color='black', s=ms)#, marker=mark)
        ax.hist2d(scl*pca_coeff[xlbl], scl*pca_coeff[ylbl], bins=100, norm=LogNorm(),
                  range=[rngi[xi],rngi[yi]])

        # Font size
        xputils.set_fontsize(ax,8.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.25)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
    else: 
        plt.show()

# ##################### #####################
# ##################### #####################
# Plots X vs PCA coeff from BOSS DR10 QSOs
def fig_boss_x_vs_pca(outfil=None, boss_fil=None, flg=0):
    '''
    flg = 0:  Redshift
    flg = 1:  imag
    '''

    # Todo
    #   Include NHI on the label
    # Imports

    # Read PCA FITS table
    if boss_fil is None:
        boss_fil = 'BOSS_DR10Lya_PCA_values.fits.gz'
    hdu = fits.open(boss_fil)
    pca_coeff = hdu[1].data

    # Read BOSS catalog table
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    #xdb.set_trace()
    if flg == 0:
        xparm = t_boss['z_pipe']
        xmnx = [2., 4.]
        xlbl=str(r'$z_{\rm QSO}$')
    elif flg == 1:
        tmp = t_boss['PSFMAG']
        xparm = tmp[:,3] # i-band mag
        xlbl=str('i mag')
        xmnx = [17.,22.5]
    else:
        raise ValueError('fig_boss_x_vs_pca: flg={:d} not allowed'.format(flg))
    
    # Initialize
    #if 'xmnx' not in locals():
    #    xmnx = (17.0, 20.4)
    ymnx = ((-0.02, 0.3),
            (-0.2,0.4),
            (-0.2, 0.5),
            (-0.2, 0.4))
                
    ms = 1. # point size
    scl = 100.
            
    
    # Start the plot
    if outfil != None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(4, 4))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)


    # Looping
    for ii in range(4):

        # Axis
        ax = plt.subplot(gs[ii])
        #ax = plt.subplot(gs[ii//2,ii%2])


        ylbl=str('PCA'+str(ii))

        # Labels
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)

        # Data

        # Values
        #xdb.set_trace()
        ax.hist2d(xparm, scl*pca_coeff[ylbl], bins=100, norm=LogNorm(),
                  range=[xmnx, [ymnx[ii][0], ymnx[ii][1]]])

        # Font size
        xputils.set_fontsize(ax,8.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.25)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
    else: 
        plt.show()


#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

# Command line execution
if __name__ == '__main__':

    flg_fig = 0 
    flg_fig += 1  # PCA vs PCA
    flg_fig += 2**1  # PCA vs z
    flg_fig += 2**2  # PCA vs i

    # PCA vs PCA
    if (flg_fig % 2) == 1:
        fig_boss_pca_coeff(outfil='fig_boss_pca.pdf')

    # PCA vs z
    if (flg_fig % 2**2) >= 2**1:
        fig_boss_x_vs_pca(outfil='fig_boss_z_vs_pca.pdf',flg=0)

    # PCA vs imag
    if (flg_fig % 2**3) >= 2**2:
        fig_boss_x_vs_pca(outfil='fig_boss_imag_vs_pca.pdf',flg=1)
