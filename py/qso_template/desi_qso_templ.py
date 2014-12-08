"""
#;+ 
#; NAME:
#; fit_boss_qsos
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for Fitting PCA to the BOSS QSOs
#;   01-Dec-2014 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import os

from astropy.io import fits

import fit_boss_qsos as fbq

flg_xdb = True
try: 
    from xastropy.xutils import xdebug as xdb
except ImportError:
    flg_xdb = False


##
def mean_templ_zi(zimag, debug=False, i_wind=0.1, z_wind=0.05,
                  boss_pca_fil=None):
    '''
    Generate 'mean' templates at given z,i

    Parameters
    ----------
    zimag: list of tuples
      Redshift, imag pairs for the templates
    i_wind: float (0.1 mag)
      Window for smoothing imag
    z_wind: float (0.05 mag)
      Window for smoothing redshift
    '''
    # PCA values
    if boss_pca_fil is None:
        boss_pca_fil = 'BOSS_DR10Lya_PCA_values_nocut.fits.gz'
    hdu = fits.open(boss_pca_fil)
    pca_coeff = hdu[1].data

    # BOSS Eigenvectors
    eigen, eigen_wave = fbq.read_qso_eigen()
    npix = len(eigen_wave)

    # Open the BOSS catalog file
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    nqso = len(t_boss)
    zQSO = t_boss['z_pipe']
    tmp = t_boss['PSFMAG']
    imag = tmp[:,3] # i-band mag

    # Output array
    ntempl = len(zimag)
    out_spec = np.zeros( (ntempl, npix) ) 

    # Iterate on z,imag
    for izi in zimag:
        tt = zimag.index(izi)
        # Find matches
        idx = np.where( (np.fabs(imag-izi[1]) < i_wind) &
                        (np.fabs(zQSO-izi[0]) < z_wind))[0]
        if len(idx) < 50:
            raise ValueError('mean_templ_zi: Not enough QSOs! {:d}'.format(len(idx)))

        # Calculate median PCA values
        PCA0 = np.median(pca_coeff['PCA0'][idx])
        PCA1 = np.median(pca_coeff['PCA1'][idx])
        PCA2 = np.median(pca_coeff['PCA2'][idx])
        PCA3 = np.median(pca_coeff['PCA3'][idx])
        acoeff = np.array( [PCA0, PCA1, PCA2, PCA3] )

        # Make the template
        out_spec[tt,:] = np.dot(eigen.T,acoeff)
        if debug is True:
            xdb.xplot(eigen_wave*(1.+izi[0]), out_spec[tt,:])
            xdb.set_trace()

    # Return
    return out_spec

# ##################### #####################
# ##################### #####################
# Plots DESI templates at a range of z and imag
def fig_desi_templ_z_i(outfil=None, boss_fil=None, flg=0):
    '''
    flg = 0:  Redshift
    flg = 1:  imag
    '''

    # Todo
    #   Include NHI on the label
    # Imports
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    # Eigen (for wavelengths)
    eigen, eigen_wave = fbq.read_qso_eigen()

    all_zi = [ [ (2.3, 18.5), (2.3, 19.5), (2.3, 20.5), (2.3, 21.5) ],
               [ (2.5, 18.5), (2.5, 19.5), (2.5, 20.5), (2.5, 21.5) ],
               [ (2.7, 19.5), (2.7, 20.5), (2.7, 21.5) ],
               [ (3.2, 19.5), (3.2, 20.5), (3.2, 21.5) ] ]
               
    xmnx = (3600., 9000.)
                
    # Start the plot
    if outfil != None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(4, 1)
    #xdb.set_trace()

    # Looping
    for ii in range(4):

        # Get the templates
        ztempl = all_zi[ii][0][0]
        spec = mean_templ_zi(all_zi[ii])

        # Axis
        ax = plt.subplot(gs[ii])
        #ax = plt.subplot(gs[ii//2,ii%2])

        # Labels
        if ii == 3:
            ax.set_xlabel('Wavelength')
        else: 
            ax.get_xaxis().set_ticks([])
        ax.set_ylabel('Flux')

        ax.set_xlim(xmnx)

        # Data

        # Values
        for jj in range(len(all_zi[ii])):
            ax.plot( eigen_wave*(1.+ ztempl),
                     spec[jj,:], '-',drawstyle='steps-mid', linewidth=0.5)
            if jj == 0:
                ymx = 1.05*np.max(spec[jj,:])
                ax.set_ylim((0., ymx))
        # Label
        zlbl = 'z={:g}'.format(ztempl)
        ax.text(7000., ymx*0.7, zlbl)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.25)
    if outfil != None:
        pp.savefig(bbox_inches='tight')
        pp.close()
    else: 
        plt.show()

    
## #################################    
## #################################    
## TESTING
## #################################    
if __name__ == '__main__':

    # Run
    flg_test = 0 
    #flg_test += 1  # Mean templates with z,imag
    flg_test += 2  # Mean templates with z,imag

    # Make Mean templates
    if (flg_test % 2) == 1:
        zimag = [ (2.3, 19.) ]
        mean_templ_zi(zimag)

    # Mean template fig
    if (flg_test % 2**2) >= 2**1:
        fig_desi_templ_z_i(outfil='fig_desi_templ_z_i.pdf')


    # Done
    #xdb.set_trace()
    print('All done')
