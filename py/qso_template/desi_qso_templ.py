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

from xastropy.stats import basic as xstat_b

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

##
def desi_qso_templates(z_wind=0.2, zmnx=(0.4,4.), outfil=None, Ntempl=500,
                       boss_pca_fil=None, wvmnx=(3500., 10000.),
                       sdss_pca_fil=None, no_write=False):
    '''
    Generate 10000 templates for DESI from z=0.4 to 4 

    Parameters
    ----------
    z_wind: float (0.2) 
      Window for sampling
    zmnx: tuple  ( (0.5,4) )
      Min/max for generation
    Ntempl: int  (500)
      Number of draws per redshift window
    '''
    # Cosmology
    from astropy import cosmology 
    cosmo = cosmology.core.FlatLambdaCDM(70., 0.3)

    # PCA values
    if boss_pca_fil is None:
        boss_pca_fil = 'BOSS_DR10Lya_PCA_values_nocut.fits.gz'
    hdu = fits.open(boss_pca_fil)
    boss_pca_coeff = hdu[1].data

    if sdss_pca_fil is None:
        sdss_pca_fil = 'SDSS_DR7Lya_PCA_values_nocut.fits'
    hdu2 = fits.open(sdss_pca_fil)
    sdss_pca_coeff = hdu2[1].data
    

    # Eigenvectors
    eigen, eigen_wave = fbq.read_qso_eigen()
    npix = len(eigen_wave)
    chkpix = np.where((eigen_wave > 900.) & (eigen_wave < 5000.) )[0]
    lambda_912 = 911.76
    pix912 = np.argmin( np.abs(eigen_wave-lambda_912) )

    # Open the BOSS catalog file
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    boss_zQSO = t_boss['z_pipe']

    # Open the SDSS catalog file
    sdss_cat_fil = os.environ.get('SDSSPATH')+'/DR7_QSO/dr7_qso.fits.gz'
    scat_hdu = fits.open(sdss_cat_fil)
    t_sdss = scat_hdu[1].data
    sdss_zQSO = t_sdss['z']
    if len(sdss_pca_coeff) != len(sdss_zQSO):
        print('Need to run all SDSS models!')
        sdss_zQSO = sdss_zQSO[0:len(sdss_pca_coeff)]

    # Outfil
    if outfil is None:
        outfil = 'DESI_QSO_Templates_v1.1.fits'

    # Loop on redshift
    z0 = np.arange(zmnx[0],zmnx[1],z_wind)
    z1 = z0 + z_wind

    pca_list = ['PCA0', 'PCA1', 'PCA2', 'PCA3']
    PCA_mean = np.zeros(4)
    PCA_sig = np.zeros(4)
    PCA_rand = np.zeros( (4,Ntempl*2) )

    final_spec = np.zeros( (npix, Ntempl * len(z0)) )
    final_wave = np.zeros( (npix, Ntempl * len(z0)) )
    final_z = np.zeros( Ntempl * len(z0) )

    seed = -1422
    for ii in range(len(z0)):

        # BOSS or SDSS?
        if z0[ii] > 1.99:
            zQSO = boss_zQSO
            pca_coeff = boss_pca_coeff
        else:
            zQSO = sdss_zQSO
            pca_coeff = sdss_pca_coeff

        # Random z values and wavelengths
        zrand = np.random.uniform( z0[ii], z1[ii], Ntempl*2)
        wave = np.outer(eigen_wave, 1+zrand)

        # MFP (Worseck+14)
        mfp = 37. * ( (1+zrand)/5. )**(-5.4) # Physical Mpc

        # Grab PCA mean + sigma
        idx = np.where( (zQSO >= z0[ii]) & (zQSO < z1[ii]) )[0]
        print('Making z=({:g},{:g}) with {:d} input quasars'.format(z0[ii],z1[ii],len(idx)))

        # Get PCA stats and random values
        for ipca in pca_list:
            jj = pca_list.index(ipca)
            if jj == 0: # Use bounds for PCA0 [avoids negative values]
                xmnx = xstat_b.perc( pca_coeff[ipca][idx], per=0.95 )
                PCA_rand[jj,:] = np.random.uniform( xmnx[0], xmnx[1], Ntempl*2)
            else:
                PCA_mean[jj] = np.mean(pca_coeff[ipca][idx])
                PCA_sig[jj] = np.std(pca_coeff[ipca][idx])
                # Draws
                PCA_rand[jj,:] = np.random.uniform( PCA_mean[jj] - 2*PCA_sig[jj],
                                        PCA_mean[jj] + 2*PCA_sig[jj], Ntempl*2)

        # Generate the templates (2*Ntempl)
        spec = np.dot(eigen.T,PCA_rand)

        # Take first good Ntempl

        # Truncate, MFP, Fill
        ngd = 0
        for kk in range(2*Ntempl):
            # Any zero values?
            mn = np.min(spec[chkpix,kk])
            if mn < 0.:
                continue

            # MFP
            if z0[ii] > 2.39:
                z912 = wave[0:pix912,kk]/lambda_912 - 1.
                phys_dist = np.fabs( cosmo.lookback_distance(z912) -
                                cosmo.lookback_distance(zrand[kk]) ) # Mpc
                spec[0:pix912,kk] = spec[0:pix912,kk] * np.exp(-phys_dist.value/mfp[kk]) 

            # Write
            final_spec[:, ii*Ntempl+ngd] = spec[:,kk]
            final_wave[:, ii*Ntempl+ngd] = wave[:,kk]
            final_z[ii*Ntempl+ngd] = zrand[kk]
            ngd += 1
            if ngd == Ntempl:
                break

    if no_write is True: # Mainly for plotting
        return final_wave, final_spec, final_z

    # Rebin 
    light = 2.99792458e5        # [km/s]
    velpixsize = 10.            # [km/s]
    pixsize = velpixsize/light/np.log(10) # [pixel size in log-10 A]
    minwave = np.log10(wvmnx[0])          # minimum wavelength [log10-A]
    maxwave = np.log10(wvmnx[1])          # maximum wavelength [log10-A]
    r_npix = np.round((maxwave-minwave)/pixsize+1)

    log_wave = minwave+np.arange(r_npix)*pixsize # constant log-10 spacing

    totN = Ntempl * len(z0)
    rebin_spec = np.zeros((r_npix, totN))
    
    from scipy.interpolate import interp1d
    
    for ii in range(totN):
        # Interpolate (in log space)
        f1d = interp1d(np.log10(final_wave[:,ii]), final_spec[:,ii])
        rebin_spec[:,ii] = f1d(log_wave)
        #xdb.xplot(final_wave[:,ii], final_spec[:,ii], xtwo=10.**log_wave, ytwo=rebin_spec[:,ii])
        #xdb.set_trace()

    # Write
    hdu = fits.PrimaryHDU(rebin_spec)
    hdu.header.set('PROJECT', 'DESI QSO TEMPLATES')
    hdu.header.set('VERSION', '1.1')
    hdu.header.set('OBJTYPE', 'QSO')
    hdu.header.set('DISPAXIS',  1, 'dispersion axis')
    hdu.header.set('CRPIX1',  1, 'reference pixel number')
    hdu.header.set('CRVAL1',  minwave, 'reference log10(Ang)')
    hdu.header.set('CDELT1',  pixsize, 'delta log10(Ang)')
    hdu.header.set('LOGLAM',  1, 'log10 spaced wavelengths?')
    hdu.header.set('AIRORVAC', 'vac', ' wavelengths in vacuum (vac) or air')
    hdu.header.set('VELSCALE', velpixsize, ' pixel size in km/s')
    hdu.header.set('WAVEUNIT', 'Angstrom', ' wavelength units')

    idval = range(totN)
    col0 = fits.Column(name='idval',format='K', array=idval)
    col1 = fits.Column(name='zQSO',format='E',array=final_z)
    cols = fits.ColDefs([col0, col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    hdulist = fits.HDUList([hdu, tbhdu])
    hdulist.writeto(outfil, clobber=True)

    return final_wave, final_spec, final_z

# ##################### #####################
# ##################### #####################
# Plots DESI templates at a range of z and imag
def chk_desi_qso_templates(infil=None, outfil=None, Ntempl=100):
    '''
    '''
    # Get the templates
    if infil is None:
        final_wave, final_spec, final_z = desi_qso_templates(Ntempl=Ntempl, zmnx=(0.4,0.8),
                                                             no_write=True)
    sz = final_spec.shape
    npage = sz[1] // Ntempl

    # Imports
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'stixgeneral'
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    # Eigen (for wavelengths)
    xmnx = (3600., 10000.)
                
    # Start the plot
    if outfil != None:
        pp = PdfPages(outfil)

    #xdb.set_trace()

    # Looping
    for ii in range(npage):
    #for ii in range(1):
        i0 = ii * Ntempl
        i1 = i0 + Ntempl
        ymx = 0.

        plt.figure(figsize=(8, 5))
        plt.clf()
        gs = gridspec.GridSpec(1, 1)

        # Axis
        ax = plt.subplot(gs[0])
        #ax = plt.subplot(gs[ii//2,ii%2])

        # Labels
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.set_xlim(xmnx)

        # Data
        #for jj in range(i0,i1):
        for jj in range(i0,i0+15):
            ax.plot( final_wave[:,jj], final_spec[:,jj],
                     '-',drawstyle='steps-mid', linewidth=0.5)
            ymx = max( ymx, np.max(final_spec[:,jj]) )
        ax.set_ylim( (0., ymx*1.05) )
        # Label
        zmin = np.min(final_z[i0:i1])
        zmax = np.max(final_z[i0:i1])
        zlbl = 'z=[{:g},{:g}]'.format(zmin,zmax)
        ax.text(7000., ymx*0.7, zlbl)

        # Layout and save
        plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.25)
        if outfil != None:
            pp.savefig()#bbox_inches='tight')
            plt.close()
        else: 
            plt.show()

    pp.close()
    
## #################################    
## #################################    
## TESTING
## #################################    
if __name__ == '__main__':

    # Run
    flg_test = 0 
    #flg_test += 1  # Mean templates with z,imag
    #flg_test += 2  # Mean template fig
    flg_test += 2**2  # v1.1 templates 
    #flg_test += 2**3  # Check v1.1 templates

    # Make Mean templates
    if (flg_test % 2) == 1:
        zimag = [ (2.3, 19.) ]
        mean_templ_zi(zimag)

    # Mean template fig
    if (flg_test % 2**2) >= 2**1:
        fig_desi_templ_z_i(outfil='fig_desi_templ_z_i.pdf')

    # Make z=2-4 templates; v1.1
    if (flg_test % 2**3) >= 2**2:
        aa,bb,cc = desi_qso_templates(zmnx=(2.,2.4))

    # Check z=0.4-4 templates; v1.1
    if (flg_test % 2**4) >= 2**3:
        chk_desi_qso_templates(outfil='chk_desi_qso_templates.pdf', Ntempl=20)


    # Done
    #xdb.set_trace()
    print('All done')
