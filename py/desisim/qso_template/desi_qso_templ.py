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
import imp
import pdb

from scipy.interpolate import interp1d

from astropy.io import fits

from desisim.qso_template import fit_boss_qsos as fbq
from desiutil.stats import perc
import desisim.io

from desispec.log import get_logger
log = get_logger()

#from xastropy.stats.basic import perc

flg_xdb = True
try:
    from xastropy.xutils import xdebug as xdb
except ImportError:
    flg_xdb = False


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


def desi_qso_templates(z_wind=0.2, zmnx=(0.4,4.), outfil=None, N_perz=500,
                       boss_pca_fil=None, wvmnx=(3500., 10000.),
                       rebin_wave=None, rstate=None,
                       sdss_pca_fil=None, no_write=False, redshift=None,
                       seed=None, old_read=False, ipad=20):
    """ Generate QSO templates for DESI

    Rebins to input wavelength array (or log10 in wvmnx)

    Parameters
    ----------
    z_wind : float, optional
      Window for sampling
    zmnx : tuple, optional
      Min/max for generation
    N_perz : int, optional
      Number of draws per redshift window
    old_read : bool, optional
      Read the files the old way
    seed : int, optional
      Seed for the random number state
    rebin_wave : ndarray, optional
      Input wavelengths for rebinning
    wvmnx : tuple, optional
      Wavelength limits for rebinning (not used with rebin_wave)
    redshift : ndarray, optional
      Redshifts desired for the templates
    ipad : int, optional
      Padding for enabling enough models

    Returns
    -------
    wave : ndarray
      Wavelengths that the spectra were rebinned to
    flux : ndarray (2D; flux vs. model)
    z : ndarray
      Redshifts
    """
    # Cosmology
    from astropy import cosmology
    from desispec.interpolation import resample_flux
    cosmo = cosmology.core.FlatLambdaCDM(70., 0.3)

    if old_read:
        # PCA values
        if boss_pca_fil is None:
            boss_pca_fil = 'BOSS_DR10Lya_PCA_values_nocut.fits.gz'
        hdu = fits.open(boss_pca_fil)
        boss_pca_coeff = hdu[1].data

        if sdss_pca_fil is None:
            sdss_pca_fil = 'SDSS_DR7Lya_PCA_values_nocut.fits.gz'
        hdu2 = fits.open(sdss_pca_fil)
        sdss_pca_coeff = hdu2[1].data

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
            print('Need to finish running the SDSS models!')
            sdss_zQSO = sdss_zQSO[0:len(sdss_pca_coeff)]
        # Eigenvectors
        eigen, eigen_wave = fbq.read_qso_eigen()
    else:
        infile = desisim.io.find_basis_template('qso')
        hdus = fits.open(infile)

        hdu_names = [hdus[ii].name for ii in range(len(hdus))]
        boss_pca_coeff = hdus[hdu_names.index('BOSS_PCA')].data
        sdss_pca_coeff = hdus[hdu_names.index('SDSS_PCA')].data
        boss_zQSO = hdus[hdu_names.index('BOSS_Z')].data
        sdss_zQSO = hdus[hdu_names.index('SDSS_Z')].data
        eigen = hdus[hdu_names.index('SDSS_EIGEN')].data
        eigen_wave = hdus[hdu_names.index('SDSS_EIGEN_WAVE')].data

    # Fiddle with the eigen-vectors
    npix = len(eigen_wave)
    chkpix = np.where((eigen_wave > 900.) & (eigen_wave < 5000.) )[0]
    lambda_912 = 911.76
    pix912 = np.argmin( np.abs(eigen_wave-lambda_912) )

    # Loop on redshift.  If the
    if redshift is None:
        z0 = np.arange(zmnx[0],zmnx[1],z_wind)
        z1 = z0 + z_wind
    else:
        z0 = np.array([redshift])
        z1 = z0

    pca_list = ['PCA0', 'PCA1', 'PCA2', 'PCA3']
    PCA_mean = np.zeros(4)
    PCA_sig = np.zeros(4)
    PCA_rand = np.zeros((4,N_perz*ipad))

    final_spec = np.zeros((npix, N_perz * len(z0)))
    final_wave = np.zeros((npix, N_perz * len(z0)))
    final_z = np.zeros(N_perz * len(z0))

    # Random state
    if rstate is None:
        rstate = np.random.RandomState(seed)

    for ii in range(len(z0)):

        # BOSS or SDSS?
        if z0[ii] > 1.99:
            zQSO = boss_zQSO
            pca_coeff = boss_pca_coeff
        else:
            zQSO = sdss_zQSO
            pca_coeff = sdss_pca_coeff

        # Random z values and wavelengths
        zrand = rstate.uniform( z0[ii], z1[ii], N_perz*ipad)
        wave = np.outer(eigen_wave, 1+zrand)

        # MFP (Worseck+14)
        mfp = 37. * ( (1+zrand)/5. )**(-5.4) # Physical Mpc

        # Grab PCA mean + sigma
        if redshift is None:
            idx = np.where( (zQSO >= z0[ii]) & (zQSO < z1[ii]) )[0]
        else:
            # Hack by @moustakas: add a little jitter to get the set of QSOs
            # that are *nearest* in redshift to the desired output redshift.
            idx = np.where( (zQSO >= z0[ii]-0.001) & (zQSO < z1[ii]+0.001) )[0]
            #idx = np.array([(np.abs(zQSO-zrand[0])).argmin()])
        #pdb.set_trace()
        log.debug('Making z=({:g},{:g}) with {:d} input quasars'.format(z0[ii],z1[ii],len(idx)))

        # Get PCA stats and random values
        for jj,ipca in enumerate(pca_list):
            if jj == 0:  # Use bounds for PCA0 [avoids negative values]
                xmnx = perc(pca_coeff[ipca][idx], per=95)
                PCA_rand[jj, :] = rstate.uniform(xmnx[0], xmnx[1], N_perz*ipad)
            else:
                PCA_mean[jj] = np.mean(pca_coeff[ipca][idx])
                PCA_sig[jj] = np.std(pca_coeff[ipca][idx])
                # Draws
                PCA_rand[jj, :] = rstate.uniform( PCA_mean[jj] - 2*PCA_sig[jj],
                                        PCA_mean[jj] + 2*PCA_sig[jj], N_perz*ipad)

        # Generate the templates (ipad*N_perz)
        spec = np.dot(eigen.T, PCA_rand)

        # Take first good N_perz

        # Truncate, MFP, Fill
        ngd = 0
        nbad = 0
        for kk in range(ipad*N_perz):
            # Any zero values?
            mn = np.min(spec[chkpix, kk])
            if mn < 0.:
                nbad += 1
                continue

            # MFP
            if z0[ii] > 2.39:
                z912 = wave[0:pix912,kk]/lambda_912 - 1.
                phys_dist = np.fabs( cosmo.lookback_distance(z912) -
                                cosmo.lookback_distance(zrand[kk]) ) # Mpc
                spec[0:pix912, kk] = spec[0:pix912,kk] * np.exp(-phys_dist.value/mfp[kk])

            # Write
            final_spec[:, ii*N_perz+ngd] = spec[:,kk]
            final_wave[:, ii*N_perz+ngd] = wave[:,kk]
            final_z[ii*N_perz+ngd] = zrand[kk]
            ngd += 1
            if ngd == N_perz:
                break
        if ngd != N_perz:
            print('Did not make enough!')
            pdb.set_trace()

    # Rebin
    if rebin_wave is None:
        light = 2.99792458e5        # [km/s]
        velpixsize = 10.            # [km/s]
        pixsize = velpixsize/light/np.log(10) # [pixel size in log-10 A]
        minwave = np.log10(wvmnx[0])          # minimum wavelength [log10-A]
        maxwave = np.log10(wvmnx[1])          # maximum wavelength [log10-A]
        r_npix = np.round((maxwave-minwave)/pixsize+1)

        log_wave = minwave+np.arange(r_npix)*pixsize # constant log-10 spacing
    else:
        log_wave = np.log10(rebin_wave)
        r_npix = len(log_wave)

    totN = N_perz * len(z0)
    rebin_spec = np.zeros((r_npix, totN))


    for ii in range(totN):
        # Interpolate (in log space)
        rebin_spec[:, ii] = resample_flux(log_wave, np.log10(final_wave[:, ii]), final_spec[:, ii])
        #f1d = interp1d(np.log10(final_wave[:,ii]), final_spec[:,ii])
        #rebin_spec[:,ii] = f1d(log_wave)

    if outfil is None:
        return 10.**log_wave, rebin_spec, final_z

    # Transpose for consistency
    out_spec = np.array(rebin_spec.T, dtype='float32')

    # Write
    hdu = fits.PrimaryHDU(out_spec)
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
    hdu.header.set('BUNIT', '1e-17 erg/s/cm2/A', ' flux unit')

    idval = range(totN)
    col0 = fits.Column(name=str('TEMPLATEID'),format=str('J'), array=idval)
    col1 = fits.Column(name=str('Z'),format=str('E'),array=final_z)
    cols = fits.ColDefs([col0, col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.header.set('EXTNAME','METADATA')

    hdulist = fits.HDUList([hdu, tbhdu])
    hdulist.writeto(outfil, clobber=True)

    return final_wave, final_spec, final_z

# ##################### #####################
# ##################### #####################
# Plots DESI templates at a range of z and imag
def chk_desi_qso_templates(infil=None, outfil=None, N_perz=100):
    '''
    '''
    # Get the templates
    if infil is None:
        final_wave, final_spec, final_z = desi_qso_templates(N_perz=N_perz, #zmnx=(0.4,0.8),
                                                             no_write=True)
    sz = final_spec.shape
    npage = sz[1] // N_perz

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

    # Looping
    for ii in range(npage):
    #for ii in range(1):
        i0 = ii * N_perz
        i1 = i0 + N_perz
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

def repackage_coeff(boss_pca_fil=None, sdss_pca_fil=None,
                    outfil='qso_templates_v2.0.fits'):
    """ Repackage the coefficients and redshifts into a single FITS file

    :return:
    """
    # PCA values
    if boss_pca_fil is None:
        boss_pca_fil = 'BOSS_DR10Lya_PCA_values_nocut.fits.gz'
    hdu = fits.open(boss_pca_fil)
    boss_pca_coeff = hdu[1].data

    if sdss_pca_fil is None:
        sdss_pca_fil = 'SDSS_DR7Lya_PCA_values_nocut.fits.gz'
    hdu2 = fits.open(sdss_pca_fil)
    sdss_pca_coeff = hdu2[1].data

    # Redshifts
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    boss_zQSO = np.array(t_boss['z_pipe'])

    # Open the SDSS catalog file
    sdss_cat_fil = os.environ.get('SDSSPATH')+'/DR7_QSO/dr7_qso.fits.gz'
    scat_hdu = fits.open(sdss_cat_fil)
    t_sdss = scat_hdu[1].data
    sdss_zQSO = t_sdss['z']
    if len(sdss_pca_coeff) != len(sdss_zQSO):
        print('Need to finish running the SDSS models!')
        sdss_zQSO = sdss_zQSO[0:len(sdss_pca_coeff)]

    # Eigen vectors
    eigen, eigen_wave = fbq.read_qso_eigen()

    # Write
    phdu = fits.PrimaryHDU()
    bp_hdu = fits.BinTableHDU(boss_pca_coeff)
    bp_hdu.name = 'BOSS_PCA'
    bz_hdu = fits.ImageHDU(boss_zQSO)
    bz_hdu.name = 'BOSS_z'
    sp_hdu = fits.BinTableHDU(sdss_pca_coeff)
    sp_hdu.name = 'SDSS_PCA'
    sz_hdu = fits.ImageHDU(sdss_zQSO)
    sz_hdu.name = 'SDSS_z'
    e_hdu = fits.ImageHDU(eigen)
    e_hdu.name = 'SDSS_EIGEN'
    ew_hdu = fits.ImageHDU(eigen_wave)
    ew_hdu.name = 'SDSS_EIGEN_WAVE'

    hdulist = fits.HDUList([phdu, bp_hdu, bz_hdu, sp_hdu, sz_hdu,
                            e_hdu, ew_hdu])
    hdulist.writeto(outfil, clobber=True)
    print('Wrote {:s}'.format(outfil))


def tst_random_set():
    """ Generate a small set of random templates for testing
    :return:
    """
    final_wave, final_spec, final_z = desi_qso_templates(
            outfil='test_random_set.fits', N_perz=100, seed=12345)


## #################################
## #################################
## TESTING
## #################################
if __name__ == '__main__':

    # Run
    flg_test = 0
    #flg_test += 1  # Mean templates with z,imag
    #flg_test += 2  # Mean template fig
    #flg_test += 2**2  # v1.1 templates
    #flg_test += 2**3  # Check v1.1 templates
    #flg_test += 2**4  # PCA file
    flg_test += 2**5  # Generate a new random set

    # Make Mean templates
    if (flg_test % 2) == 1:
        zimag = [ (2.3, 19.) ]
        mean_templ_zi(zimag)

    # Mean template fig
    if (flg_test % 2**2) >= 2**1:
        fig_desi_templ_z_i(outfil='fig_desi_templ_z_i.pdf')

    # Make z=2-4 templates; v1.1
    if (flg_test % 2**3) >= 2**2:
        aa,bb,cc = desi_qso_templates(outfil='DESI_QSO_Templates_v1.1.fits')

    # Check z=0.4-4 templates; v1.1
    if (flg_test % 2**4) >= 2**3:
        chk_desi_qso_templates(outfil='chk_desi_qso_templates.pdf', N_perz=20)

    # Re-package PCA info
    if (flg_test % 2**5) >= 2**4:
        repackage_coeff()

    # Test random generation
    if (flg_test % 2**6) >= 2**5:
        tst_random_set()


    # Done
    print('All done')
