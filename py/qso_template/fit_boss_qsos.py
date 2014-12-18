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
import multiprocessing as mp
import Queue

#Queue.Queue(30000000)

from astropy.io import fits

flg_xdb = True
try: 
    from xastropy.xutils import xdebug as xdb
except ImportError:
    flg_xdb = False

#
def read_qso_eigen(eigen_fil=None):
    '''
    Input the QSO Eigenspectra
    '''
    # File
    if eigen_fil is None:
        eigen_fil = os.environ.get('IDLSPEC2D_DIR')+'/templates/spEigenQSO-55732.fits'
    print('Using these eigen spectra: {:s}'.format(eigen_fil))
    hdu = fits.open(eigen_fil)
    eigen = hdu[0].data
    head = hdu[0].header
    # Rest wavelength
    eigen_wave = 10.**(head['COEFF0'] + np.arange(head['NAXIS1'])*head['COEFF1'])

    # Return
    return eigen, eigen_wave

##
def fit_eigen(flux,ivar,eigen_flux):
    '''
    Fit the spectrum with the eigenvectors.
    Pass back the coefficients
    '''
    #C = np.diag(1./ivar)
    Cinv = np.diag(ivar)
    A = eigen_flux.T

    #alpha = np.dot(A.T, np.linalg.solve(C, A))  # Numerical Recipe notation
    alpha = np.dot(A.T, np.dot(Cinv,A))
    cov = np.linalg.inv(alpha)
    #beta = np.dot(A.T, np.linalg.solve(C, y))
    beta = np.dot(A.T, np.dot(Cinv, flux))
    acoeff = np.dot(cov, beta)

    # Return
    return acoeff

##
def do_boss_lya_parallel(istart, iend, cut_Lya, output, debug=False):
    '''
    Generate PCA coeff for the BOSS Lya DR10 dataset, v2.1

    Parameters
    ----------
    cut_Lya: boolean (True)
      Avoid using the Lya forest in the analysis
    '''
    # Eigen
    eigen, eigen_wave = read_qso_eigen()

    # Open the BOSS catalog file
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    nqso = len(t_boss)

    pca_val = np.zeros((iend-istart, 4))

    if cut_Lya is False:
        print('do_boss: Not cutting the Lya Forest in the fit')

    # Loop us -- Should spawn on multiple CPU
    #for ii in range(nqso):
    datdir =  os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_spectra_v2.1/'
    jj = 0
    for ii in range(istart,iend):
        if (ii % 1000) == 0:
            print('ii = {:d}'.format(ii))
        # Spectrum file
        pnm = str(t_boss['PLATE'][ii])
        fnm = str(t_boss['FIBERID'][ii]).rjust(4,str('0'))
        mjd = str(t_boss['MJD'][ii])
        sfil = datdir+pnm+'/speclya-'
        sfil = sfil+pnm+'-'+mjd+'-'+fnm+'.fits.gz'
        # Read spectrum
        spec_hdu = fits.open(sfil)
        t = spec_hdu[1].data
        flux = t['flux']
        wave = 10.**t['loglam']
        ivar = t['ivar']
        zqso = t_boss['z_pipe'][ii]

        wrest  = wave / (1+zqso)
        wlya = 1215. 

        # Cut Lya forest?
        if cut_Lya is True:
            Ly_imn = np.argmin(np.abs(wrest-wlya))
        else:
            Ly_imn = 0
            
        # Pack
        imn = np.argmin(np.abs(wrest[Ly_imn]-eigen_wave))
        npix = len(wrest[Ly_imn:])
        imx = npix+imn
        eigen_flux = eigen[:,imn:imx]

        # FIT
        acoeff = fit_eigen(flux[Ly_imn:], ivar[Ly_imn:], eigen_flux)
        pca_val[jj,:] = acoeff
        jj += 1

        # Check
        if debug is True:
            model = np.dot(eigen.T,acoeff)
            if flg_xdb is True:
                xdb.xplot(wrest, flux, xtwo=eigen_wave, ytwo=model)
            xdb.set_trace()

    #xdb.set_trace()
    print('Done with my subset {:d}, {:d}'.format(istart,iend))
    if output is not None:
        output.put((istart,iend,pca_val))
        #output.put(None)

##
def do_sdss_lya_parallel(istart, iend, cut_Lya, output, debug=False):
    '''
    Generate PCA coeff for the SDSS DR7 dataset, 0.5<z<2

    Parameters
    ----------
    cut_Lya: boolean (True)
      Avoid using the Lya forest in the analysis
    '''
    # Eigen
    eigen, eigen_wave = read_qso_eigen()

    # Open the BOSS catalog file
    sdss_cat_fil = os.environ.get('SDSSPATH')+'/DR7_QSO/dr7_qso.fits.gz'
    bcat_hdu = fits.open(sdss_cat_fil)
    t_sdss = bcat_hdu[1].data
    nqso = len(t_sdss)

    pca_val = np.zeros((iend-istart, 4))

    if cut_Lya is False:
        print('do_sdss: Not cutting the Lya Forest in the fit')

    # Loop us -- Should spawn on multiple CPU
    #for ii in range(nqso):
    datdir =  os.environ.get('SDSSPATH')+'/DR7_QSO/spectro/1d_26/'
    jj = 0
    for ii in range(istart,iend):
        if (ii % 1000) == 0:
            print('SDSS ii = {:d}'.format(ii))
        # Spectrum file
        pnm = str(t_sdss['PLATE'][ii]).rjust(4,str('0'))
        fnm = str(t_sdss['FIBERID'][ii]).rjust(3,str('0'))
        mjd = str(t_sdss['MJD'][ii])
        sfil = datdir+pnm+'/1d/spSpec-'
        sfil = sfil+mjd+'-'+pnm+'-'+fnm+'.fit.gz'
        # Read spectrum
        spec_hdu = fits.open(sfil)
        head = spec_hdu[0].header
        iwave = head['CRVAL1']
        cdelt = head['CD1_1']

        t = spec_hdu[0].data
        flux = t[0,:]
        sig = t[2,:]
        npix = len(flux)
        wave = 10.**(iwave + np.arange(npix)*cdelt)
        ivar = np.zeros(npix)
        gd = np.where(sig>0.)[0]
        ivar[gd] = 1./sig[gd]**2
        zqso = t_sdss['z'][ii]

        wrest  = wave / (1+zqso)
        wlya = 1215. 

        # Cut Lya forest?
        if cut_Lya is True:
            Ly_imn = np.argmin(np.abs(wrest-wlya))
        else:
            Ly_imn = 0
            
        # Pack
        imn = np.argmin(np.abs(wrest[Ly_imn]-eigen_wave))
        npix = len(wrest[Ly_imn:])
        imx = npix+imn
        eigen_flux = eigen[:,imn:imx]

        # FIT
        acoeff = fit_eigen(flux[Ly_imn:], ivar[Ly_imn:], eigen_flux)
        pca_val[jj,:] = acoeff
        jj += 1

        # Check
        if debug is True:
            model = np.dot(eigen.T,acoeff)
            xdb.set_trace()
            if flg_xdb is True:
                xdb.xplot(wrest, flux, xtwo=eigen_wave, ytwo=model)
            xdb.set_trace()

    #xdb.set_trace()
    print('Done with my subset {:d}, {:d}'.format(istart,iend))
    if output is not None:
        output.put((istart,iend,pca_val))
        #output.put(None)
    
## #################################    
## #################################    
## TESTING
## #################################    
if __name__ == '__main__':

    # Run
    #do_boss_lya_parallel(0,10,None,debug=True,cut_Lya=True)
    #xdb.set_trace()

    flg = 0 # 0=BOSS, 1=SDSS

    ## ############################
    # Parallel
    if flg == 0:
        boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
        bcat_hdu = fits.open(boss_cat_fil)
        t_boss = bcat_hdu[1].data
        nqso = len(t_boss)
    elif flg == 1:
        sdss_cat_fil = os.environ.get('SDSSPATH')+'/DR7_QSO/dr7_qso.fits.gz'
        scat_hdu = fits.open(sdss_cat_fil)
        t_sdss = scat_hdu[1].data
        nqso = len(t_sdss)
        outfil = 'SDSS_DR7Lya_PCA_values_nocut.fits'

    nqso = 450  # Testing

    #do_sdss_lya_parallel(0, nqso, False, None)
    #xdb.set_trace()

    output = mp.Queue()
    processes = []
    nproc = 8
    nsub = nqso // nproc
    
    cut_Lya = False

    # Setup the Processes
    for ii in range(nproc):
        # Generate
        istrt = ii * nsub
        if ii == (nproc-1):
            iend = nqso
        else:
            iend = (ii+1)*nsub
        #xdb.set_trace()
        if flg == 0:
            process = mp.Process(target=do_boss_lya_parallel,
                                args=(istrt,iend,cut_Lya, output))
        elif flg == 1:
            process = mp.Process(target=do_sdss_lya_parallel,
                                args=(istrt,iend,cut_Lya, output))
        processes.append(process)

    # Run processes
    for p in processes:
        p.start()

    #xdb.set_trace()
    # Get process results from the output queue
    print('Grabbing Output')
    results = [output.get() for p in processes]
    xdb.set_trace()

    # Exit the completed processes
    print('Joining')
    for p in processes:
        p.join()


    # Bring together
    #sorted(results, key=lambda result: result[0])
    #all_is = [ir[0] for ir in results]
    pca_val = np.zeros((nqso, 4))
    for ir in results:
        pca_val[ir[0]:ir[1],:] = ir[2]

    # Write to disk as a binary FITS table
    col0 = fits.Column(name='PCA0',format='E',array=pca_val[:,0])
    col1 = fits.Column(name='PCA1',format='E',array=pca_val[:,1])
    col2 = fits.Column(name='PCA2',format='E',array=pca_val[:,2])
    col3 = fits.Column(name='PCA3',format='E',array=pca_val[:,3])
    cols = fits.ColDefs([col0, col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)

    prihdr = fits.Header()
    prihdr['OBSERVER'] = 'Edwin Hubble'
    prihdr['COMMENT'] = "Here's some commentary about this FITS file."
    prihdu = fits.PrimaryHDU(header=prihdr)

    thdulist = fits.HDUList([prihdu, tbhdu])
    if not (outfil in locals()):
        if cut_Lya is False:
            outfil = 'BOSS_DR10Lya_PCA_values_nocut.fits'
        else:
            outfil = 'BOSS_DR10Lya_PCA_values.fits'
    thdulist.writeto(outfil, clobber=True)

    # Done
    #xdb.set_trace()
    print('All done')
