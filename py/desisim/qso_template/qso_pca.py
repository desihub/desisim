"""
#;+ 
#; NAME:
#; qso_pca
#;    Version 1.0
#;
#; PURPOSE:
#;    Module for generate QSO PCA templates
#;   24-Nov-2014 by JXP
#;-
#;------------------------------------------------------------------------------
"""
from __future__ import print_function, absolute_import, division

import numpy as np
import os
import multiprocessing as mp

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
def do_boss_lya_parallel(istart, iend, output, debug=False, cut_Lya=True):
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

    # Loop us -- Should spawn on multiple CPU
    #for ii in range(nqso):
    datdir =  os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_spectra_v2.1/'
    jj = 0
    for ii in range(istart,iend):
        if (ii % 100) == 0:
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
    if output is not None:
        output.put((istart,iend,pca_val))
    
## #################################    
## #################################    
## TESTING
## #################################    
if __name__ == '__main__':

    # Run
    #do_boss_lya_parallel(0,10,None,debug=True,cut_Lya=True)
    #xdb.set_trace()

    ## ############################
    # Parallel
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    nqso = len(t_boss)
    nqso = 45  # Testing

    output = mp.Queue()
    processes = []
    nproc = 4
    nsub = nqso // nproc
    # Setup the Processes
    for ii in range(nproc):
        # Generate
        istrt = ii * nsub
        if ii == (nproc-1):
            iend = nqso
        else:
            iend = (ii+1)*nsub
        #xdb.set_trace()
        process = mp.Process(target=do_boss_lya_parallel,
                               args=(istrt,iend,output))
        processes.append(process)

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]

    # Bring together
    #sorted(results, key=lambda result: result[0])
    #all_is = [ir[0] for ir in results]
    pca_val = np.zeros((nqso, 4))
    for ir in results:
        pca_val[ir[0]:ir[1],:] = ir[2]
    xdb.set_trace()
