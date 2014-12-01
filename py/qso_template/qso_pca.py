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
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import os

from astropy.io import fits
from xastropy.xutils import xdebug as xdb

"""
## Tinkering about
#   JXP on 01 Dec 2014

hdu = fits.open('spEigenQSO-55732.fits')
eigen = hdu[0].data  # There are 4 eigenvectors
wav = 10.**(2.6534 + np.arange(13637)*0.0001)  # Rest-frame, Ang
xdb.xplot(wav,eigen[0,:])
  # Looks sensible enough

# QSO sample used to generate the PCA eigenvectors
qsos = hdu[1].data
xdb.xhist(qsos['REDSHIFT']) # Skewed to high z
"""


"""
Let's test in IDL on a BOSS spectrum

cd /Users/xavier/BOSS/DR10/BOSSLyaDR10_spectra_v2.1/5870

IDL> boss_objinf, [5870,1000], /plot
    Name             RA(2000)   DEC(2000) Mag(R) zobj
5870-56065-100     14:07:37.28 +21:48:38.5 21.06 2.180

Looks like a fine quasar

Read the eigenspectra

eigen = xmrdfits(getenv('IDLSPEC2D_DIR')+'/templates/spEigenQSO-55732.fits',0,head)
eigen_wave = 10.^(sxpar(head,'COEFF0') + dindgen(sxpar(head,'NAXIS1'))*sxpar(head,'COEFF1'))


BOSS spectrum

boss_readspec, 'speclya-5870-56065-1000.fits', flux, wave, ivar=ivar
zqso = 2.180
wrest  = wave / (1+zqso)
npix = n_elements(wrest)

Are they 'registered'?
mn = min(abs(wrest[0]-eigen_wave),imn)
x_splot, findgen(npix), flux, ytwo=eigen[imn:*,0]/5292*4.4
  Yes, they are..

Build the inputs to computechi2

objflux = flux
sqivar = ivar
imx = npix+imn-1
starflux = eigen[imn:imx,*]

chi2 = computechi2(objflux, sqivar, starflux, acoeff=acoeff, yfit=yfit)
  ;; The fit looks ok

IDL> print, acoeff
   0.00045629701  -0.00034376233   0.00047730298   0.00045008259
"""


# Read Eigenspectrum

hdu = fits.open(os.environ.get('IDLSPEC2D_DIR')+'/templates/spEigenQSO-55732.fits')
eigen = hdu[0].data
head = hdu[0].header
eigen_wave = 10.**(head['COEFF0'] + np.arange(head['NAXIS1'])*head['COEFF1'])

#xdb.xplot(eigen_wave, eigen[0,:])
#xdb.set_trace()

#BOSS spectrum

boss_hdu = fits.open('/Users/xavier/BOSS/DR10/BOSSLyaDR10_spectra_v2.1/5870/speclya-5870-56065-1000.fits.gz')
t = boss_hdu[1].data
flux = t['flux']
wave = 10.**t['loglam']
ivar = t['ivar']

zqso = 2.180
wrest  = wave / (1+zqso)
npix = len(wrest)
imx = npix+imn

imn = np.argmin(np.abs(wrest[0]-eigen_wave))
eigen_flux = eigen[:,imn:imx]


# Generate the matrices
"""
Now Python with simple approach following Dan F-M http://dan.iel.fm/emcee/current/user/line/
"""

C = np.diag(1./ivar)
A = eigen_flux.T
y = flux

cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
acoeff = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

#In [48]: acoeff
#Out[48]: array([ 0.00045461, -0.00012764,  0.00010843,  0.0003281 ], dtype=float32)

model = np.dot(A,acoeff)
xdb.xplot(wrest, flux, model)
  # Looks good to me
