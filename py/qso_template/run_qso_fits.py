"popen_ex.py"
from subprocess import Popen
import os
from astropy.io import fits

flg = 1 # SDSS
nproc = 4

## ############################
# "Parallel"
if flg == 0:
    print('Running BOSS!')
    boss_cat_fil = os.environ.get('BOSSPATH')+'/DR10/BOSSLyaDR10_cat_v2.1.fits.gz'
    bcat_hdu = fits.open(boss_cat_fil)
    t_boss = bcat_hdu[1].data
    nqso = len(t_boss)
    outroot = 'Output/BOSS_DR10Lya_PCA_values_nocut'
elif flg == 1:
    print('Running SDSS!')
    sdss_cat_fil = os.environ.get('SDSSPATH')+'/DR7_QSO/dr7_qso.fits.gz'
    scat_hdu = fits.open(sdss_cat_fil)
    t_sdss = scat_hdu[1].data
    nqso = len(t_sdss)
    outroot = 'Output/SDSS_DR7Lya_PCA_values_nocut'

nqso = 20000  # Testing
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

    outfil = outroot+str(ii)+'.fits'
    Popen(['python', './fit_boss_qsos.py', str(flg), str(istrt), str(iend), outfil])

