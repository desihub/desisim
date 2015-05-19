#!/usr/bin/env python

"""
Generate figures for the simulate-templates.pdf TechNote.
"""

from __future__ import division, print_function

import os
import sys
import argparse

import numpy as np
import scipy as sci

import triangle
import seaborn as sns

from desisim.util import medxbin
from desisim.templates import read_base_templates
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--d4000', action='store_true', help='DEEP2: EW([OII]) vs D(4000)')
    parser.add_argument('--linesigma', action='store_true',
                        help='DEEP2: emission-line velocity width distribution')
    parser.add_argument('--oiiihb', action='store_true',
                        help='[OIII]/H-beta vs various line-ratios')

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    qadir = os.path.join(os.getenv('DESISIM'),'doc','tex',
                         'simulate-templates','figures')
    cflux, cwave, cmeta = read_base_templates(objtype='elg',observed=True)

    # Set sns preferences
    sns.set_context('talk', font_scale=1.0)
    sns.set_style('white')
    sns.set_palette('bright') #deep, muted, bright, pastel, dark, colorblind
    
    # Figure: DEEP2 [OII] emission-line velocity width distribution 
    if args.oiiihb:
        from astropy.io import fits

        atlas = fits.getdata(os.path.join(os.getenv('IM_PROJECTS_DIR'),'desi','data',
                                          'atlas-emlines.fits.gz'),ext=1)
        sdss = fits.getdata(os.path.join(os.getenv('IM_PROJECTS_DIR'),'desi','data',
                                          'sdss-emlines.fits.gz'),ext=1)
        hii = fits.getdata(os.path.join(os.getenv('IM_PROJECTS_DIR'),'desi','data',
                                        'hii-emlines.fits.gz'),ext=1)

        oiiihb = np.array(zip(sdss['OIIIHB'][0,:],hii['OIIIHB'][0,:])).flatten()
        oiihb = np.array(zip(sdss['OIIHB'][0,:],hii['OIIHB'][0,:])).flatten()
        niihb = np.array(zip(sdss['NIIHB'][0,:],hii['NIIHB'][0,:])).flatten()
        siihb = np.array(zip(sdss['SIIHB'][0,:],hii['SIIHB'][0,:])).flatten()

        oiiihbcoeff = sci.polyfit(oiiihb,oiihb,3)
        niihbcoeff = sci.polyfit(oiiihb,niihb,2)
        siihbcoeff = sci.polyfit(sdss['OIIIHB'][0,:],sdss['SIIHB'][0,:],2)
        #siihbcoeff = sci.polyfit(oiiihb,siihb,2)
        print(oiiihbcoeff, niihbcoeff, siihbcoeff)

        xlim = [-1.3,1.2]
        oiiihbaxis = np.linspace(-1.1,1.0,100)

        # [OIII]/Hb vs [OII]/Hb
        #fig = plt.subplots(figsize=(8,6))
        fig, ax = plt.subplots(3, sharex=True, figsize=(6,8))
        plt.tick_params(axis='both', which='major', labelsize=12)
        #fig = plt.figure(figsize=(8,6))

        ylim = [-1.5,1.0]
        sns.kdeplot(sdss['OIIIHB'][0], sdss['OIIHB'][0], range=[xlim,ylim], ax=ax[0])
        #triangle.hist2d(sdss['OIIIHB'][0], sdss['OIIHB'][0], range=[xlim,ylim],
        #                plot_density=False, ax=ax[0])
        ax[0].plot(hii['OIIIHB'][0], hii['OIIHB'][0], 's', markersize=3)

        ax[0].plot(oiiihbaxis, sci.polyval(oiiihbcoeff,oiiihbaxis), lw=3)
        ax[0].set_xlim(xlim)
        ax[0].set_ylim(ylim)
        #ax[0].xlabel(r'log$_{10}$ ([O III] $\lambda$5007 / H$\beta$)',fontsize=14)
        ax[0].set_ylabel(r'log$_{10}$ ([O II] $\lambda$3727 / H$\beta$)')
        #ax[0].tick_params(axis='both', which='major', labelsize=14)
        #fig.subplots_adjust(bottom=0.1)

        # [OIII]/Hb vs [NII]/Hb
        ylim = [-2.0,1.0]
        #triangle.hist2d(sdss['OIIIHB'][0], sdss['NIIHB'][0],range=[xlim,ylim],
        #                plot_density=False, ax=ax[1])
        sns.kdeplot(sdss['OIIIHB'][0], sdss['NIIHB'][0],range=[xlim,ylim], ax=ax[1])
        ax[1].plot(hii['OIIIHB'][0], hii['NIIHB'][0], 's', markersize=3)
    
        ax[1].plot(oiiihbaxis, sci.polyval(niihbcoeff,oiiihbaxis), lw=3)
        ax[1].set_xlim(xlim)
        ax[1].set_ylim(ylim)
        ax[1].set_ylabel(r'log$_{10}$ ([N II] $\lambda$6584 / H$\beta$)')

        # [OIII]/Hb vs [SII]/Hb
        ylim = [-1.0,0.5]
        #triangle.hist2d(sdss['OIIIHB'][0], sdss['SIIHB'][0],range=[xlim,ylim],
        #                plot_density=False, ax=ax[2])
        sns.kdeplot(sdss['OIIIHB'][0], sdss['SIIHB'][0],range=[xlim,ylim], ax=ax[2])
        ax[2].plot(hii['OIIIHB'][0], hii['SIIHB'][0], 's', markersize=3)
        ax[2].plot(oiiihbaxis, sci.polyval(siihbcoeff,oiiihbaxis), lw=3)
        ax[2].set_xlim(xlim)
        ax[2].set_ylim(ylim)
        ax[2].set_xlabel(r'log$_{10}$ ([O III] $\lambda$5007 / H$\beta$)')
        ax[2].set_ylabel(r'log$_{10}$ ([S II] $\lambda$6716,31 / H$\beta$)')

        fig.subplots_adjust(left=0.2)
        fig.savefig(qadir+'/oiiihb.pdf')
        
    # Figure: DEEP2 [OII] emission-line velocity width distribution 
    if args.linesigma:
        from astropy.modeling import models, fitting
        sigma = cmeta['SIGMA_KMS']
        sigminmax = np.log10([8.0,500.0])
        binsz = 0.05
        nbin = long((sigminmax[1]-sigminmax[0])/binsz)
        ngal, bins = np.histogram(np.log10(sigma), bins=nbin, range=sigminmax)
        cbins = (bins[:-1] + bins[1:]) / 2.0
        #ngal, bins, patches = plt.hist(sigma, bins=30)

        ginit = models.Gaussian1D(ngal.max(), mean=70.0, stddev=20.0)
        gfit = fitting.LevMarLSQFitter()
        gauss = gfit(ginit,cbins,ngal)
        
        xgauss = np.linspace(sigminmax[0],sigminmax[1],nbin*10)
        fig = plt.figure(figsize=(8,6))
        plt.bar(10**cbins, ngal, align='center', width=binsz*np.log(10)*10**cbins),
        plt.plot(10**xgauss, gauss(xgauss), '-', lw=3, color='black')
        plt.xlim(10**sigminmax)
        plt.xlabel('log$_{10}$ Emission-line Velocity Width $\sigma$ (km/s)',fontsize=18)
        plt.ylabel('Number of Galaxies',fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.text(0.95,0.9,'$<log_{10}$'+' $\sigma>$ = '+
                 '{:.3f}$\pm${:.3f} km/s'.format(gauss.mean.value,gauss.stddev.value),
                 horizontalalignment='right',color='black',
                 transform=plt.gca().transAxes, fontsize=18)

        fig.subplots_adjust(bottom=0.1)
        fig.savefig(qadir+'/linesigma.pdf')
        
    # Figure: DEEP2 EW([OII]) vs D(4000)
    if args.d4000:
        d4000 = cmeta['D4000']
        ewoii = np.log10(cmeta['OII_3727_EW'])
        bins, stats = medxbin(d4000,ewoii,0.05,minpts=20,xmin=1.0,xmax=1.7)
        
        coeff = sci.polyfit(bins,stats['median'],2)
        #print(coeff,len(bins))
        #print(stats['median'], stats['sigma'], np.mean(stats['sigma']))
        
        strcoeff = []
        for cc in coeff: strcoeff.append('{:.4f}'.format(cc))
        
        plt.ioff()
        fig = plt.figure(figsize=(8,6))
        hist2d = triangle.hist2d(d4000, ewoii, plot_density=False)
        plt.xlim([0.9,1.8])
        plt.xlabel('D$_{n}$(4000)', fontsize=18)
        plt.ylabel('log$_{10}$ EW([O II] $\lambda\lambda3726,29$) ($\AA$, rest-frame)',
                   fontsize=18)
        plt.text(0.95,0.9,'{'+','.join(strcoeff)+'}',horizontalalignment='right',color='black',
                 transform=plt.gca().transAxes, fontsize=18)
        plt.errorbar(bins,stats['median'],yerr=stats['sigma'],color='dodgerblue',fmt='o',markersize=8)
        plt.plot(bins,sci.polyval(coeff,bins),color='red')
        #hist2d = triangle.hist2d(np.array([d4000,ewoii]).transpose())
        #plt.plot(d4000,ewoii,'bo')
        fig.savefig(qadir+'/d4000_ewoii.pdf')
    
if __name__ == '__main__':
    main()



#        atlas = fits.getdata(os.path.join(os.getenv('IM_PROJECTS_DIR'),'desi','data',
#                                          'atlas_specdata_solar_drift_v1.0.fits.gz'),ext=1)
#        sdss = fits.getdata(os.path.join(os.getenv('IM_PROJECTS_DIR'),'desi','data',
#                                          'ispecline.dr72bsafe0.fits.gz'),ext=1)
#
#        def getlines(cat,snr=3.0):
#            keep = np.array(np.where((np.array(zip(
#                (cat['OIII_5007'][:,0]/cat['OIII_5007'][:,1])>snr,
#                (cat['OII_3727'][:,0]/cat['OII_3727'][:,1])>snr,
#                (cat['H_ALPHA'][:,0]/cat['H_ALPHA'][:,1])>snr,
#                (cat['NII_6584'][:,0]/cat['NII_6584'][:,1])>snr,
#                (cat['SII_6716'][:,0]/cat['SII_6716'][:,1])>snr,
#                (cat['SII_6731'][:,0]/cat['SII_6731'][:,1])>snr,
#                (cat['H_BETA'][:,0]/cat['H_BETA'][:,1])>snr)).all(axis=1))==True))
#            oiiihb = np.log10(cat['OIII_5007'][keep,0]/cat['H_BETA'][keep,0])
#            oiihb = np.log10(cat['OII_3727'][keep,0]/cat['H_BETA'][keep,0])
#            return oiiihb, oiihb
#        oiiihb_atlas, oiihb_atlas = getlines(atlas)
#        oiiihb_sdss, oiihb_sdss = getlines(sdss)
