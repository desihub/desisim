"""
desisim.spec_qa.high_level
============

Module to run high_level QA on a given DESI run
 Written by JXP on 3 Sep 2015
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import sys, os, pdb, glob

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy.table import Table, vstack, hstack, MaskedColumn

from desispec import util
from desispec.io import read_fiberflat
from desispec import interpolation as desi_interp

from desispec.io import frame as desi_io_frame
from desispec.io import fibermap as desi_io_fmap
from desispec.io import read_sky
from desispec.io import meta as dio_meta
from desispec import sky as dspec_sky

from desispec.fiberflat import apply_fiberflat

def calc_dz(simz_tab):
    '''Calcualte deltaz/(1+z) for a given simz_tab
    '''
    dz = (simz_tab['REDM_Z']-simz_tab['REDSHIFT'])/(1+simz_tab['REDSHIFT'])
    #
    return dz

def calc_dzsig(simz_tab):
    '''Calcualte deltaz/sig(z) for a given simz_tab
    '''
    dzsig = (simz_tab['REDM_Z']-simz_tab['REDSHIFT'])/simz_tab['REDM_ZERR']
    #
    return dzsig

def elg_flux_lim(z, oii_flux):
    '''Assess which objects pass the ELG flux limit
    Uses DESI document 318 from August 2014

    Parameters:
    -----------
    z: ndarray
      ELG redshifts
    oii_flux: ndarray
      [OII] fluxes
    '''
    # Init mask
    mask = np.array([False]*len(z))
    # 
    # Flux limits from document
    zmin = 0.6
    zstep = 0.2
    OII_lim = np.array([10., 9., 9., 9., 9])*1e-17
    # Get bin
    zbin = (z-0.6)/zstep
    gd = np.where((zbin>0.) & (zbin<len(OII_lim)) &
        (oii_flux>0.))[0]
    # Fill gd
    OII_match = np.zeros(len(gd))
    for igd in gd:
        OII_match[igd] = OII_lim[int(zbin[igd])]
    # Query
    gd_gd = np.where(oii_flux[gd] > OII_match)[0]
    mask[gd_gd] = True
    # Return
    return mask

def calc_stats(simz_tab, objtype, flux_lim=True):
    '''Calculate redshift statistics for a given objtype

    Parameters:
    -----------
    objtype: str
      Object type, e.g. 'ELG', 'LRG'
    flux_lim: bool, optional
      Impose the flux limit for ELGs? [True] 
    '''
    # Cut on targets that were analyzed by RedMonster

    # Init
    stat_dict = {} #dict(OBJTYPE=objtype)

    # N targets 
    obj_tab = slice_simz(simz_tab, objtype=objtype)
    stat_dict['NTARG'] = len(obj_tab)

    # Number of objects with RedMonster
    zobj_tab = slice_simz(simz_tab,objtype=objtype,redm=True)
    stat_dict['N_RM'] = len(zobj_tab) 

    # Redshift measured (includes catastrophics)
    #   For ELGs, cut on OII_Flux too
    survey_tab = slice_simz(simz_tab,objtype=objtype,survey=True)
    stat_dict['N_SURVEY'] = len(survey_tab) 

    # Catastrophic failures
    cat_tab = slice_simz(simz_tab,objtype=objtype,
        survey=True,catastrophic=True)
    stat_dict['NCAT'] = len(cat_tab)
    stat_dict['CAT_RATE'] = len(cat_tab)/stat_dict['N_SURVEY']

    # Good redshifts
    gdz_tab = slice_simz(simz_tab,objtype=objtype,
        survey=True,good=True)
    stat_dict['N_GDZ'] = len(gdz_tab)

    # Efficiency
    stat_dict['EFF'] = float(len(gdz_tab))/float(stat_dict['N_SURVEY'])

    # delta z
    dz = calc_dz(gdz_tab)
    stat_dict['MEAN_DZ'] = float(np.mean(dz))
    stat_dict['MEDIAN_DZ'] = float(np.median(dz))
    stat_dict['RMS_DZ'] = float(np.std(dz))

    # Return
    return stat_dict

def catastrophic_dv(objtype):
    '''Pass back catastrophic velocity limit for given objtype
    From DESI document 318 (August 2014) in docdb

    Parameters:
    -----------
    objtype: str
      Object type, e.g. 'ELG', 'LRG'
    '''
    cat_dict = dict(ELG=1000., LRG=1000., QSO_L=1000., QSO_T=1000.)
    #
    return cat_dict[objtype]

def get_sty_otype():
    '''Styles for plots'''
    sty_otype = dict(ELG={'color':'green', 'lbl':'ELG'},
        LRG={'color':'red', 'lbl':'LRG'},
        QSO={'color':'blue', 'lbl':'QSO'},
        QSO_L={'color':'blue', 'lbl':'QSO z>2.1'},
        QSO_T={'color':'cyan', 'lbl':'QSO z<2.1'})
    return sty_otype

def load_z(fibermap_files, zbest_files, outfil=None):
    '''Load input and output redshift values for a set of exposures

    Parameters:
    -----------
    fibermap_files: list
      List of fibermap files
    zbest_files: list
      List of zbest output files from Redmonster
    outfil: str, optional
      Output file for the table

    Returns:
    -----------
    simz_tab: Table  
      Table of target info including Redmonster redshifts
    '''
    # imports

    # Init

    # Load up fibermap and simspec tables
    fbm_tabs = []
    sps_tabs = []
    for fibermap_file in fibermap_files:
        fbm_hdu = fits.open(fibermap_file)
        print('Reading: {:s}'.format(fibermap_file))
        # Load simspec
        simspec_fil = fibermap_file.replace('fibermap','simspec')
        sps_hdu = fits.open(simspec_fil)
        # Make Tables
        assert fbm_hdu[1].name == 'FIBERMAP'
        fbm_tabs.append(Table(fbm_hdu[1].data))
        assert sps_hdu[2].name == 'METADATA'
        sps_tabs.append(Table(sps_hdu[2].data))
    # Stack
    fbm_tab = vstack(fbm_tabs)
    sps_tab = vstack(sps_tabs)
    del fbm_tabs, sps_tabs

    # Drop to unique
    univ, uni_idx = np.unique(np.array(fbm_tab['TARGETID']),return_index=True)
    fbm_tab = fbm_tab[uni_idx]
    sps_tab = sps_tab[uni_idx]

    # Combine + Sort
    simz_tab = hstack([fbm_tab,sps_tab],join_type='exact')
    simz_tab.sort('TARGETID')
    nsim = len(simz_tab)

    # Rename QSO
    qsol = np.where( (simz_tab['OBJTYPE_1'] == 'QSO') & 
        (simz_tab['REDSHIFT'] >= 2.1))[0]
    simz_tab['OBJTYPE_1'][qsol] = 'QSO_L'
    qsot = np.where( (simz_tab['OBJTYPE_1'] == 'QSO') & 
        (simz_tab['REDSHIFT'] < 2.1))[0]
    simz_tab['OBJTYPE_1'][qsot] = 'QSO_T'

    # Load up zbest files
    zb_tabs = []
    for zbest_file in zbest_files:
        zb_hdu = fits.open(zbest_file)
        zb_tabs.append(Table(zb_hdu[1].data))
    # Stack
    zb_tab = vstack(zb_tabs)
    univ, uni_idx = np.unique(np.array(zb_tab['TARGETID']),return_index=True)
    zb_tab = zb_tab[uni_idx]

    # Match up
    sim_id = np.array(simz_tab['TARGETID'])
    z_id = np.array(zb_tab['TARGETID'])
    inz = np.in1d(z_id,sim_id,assume_unique=True)
    ins = np.in1d(sim_id,z_id,assume_unique=True)

    z_idx = np.arange(z_id.shape[0])[inz]
    sim_idx = np.arange(sim_id.shape[0])[ins]
    assert np.array_equal(sim_id[sim_idx],z_id[z_idx])

    # Fill up
    ztags = ['Z','ZERR','ZWARN','TYPE']
    new_tags = ['REDM_Z','REDM_ZERR','REDM_ZWARN','REDM_TYPE']
    new_clms = []
    mask = np.array([True]*nsim)
    mask[sim_idx] = False
    for kk,ztag in enumerate(ztags):
        # Generate a MaskedColumn
        new_clm  = MaskedColumn([zb_tab[ztag][z_idx[0]]]*nsim,
            name=new_tags[kk], mask=mask)
        # Fill
        new_clm[sim_idx] = zb_tab[ztag][z_idx]
        # Append
        new_clms.append(new_clm)
    # Add columns
    simz_tab.add_columns(new_clms)

    # Write?
    if outfil is not None:
        simz_tab.write(outfil,overwrite=True)
    # Return
    return simz_tab # Masked Table

def obj_requirements(zstats, objtype):
    '''Assess where a given objtype passes the requirements
    Requirements from Doc 318 (August 2014)

    Parameters:
    -----------
    objtype: str
      Object type, e.g. 'ELG', 'LRG'
    Returns:
    -------------
    pf_dict: dict  
      Pass/fail dict
    '''
    pf_dict = {}
    # 
    all_dict=dict(ELG={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.90},
        LRG={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.95},
        QSO_T={'RMS_DZ':0.0025, 'MEAN_DZ': 0.0004, 'CAT_RATE': 0.05, 'EFF': 0.90},
        QSO_L={'RMS_DZ':0.0025, 'CAT_RATE': 0.02, 'EFF': 0.90})
    req_dict = all_dict[objtype]

    tst_fail = ''
    passf = 'PASS'
    for key in req_dict.keys():
        ipassf = 'PASS'
        if key in ['EFF']: # Greater than requirement
            if zstats[key] < req_dict[key]:
                ipassf = 'FAIL'
                tst_fail = tst_fail+key+'-'
        else:
            if zstats[key] > req_dict[key]:
                ipassf = 'FAIL'
                tst_fail = tst_fail+key+'-'
        # Update
        pf_dict[key] = ipassf
        if ipassf == 'FAIL':
            passf = 'FAIL'
    if passf == 'FAIL':
        tst_fail = tst_fail[:-1]
        print('OBJ={:s} failed tests {:s}'.format(objtype,tst_fail))
    #
    #pf_dict['FINAL'] = passf
    return pf_dict, passf

def slice_simz(simz_tab, objtype=None, redm=False, survey=False, 
    catastrophic=False, good=False):
    '''Slice input simz_tab in one of many ways
    Parameters:
    ----------
    redm: bool, optional
      RedMonster analysis required?
    '''
    # Init
    nrow = len(simz_tab)

    # Object type
    if objtype is None:
        objtype_mask = np.array([True]*nrow)
    else:
        objtype_mask = simz_tab['OBJTYPE_1'] == objtype
    # RedMonster analysis
    if redm:
        redm_mask = simz_tab['REDM_Z'].mask == False # Not masked in Table
    else:
        redm_mask = np.array([True]*nrow)
    # Survey
    if survey:
        survey_mask = (simz_tab['REDM_Z'].mask == False)
        # Flux limit
        elg = np.where((simz_tab['OBJTYPE_1']=='ELG') & survey_mask)[0]
        elg_mask = elg_flux_lim(simz_tab['REDSHIFT'][elg], 
            simz_tab['O2FLUX'][elg])
        # Update
        survey_mask[elg[~elg_mask]] = False
    else:
        survey_mask = np.array([True]*nrow)
    # Catastrophic/Good (This gets messy...)
    if (catastrophic or good):
        if catastrophic:
            catgd_mask = np.array([False]*nrow)
        else:
            catgd_mask = simz_tab['REDM_ZWARN']==0
        for obj in ['ELG','LRG','QSO_L','QSO_T']:
            dv = catastrophic_dv(obj) # km/s
            omask = np.where((simz_tab['OBJTYPE_1'] == obj)&
                (simz_tab['REDM_ZWARN']==0))[0]
            dz = calc_dz(simz_tab[omask]) # dz/1+z
            cat = np.where(np.abs(dz)*3e5 > dv)[0]
            # Update
            if catastrophic:
                catgd_mask[omask[cat]] = True
            else:
                catgd_mask[omask[cat]] = False
    else:
        catgd_mask = np.array([True]*nrow) 

    # Final mask
    mask = objtype_mask & redm_mask & survey_mask & catgd_mask

    # Return
    return simz_tab[mask]

def obj_fig(simz_tab, objtype, summ_stats, outfil=None, pp=None):
    """Generate QA plot for a given object type
    Parameters:
    simz_tab
    """
    from scipy.stats import norm, chi2

    #sobj = np.where(summ_stats['OBJTYPE'] == objtype)[0][0]

    # Plot
    sty_otype = get_sty_otype()
    fig = plt.figure(figsize=(8, 6.0))
    gs = gridspec.GridSpec(2,2)
    fig.suptitle('{:s}: Summary'.format(sty_otype[objtype]['lbl']), 
        fontsize='large')

    # Title 

    survey_tab = slice_simz(simz_tab,objtype=objtype,survey=True)
    gdz_tab = slice_simz(simz_tab,objtype=objtype, survey=True,good=True)

    # Offset
    for kk in range(4):
        yoff = 0.
        ax= plt.subplot(gs[kk])
        if kk == 0:
            yval = calc_dzsig(gdz_tab)
            ylbl = (r'$(z_{\rm red}-z_{\rm true}) / \sigma(z)$')
            ylim = 5.
            # Stats
            rms = np.std(yval)
            redchi2 = np.sum(yval**2)/len(yval)
            # 
            xtxt = 0.05
            ytxt = 1.0
            for req_tst in ['EFF','CAT_RATE']:
                ytxt -= 0.12
                if summ_stats[objtype]['REQ_INDIV'][req_tst] == 'FAIL':
                    tcolor='red'
                else:
                    tcolor='green'
                ax.text(xtxt, ytxt, '{:s}: {:.3f}'.format(req_tst, 
                    summ_stats[objtype][req_tst]), color=tcolor,
                    transform=ax.transAxes, ha='left', fontsize='small')
            # Additional
            ytxt -= 0.12
            ax.text(xtxt, ytxt, '{:s}: {:.3f}'.format('RMS:',
                rms), color='black', transform=ax.transAxes, 
                ha='left', fontsize='small')
            ytxt -= 0.12
            ax.text(xtxt, ytxt, '{:s}: {:.3f}'.format(r'$\chi^2_\nu$:',
                redchi2), color='black', transform=ax.transAxes, 
                ha='left', fontsize='small')
        else:
            yval = calc_dz(gdz_tab)
            ylbl = (r'$(z_{\rm red}-z_{\rm true}) / (1+z)$')
            ylim = 5.*summ_stats[objtype]['RMS_DZ'] 
            if (np.median(summ_stats[objtype]['MEDIAN_DZ']) > 
                summ_stats[objtype]['RMS_DZ']):
                yoff = summ_stats[objtype]['MEDIAN_DZ']

        if kk==1:
            # Stats
            xtxt = 0.05
            ytxt = 1.0
            dx = ((ylim/2.)//0.0001 +1)*0.0001
            #import pdb
            #pdb.set_trace()
            ax.xaxis.set_major_locator(plt.MultipleLocator(dx))
            for stat in ['RMS_DZ','MEAN_DZ', 'MEDIAN_DZ']:
                ytxt -= 0.12
                try:
                    pfail = summ_stats[objtype]['REQ_INDIV'][stat] 
                except KeyError:
                    tcolor='black'
                else:
                    if pfail == 'FAIL':
                        tcolor='red'
                    else:
                        tcolor='green'
                ax.text(xtxt, ytxt, '{:s}: {:.5f}'.format(stat, 
                    summ_stats[objtype][stat]), color=tcolor,
                    transform=ax.transAxes, ha='left', fontsize='small')
        # Histogram
        if kk < 2:
            binsz = ylim/10.
            i0, i1 = int( np.min(yval) / binsz) - 1, int( np.max(yval) / binsz) + 1
            rng = tuple( binsz*np.array([i0,i1]) )
            nbin = i1-i0
            # Histogram
            hist, edges = np.histogram(yval, range=rng, bins=nbin)
            xhist = (edges[1:] + edges[:-1])/2.
            #ax.hist(xhist, color='black', bins=edges, weights=hist)#, histtype='step')
            ax.hist(xhist, color=sty_otype[objtype]['color'], bins=edges, weights=hist)#, histtype='step')
            ax.set_xlabel(ylbl)
            ax.set_xlim(-ylim, ylim)

        else:
            if kk == 2:
                lbl = r'$z_{\rm true}$'
                xval = gdz_tab['REDSHIFT']
                xmin,xmax=np.min(xval),np.max(xval)
                dx = np.maximum(1,(xmax-xmin)//0.5)*0.1
                ax.xaxis.set_major_locator(plt.MultipleLocator(dx))
                #xmin,xmax=0.6,1.65
            elif kk == 3:
                if objtype == 'ELG':
                    lbl = r'[OII] Flux ($10^{-16}$)'
                    xval = gdz_tab['O2FLUX']*1e16
                    xmin,xmax=0.3,20
                    ax.set_xscale("log", nonposy='clip')
                else:
                    lbl = '{:s} (Mag)'.format(gdz_tab[0]['FILTER'][0])
                    xval = gdz_tab['MAG'][:,0]
                    xmin,xmax=np.min(xval),np.max(xval)
            # Labels
            ax.set_xlabel(lbl)
            ax.set_xlim(xmin,xmax)
            ax.set_ylabel(ylbl)
            ax.set_ylim(-ylim+yoff, ylim+yoff)

            # Points
            ax.plot([xmin,xmax], [0.,0], '--', color='gray')
            ax.scatter(xval, yval, marker='o', s=1, label=objtype, 
                color=sty_otype[objtype]['color'])

    # Finish
    plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.3)
    plt.subplots_adjust(top=0.92)
    if outfil is not None:
        plt.savefig(outfil)
    else:
        if pp is not None:
            pp.savefig()
        plt.close()

def summ_fig(simz_tab, summ_tab, meta, outfil=None, pp=None):
    '''Generate summary summ_fig
    '''
    # Plot
    sty_otype = get_sty_otype()
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(3,2)

    # RedMonster objects
    zobj_tab = slice_simz(simz_tab,redm=True)
    otypes = ['ELG','LRG','QSO_L','QSO_T']

    # z vs. z plot
    jj=0
    ax= plt.subplot(gs[0:2,jj])

    # Catastrophic
    cat_tab = slice_simz(simz_tab,survey=True, catastrophic=True)
    ax.scatter(cat_tab['REDSHIFT'], cat_tab['REDM_Z'],
        marker='x', s=9, label='CAT', color='red')

    notype = []
    for otype in otypes: 
        gd_o = np.where(zobj_tab['OBJTYPE_1']==otype)[0]
        notype.append(len(gd_o))
        ax.scatter(zobj_tab['REDSHIFT'][gd_o], zobj_tab['REDM_Z'][gd_o], 
            marker='o', s=1, label=sty_otype[otype]['lbl'], color=sty_otype[otype]['color'])
    ax.set_ylabel(r'$z_{\rm red}$')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_xlim(-0.1, 1.02*np.max(np.array([np.max(zobj_tab['REDSHIFT']),
        np.max(zobj_tab['REDM_Z'])])))
    ax.set_ylim(-0.1, np.max(np.array([np.max(zobj_tab['REDSHIFT']),
        np.max(zobj_tab['REDM_Z'])])))
    # Legend
    legend = ax.legend(loc='upper left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Zoom
    jj=1
    ax= plt.subplot(gs[0:2,jj])

    for otype in otypes: 
        # Grab
        gd_o = np.where(zobj_tab['OBJTYPE_1']==otype)[0]
        # Stat
        dz = calc_dz(zobj_tab[gd_o]) 
        ax.scatter(zobj_tab['REDSHIFT'][gd_o], dz, marker='o', 
            s=1, label=sty_otype[otype]['lbl'], color=sty_otype[otype]['color'])

    #ax.set_xlim(xmin, xmax)
    ax.set_ylabel(r'$(z_{\rm red}-z_{\rm true}) / (1+z)$')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_xlim(0.,4)
    deltaz = 0.002
    ax.set_ylim(-deltaz/2,deltaz)

    # Legend
    legend = ax.legend(loc='lower right', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Meta text
    ax= plt.subplot(gs[2,0])
    ax.set_axis_off()
    # Meta
    xlbl = 0.1
    ylbl = 0.85
    ax.text(xlbl, ylbl, 'PRODNAME: {:s}'.format(meta['PRODNAME']), transform=ax.transAxes, ha='left')
    yoff=0.15
    for key in meta.keys():
        if key == 'PRODNAME':
            continue
        ylbl -= yoff
        ax.text(xlbl+0.1, ylbl, key+': {:s}'.format(meta[key]), 
            transform=ax.transAxes, ha='left', fontsize='small')

    # Target stats
    ax= plt.subplot(gs[2,1])
    ax.set_axis_off()
    xlbl = 0.1
    ylbl = 0.85
    ax.text(xlbl, ylbl, 'Targets', transform=ax.transAxes, ha='left')
    yoff=0.15
    for jj,otype in enumerate(otypes):
        ylbl -= yoff
        gd_o = simz_tab['OBJTYPE_1']==otype
        ax.text(xlbl+0.1, ylbl, sty_otype[otype]['lbl']+': {:d} ({:d})'.format(np.sum(gd_o),notype[jj]),
            transform=ax.transAxes, ha='left', fontsize='small')

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.1)
    if outfil is not None:
        plt.savefig(outfil)
    else:
        if pp is not None:
            pp.savefig()
        plt.close()



def summ_stats(simz_tab, outfil=None):
    '''Generate summary stats 

    Parameters:
    -----------
    simz_tab: Table
      Table summarizing redshifts

    Returns:
    ---------
    summ_stats: List
      List of summary stat dicts
    '''
    otypes = ['ELG','LRG', 'QSO_L', 'QSO_T']  # WILL HAVE TO DEAL WITH QSO_TRACER vs QSO_LYA
    summ_dict = {}

    rows = []
    for otype in otypes:
        # Calculate stats
        stat_dict = calc_stats(simz_tab, otype)
        summ_dict[otype] = stat_dict
        # Check requirements
        summ_dict[otype]['REQ_INDIV'], passf = obj_requirements(stat_dict,otype)
        summ_dict[otype]['REQ_FINAL'] = passf

    # Generate Table
    #stat_tab = Table(rows=rows)

    # Return
    return summ_dict
    #return stat_tab 


