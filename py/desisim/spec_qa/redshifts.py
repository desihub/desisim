"""
desisim.spec_qa.redshifts
=========================

Module to run high_level QA on a given DESI run

Written by JXP on 3 Sep 2015
"""
from __future__ import print_function, absolute_import, division

import matplotlib
# matplotlib.use('Agg')

import numpy as np
import sys, os, pdb, glob

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy.table import Table, vstack, hstack, MaskedColumn, join

import desispec.io
from .utils import elg_flux_lim, get_sty_otype, catastrophic_dv, match_otype

from desiutil.log import get_logger, DEBUG


def calc_dz(simz_tab):
    '''Calcualte deltaz/(1+z) for a given simz_tab
    '''
    dz = (simz_tab['Z']-simz_tab['TRUEZ'])/(1+simz_tab['TRUEZ'])
    #
    return dz


def calc_dzsig(simz_tab):
    '''Calcualte deltaz/sig(z) for a given simz_tab
    '''
    dzsig = (simz_tab['Z']-simz_tab['TRUEZ'])/simz_tab['ZERR']
    #
    return dzsig


def calc_obj_stats(simz_tab, objtype):
    """Calculate redshift statistics for a given objtype

    Parameters
    ----------
    simz_tab : Table
        TODO: document this
    objtype : str
        Object type, e.g. 'ELG', 'LRG'

    Returns
    -------
    stat_dict : dict
        Survey results for a given object type
    """
    # zstats
    ngood, nfail, nmiss, nlost = zstats(simz_tab, objtype=objtype, count=True, survey=True)
    ntot = ngood+nfail+nmiss+nlost

    # Grab the masks
    objtype_mask, z_mask, survey_mask, dv_mask, zwarn_mask = criteria(simz_tab, objtype=objtype)

    # Init
    stat_dict = {} #dict(OBJTYPE=objtype)

    # N targets (irrespective of the Survey)
    stat_dict['N_TARG'] = ntot

    # Number of objects with Redshift Analysis
    stat_dict['N_zA'] = np.count_nonzero(z_mask & objtype_mask)

    # Redshift measured (includes catastrophics)
    #   For ELGs, cut on OII_Flux too
    stat_dict['N_SURVEY'] = np.count_nonzero(survey_mask & objtype_mask & z_mask)

    # Catastrophic failures in the survey
    stat_dict['N_CAT'] = nfail
    if stat_dict['N_SURVEY'] > 0:
        stat_dict['CAT_RATE'] = float(nfail)/stat_dict['N_SURVEY']
    else:
        stat_dict['CAT_RATE'] = 0

    # Good redshifts in the survey
    stat_dict['N_GOODZ'] = ngood

    # Redshift with ZWARN=0 in the survey
    stat_dict['N_ZWARN0'] = ngood+nfail

    # Survey Efficiency
    if stat_dict['N_SURVEY'] > 0:
        stat_dict['EFF'] = float(ngood)/float(stat_dict['N_SURVEY'])
    else:
        stat_dict['EFF'] = 1.

    # Purity
    if stat_dict['N_ZWARN0'] > 0:
        stat_dict['PURITY'] = float(ngood)/float(stat_dict['N_ZWARN0'])
    else:
        stat_dict['PURITY'] = 1.

    # delta z
    gdz_tab = slice_simz(simz_tab, objtype=objtype, survey=True, goodz=True, all_zwarn0=True, z_analy=True)
    dz = calc_dz(gdz_tab)
    if len(dz) == 0:
        dz = np.zeros(1)
    not_nan = np.isfinite(dz)
    stat_dict['MEAN_DZ'] = float(np.mean(dz[not_nan]))
    stat_dict['MEDIAN_DZ'] = float(np.median(dz[not_nan]))
    stat_dict['RMS_DZ'] = float(np.std(dz[not_nan]))

    # Return
    return stat_dict


def spectype_confusion(simz_tab, zb_tab=None):
    """ Generate a Confusion Matrix for spectral types
    See the Confusion_matrix_spectypes Notebook in docs/nb for an example

    Parameters
    ----------
    simz_tab : Table
      Truth table;  may be input from truth.fits
    zb_tab : Table (optional)
      zcatalog/zbest table; may be input from zcatalog-mini.fits
      If provided, used to match the simz_tab to the zbest quantities

    Returns
    -------
    simz_tab : astropy.Table
      Merged table of simpsec data
    results : dict
      Nested dict.
      First key is the TRUESPECTYPE
      Second key is the SPECTYPE
      e.g.  results['QSO']['QSO'] reports the number of True QSO classified as QSO
      results['QSO']['Galaxy'] reports the number of True QSO classified as Galaxy
    """

    # Process simz_tab as need be
    if zb_tab is not None:
        match_truth_z(simz_tab, zb_tab, mini_read=True)
    # Cut down to those processed with the Redshift fitter
    measured_z = simz_tab['ZWARN'].mask == False
    cut_simz = simz_tab[measured_z]

    # Strip those columns
    strip_ttypes = np.char.rstrip(cut_simz['TRUESPECTYPE'])
    strip_stypes = np.char.rstrip(cut_simz['SPECTYPE'])

    # All TRUE, SPEC types
    ttypes = np.unique(strip_ttypes)
    stypes = np.unique(strip_stypes)

    # Init
    results = {}
    for ttype in ttypes:
        results[ttype] = {}

    # Fill
    for ttype in ttypes:
        itrue = strip_ttypes == ttype
        # Init correct answer in case there are none
        results[ttype][ttype] = 0
        # import pdb; pdb.set_trace()
        for stype in stypes:
            results[ttype][stype] = np.sum(strip_stypes[itrue] == stype)

    # Return
    return results


def find_zbest_files(fibermap_data):

    from desimodel.footprint import radec2pix
    # Init
    zbest_files = []
    # Search for zbest files with healpy
    ra_targ = fibermap_data['TARGET_RA'].data
    dec_targ = fibermap_data['TARGET_DEC'].data
    # Getting some NAN in RA/DEC
    good = np.isfinite(ra_targ) & np.isfinite(dec_targ)
    pixels = radec2pix(64, ra_targ[good], dec_targ[good])
    uni_pixels = np.unique(pixels)
    for uni_pix in uni_pixels:
        zbest_files.append(desispec.io.findfile('zbest', groupname=uni_pix, nside=64))
    # Return
    return zbest_files


def load_z(fibermap_files, zbest_files=None, outfil=None):
    '''Load input and output redshift values for a set of exposures

    Parameters
    ----------
    fibermap_files: list
      List of fibermap files;  None of these should be calibration..
    zbest_files: list, optional
      List of zbest output files
      Slurped from fibermap info if not provided
    outfil: str, optional
      Output file for the table

    Returns
    -------
    simz_tab: astropy.Table
      Merged table of simpsec data
    zb_tab: astropy.Table
      Merged table of zbest output
    '''
    # imports
    log = get_logger()

    # Init
    if zbest_files is None:
        flag_load_zbest = True
        zbest_files = []
    else:
        flag_load_zbest = False
    # Load up fibermap and simspec tables
    fbm_tabs = []
    sps_tabs = []
    for fibermap_file in fibermap_files:

        # zbest?
        if flag_load_zbest:
            fibermap_data = desispec.io.read_fibermap(fibermap_file)
            zbest_files += find_zbest_files(fibermap_data)

        log.info('Reading: {:s}'.format(fibermap_file))
        # Load simspec (for fibermap too!)
        simspec_file = fibermap_file.replace('fibermap','simspec')
        sps_hdu = fits.open(simspec_file)
        # Make Tables
        fbm_tabs.append(Table(sps_hdu['FIBERMAP'].data,masked=True))
        truth = Table(sps_hdu['TRUTH'].data,masked=True)
        if 'TRUTH_ELG' in sps_hdu:
           truth_elg = Table(sps_hdu['TRUTH_ELG'].data)
           truth = join(truth, truth_elg['TARGETID', 'OIIFLUX'],
                   keys='TARGETID', join_type='left')
        else:
            truth['OIIFLUX'] = 0.0

        sps_tabs.append(truth)
        sps_hdu.close()

    # Stack + Sort
    fbm_tab = vstack(fbm_tabs)
    sps_tab = vstack(sps_tabs)
    del fbm_tabs, sps_tabs

    fbm_tab.sort('TARGETID')
    sps_tab.sort('TARGETID')

    # Add the version number header keywords from fibermap_files[0]
    hdr = fits.getheader(fibermap_files[0].replace('fibermap', 'simspec'))
    for key, value in sorted(hdr.items()):
        if key.startswith('DEPNAM') or key.startswith('DEPVER'):
            fbm_tab.meta[key] = value

    # Drop to unique
    univ, uni_idx = np.unique(np.array(fbm_tab['TARGETID']),return_index=True)
    fbm_tab = fbm_tab[uni_idx]
    sps_tab = sps_tab[uni_idx]

    # Combine
    assert np.all(fbm_tab['TARGETID'] == sps_tab['TARGETID'])
    keep_colnames = list()
    for colname in sps_tab.colnames:
        if colname not in fbm_tab.colnames:
            keep_colnames.append(colname)

    simz_tab = hstack([fbm_tab,sps_tab[keep_colnames]],join_type='exact')

    # Cleanup some names
    #simz_tab.rename_column('OBJTYPE_1', 'OBJTYPE')
    #simz_tab.rename_column('OBJTYPE_2', 'TRUETYPE')

    # Update QSO naming
    qsol = np.where( match_otype(simz_tab, 'QSO') & (simz_tab['TRUEZ'] >= 2.1))[0]
    simz_tab['TEMPLATETYPE'][qsol] = 'QSO_L'
    qsot = np.where( match_otype(simz_tab, 'QSO') & (simz_tab['TRUEZ'] < 2.1))[0]
    simz_tab['TEMPLATETYPE'][qsot] = 'QSO_T'

    # Load up zbest files
    zb_tabs = []
    for zbest_file in zbest_files:
        try:
            zb_hdu = fits.open(zbest_file)
        except FileNotFoundError:
            log.error("zbest file {} not found".format(zbest_file))
        else:
            zb_tabs.append(Table(zb_hdu[1].data))

    # Stack
    zb_tab = vstack(zb_tabs)
    univ, uni_idx = np.unique(np.array(zb_tab['TARGETID']),return_index=True)
    zb_tab = zb_tab[uni_idx]

    # Return
    return simz_tab, zb_tab


def match_truth_z(simz_tab, zb_tab, mini_read=False, outfil=None):
    """ Match truth and zbest tables
    :param simz_tab: astropy.Table; Either generated from load_z() or read from disk via 'truth.fits'
    :param zb_tab: astropy.Table; Either generated from load_z() or read from disk via 'zcatalog-mini.fits'
    :param mini_read: bool, optional; Tables were read from the summary tables written to disk
    :param outfil: str, optional
    :return: simz_tab:  modified in place
    """

    nsim = len(simz_tab)
    # Match up
    sim_id = np.array(simz_tab['TARGETID'])
    z_id = np.array(zb_tab['TARGETID'])
    inz = np.in1d(z_id,sim_id,assume_unique=True)
    ins = np.in1d(sim_id,z_id,assume_unique=True)

    z_idx = np.arange(z_id.shape[0])[inz]
    sim_idx = np.arange(sim_id.shape[0])[ins]
    assert np.array_equal(sim_id[sim_idx],z_id[z_idx])

    # Fill up
    ztags = ['Z','ZERR','ZWARN','SPECTYPE']

    # This is for truth and zcat tables read from disk as opposed to the fibermap files
    if mini_read:
        ztags += ['DESI_TARGET']
        # And clean up the QSO names
        stypes = np.char.rstrip(simz_tab['TEMPLATETYPE'])
        qsol = np.where((stypes == 'QSO') & (simz_tab['TRUEZ'] >= 2.1))[0]
        simz_tab['TEMPLATETYPE'][qsol] = 'QSO_L'
        qsot = np.where((stypes == 'QSO') & (simz_tab['TRUEZ'] < 2.1))[0]
        simz_tab['TEMPLATETYPE'][qsot] = 'QSO_T'

    # Generate the new columns
    new_clms = []
    mask = np.array([True]*nsim)
    mask[sim_idx] = False
    for kk,ztag in enumerate(ztags):
        # Generate a MaskedColumn
        new_clm = MaskedColumn([zb_tab[ztag][z_idx[0]]]*nsim, name=ztag, mask=mask)
        #name=new_tags[kk], mask=mask)
        # Fill
        new_clm[sim_idx] = zb_tab[ztag][z_idx]
        # Append
        new_clms.append(new_clm)
    # Add columns
    simz_tab.add_columns(new_clms)

    # Write?
    if outfil is not None:
        simz_tab.write(outfil,overwrite=True)
    return


def obj_requirements(zstats, objtype):
    """Assess where a given objtype passes the requirements
    Requirements from Doc 318 (August 2014)

    Parameters
    ----------
    zstats : Object
        This parameter is not documented.
    objtype : str
        Object type, e.g. 'ELG', 'LRG'

    Returns
    -------
    dict
        Pass/fail dict
    """
    log = get_logger()
    pf_dict = {}
    #
    all_dict=dict(ELG={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.90},
        LRG={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.95},
                  BGS={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.95},
                  MWS={'RMS_DZ':0.0005, 'MEAN_DZ': 0.0002, 'CAT_RATE': 0.05, 'EFF': 0.95},
        QSO_T={'RMS_DZ':0.0025, 'MEAN_DZ': 0.0004, 'CAT_RATE': 0.05, 'EFF': 0.90},
        QSO_L={'RMS_DZ':0.0025, 'CAT_RATE': 0.02, 'EFF': 0.90})
    req_dict = all_dict[objtype]

    tst_fail = ''
    passf = str('PASS')
    for key in req_dict:
        ipassf = str('PASS')
        if key in ['EFF']: # Greater than requirement
            if zstats[key] < req_dict[key]:
                ipassf = str('FAIL')
                tst_fail = tst_fail+key+'-'
                log.warning('{:s} failed requirement {:s}: {} < {}'.format(objtype, key, zstats[key], req_dict[key]))
            else:
                log.debug('{:s} passed requirement {:s}: {} >= {}'.format(objtype, key, zstats[key], req_dict[key]))
        else:
            if zstats[key] > req_dict[key]:
                ipassf = str('FAIL')
                tst_fail = tst_fail+key+'-'
                log.warning('{:s} failed requirement {:s}: {} > {}'.format(objtype, key, zstats[key], req_dict[key]))
            else:
                log.debug('{:s} passed requirement {:s}: {} <= {}'.format(objtype, key, zstats[key], req_dict[key]))
        # Update
        pf_dict[key] = ipassf
        if ipassf == str('FAIL'):
            passf = str('FAIL')
    if passf == str('FAIL'):
        tst_fail = tst_fail[:-1]
        # log.warning('OBJ={:s} failed tests {:s}'.format(objtype,tst_fail))
    #
    #pf_dict['FINAL'] = passf
    return pf_dict, passf


def zstats(simz_tab, objtype=None, dvlimit=None, count=False, survey=False):
    """ Perform statistics on the input truth+z table
    good = Satisfies dv criteria and ZWARN==0
    fail = Fails dv criteria with ZWARN==0 (catastrophic failures)
    miss = Satisfies dv criteria but ZWARN!=0 (missed opportunities)
    lost = Fails dv criteria and ZWARN!=0 (lost, but at least we knew it)

    Args:
        simz_tab:
        objtype:
        dvlimit: float, optional -- Over-rides object specific dv limits
        count: bool, optional
        survey: bool, optional -- Restrict to targets meeting the Survey criteria (e.g. ELG flux)

    Returns:
        if count=True: just the raw counts of each category :: ngood, nfail, nmiss, nlost
        else: percentile of each relative to ntot, and ntot

    """
    # Grab the masks
    objtype_mask, z_mask, survey_mask, dv_mask, zwarn_mask = criteria(
        simz_tab, dvlimit=dvlimit, objtype=objtype)
    # Score-card
    good = zwarn_mask & dv_mask & objtype_mask & z_mask
    cat = zwarn_mask & (~dv_mask) & objtype_mask & z_mask
    miss = (~zwarn_mask) & dv_mask & objtype_mask & z_mask
    lost = (~zwarn_mask) & (~dv_mask) & objtype_mask & z_mask
    # Restrict to the Survey design?
    tot_msk = objtype_mask & z_mask
    if survey:
        good &= survey_mask
        cat &= survey_mask
        miss &= survey_mask
        lost &= survey_mask
        tot_msk &= survey_mask
    #
    ngood = np.count_nonzero(good)
    nfail = np.count_nonzero(cat)
    nmiss = np.count_nonzero(miss)
    nlost = np.count_nonzero(lost)
    ntot = np.count_nonzero(tot_msk)
    # Check
    assert(ntot == ngood+nfail+nmiss+nlost)
    # Return
    if count:
        return ngood, nfail, nmiss, nlost
    elif ntot == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    else:
        return 100*ngood/ntot, 100*nfail/ntot, 100*nmiss/ntot, 100*nlost/ntot, ntot


def criteria(simz_tab, objtype=None, dvlimit=None):
    """Analyze the input table for various criteria

    Parameters
    ----------
    simz_tab : Table
    objtype : str, optional -- Restrict analysis to a specific object type

    Returns
    -------
    objtype_mask : ndarray
      Match to input objtype (if any given)
    z_mask : ndarray
      Analyzed by the redshift analysis software
    survey_mask : ndarray
      Part of the DESI survey (not filler)
    dv_mask : ndarray
      Satisfies the dv criterion;  Either specific to each objtype
      or using an input dvlimit
    zwarn_mask : ndarray
      ZWARN=0
    """
    # Init
    nrow = len(simz_tab)
    stypes = np.char.rstrip(simz_tab['TEMPLATETYPE'])

    # Object type
    if objtype is None:
        objtype_mask = np.array([True]*nrow)
    else:
        if objtype in ['STAR', 'WD', 'QSO']:
            objtype_mask = stypes == objtype
        else:
            objtype_mask = match_otype(simz_tab, objtype)  # Use DESI_TARGET when possible
    # Redshift analysis
    z_mask = simz_tab['Z'].mask == False  # Not masked in Table
    # Survey
    survey_mask = (simz_tab['Z'].mask == False)
    elg = np.where(match_otype(simz_tab, 'ELG') & survey_mask)[0]
    if len(elg) > 0:
        elg_mask = elg_flux_lim(simz_tab['TRUEZ'][elg],
                                simz_tab['OIIFLUX'][elg])
        # Update
        survey_mask[elg[~elg_mask]] = False
    # zwarn -- Masked array
    zwarn_mask = np.array([False]*nrow)
    idx = np.where((simz_tab['ZWARN'] == 0) & (simz_tab['ZWARN'].mask == False))[0]
    zwarn_mask[idx] = True
    # Catastrophic/Good (This gets a bit more messy...)
    dv_mask = np.array([True]*nrow)
    for obj in np.unique(stypes):
        if obj in ['ELG','LRG','QSO_L','QSO_T', 'BGS', 'MWS']:  # Use DESI_TARGET when possible
            omask = np.where(match_otype(simz_tab, obj))[0] # & (simz_tab['ZWARN']==0))[0]
        else:
            omask = np.where(stypes == obj)[0]
        if dvlimit is None:
            try:
                dv = catastrophic_dv(obj) # km/s
            except:
                dv = 1000.
        else:
            dv = dvlimit
        dz = calc_dz(simz_tab[omask]) # dz/1+z
        cat = np.where(np.abs(dz)*3e5 > dv)[0]
        dv_mask[omask[cat]] = False
    # Return
    return objtype_mask, z_mask, survey_mask, dv_mask, zwarn_mask


def slice_simz(simz_tab, objtype=None, z_analy=False, survey=False,
    catastrophic=False, goodz=False, all_zwarn0=False, **kwargs):
    """Slice input simz_tab in one of many ways

    Parameters
    ----------
    z_analy : bool, optional
      redshift analysis required?
    all_zwarn0 : bool, optional
      Ignores catastrophic failures in the slicing to return
      all sources with ZWARN==0
    survey : bool, optional
      Only include objects that satisfy the Survey requirements
      e.g. ELGs with sufficient OII_flux
    catastrophic : bool, optional
      Restrict to catastropic failures
    goodz : bool, optional
      Restrict to good redshifts
    all_zwarn0 : bool, optional
      Restrict to ZWARN=0 cases
    **kwargs : passed to criteria

    Returns
    -------
    simz_table : Table cut by input parameters
    """
    # Grab the masks
    objtype_mask, z_mask, survey_mask, dv_mask, zwarn_mask = criteria(
        simz_tab, objtype=objtype, **kwargs)

    # Slice me
    final_mask = objtype_mask
    if z_analy:
        final_mask &= z_mask
    if survey:
        final_mask &= survey_mask
    if catastrophic:
        final_mask &= (~dv_mask)
        final_mask &= zwarn_mask  # Must also have ZWARN=0
    if goodz:
        final_mask &= dv_mask
        final_mask &= zwarn_mask
    if all_zwarn0:
        final_mask &= zwarn_mask

    # Return
    return simz_tab[final_mask]


def obj_fig(simz_tab, objtype, summ_stats, outfile=None):
    """Generate QA plot for a given object type
    """
    from astropy.stats import sigma_clip
    logs = get_logger()
    gdz_tab = slice_simz(simz_tab,objtype=objtype, survey=True,goodz=True, all_zwarn0=True)
    if objtype == 'ELG':
        allgd_tab = slice_simz(simz_tab,objtype=objtype, survey=False,goodz=True, all_zwarn0=True)

    if len(gdz_tab) <= 1:
        logs.info("Not enough objects of type {:s} for QA".format(objtype))
        return

    # Plot
    sty_otype = get_sty_otype()
    fig = plt.figure(figsize=(8, 6.0))
    gs = gridspec.GridSpec(2,2)
    # Title
    fig.suptitle('{:s}: Summary'.format(sty_otype[objtype]['lbl']),
        fontsize='large')

    # Offset
    for kk in range(4):
        yoff = 0.
        ax= plt.subplot(gs[kk])
        if kk == 0:
            yval = calc_dzsig(gdz_tab)
            ylbl = (r'$(z_{\rm red}-z_{\rm true}) / \sigma(z)$')
            ylim = 5.
            # Stats with clipping
            clip_y = sigma_clip(yval, sigma=5.)
            rms = np.std(clip_y)
            redchi2 = np.sum(clip_y**2)/np.sum(~clip_y.mask)
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
            ax.text(xtxt, ytxt, '{:s}: {:.3f}'.format('RMS:', rms),
                    color='black', transform=ax.transAxes, ha='left', fontsize='small')
            ytxt -= 0.12
            ax.text(xtxt, ytxt, '{:s}: {:.3f}'.format(r'$\chi^2_\nu$:',
                redchi2), color='black', transform=ax.transAxes,
                ha='left', fontsize='small')
        else:
            yval = calc_dz(gdz_tab)
            if kk == 1:
                ylbl = (r'$(z_{\rm red}-z_{\rm true}) / (1+z)$')
            else:
                ylbl = r'$\delta v_{\rm red-true}$ [km/s]'
            ylim = max(5.*summ_stats[objtype]['RMS_DZ'],1e-5)
            if (np.median(summ_stats[objtype]['MEDIAN_DZ']) >
                summ_stats[objtype]['RMS_DZ']):
                yoff = summ_stats[objtype]['MEDIAN_DZ']

        if kk==1:
            # Stats
            xtxt = 0.05
            ytxt = 1.0
            dx = ((ylim/2.)//0.0001 +1)*0.0001
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
            #i0, i1 = int( np.min(yval) / binsz) - 1, int( np.max(yval) / binsz) + 1
            i0, i1 = int(-ylim/binsz) - 1, int( ylim/ binsz) + 1
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
                xval = gdz_tab['TRUEZ']
                xmin,xmax=np.min(xval),np.max(xval)
                dx = np.maximum(1,(xmax-xmin)//0.5)*0.1
                ax.xaxis.set_major_locator(plt.MultipleLocator(dx))
                #xmin,xmax=0.6,1.65
            elif kk == 3:
                if objtype == 'ELG':
                    lbl = r'[OII] Flux ($10^{-16}$)'
                    #xval = gdz_tab['OIIFLUX']*1e16
                    xval = allgd_tab['OIIFLUX']*1e16
                    yval = calc_dz(allgd_tab)
                    # Avoid NAN
                    gdy = np.isfinite(yval)
                    xval = xval[gdy]
                    yval = yval[gdy]
                    xmin,xmax=0.5,20
                    ax.set_xscale("log", nonposx='clip')
                elif objtype == 'QSO':
                    lbl = 'g (Mag)'
                    xval = 22.5 - 2.5 * np.log10(gdz_tab['FLUX_G'])
                    xmin,xmax=np.min(xval),np.max(xval)
                else:
                    lbl = 'r (Mag)'
                    xval = 22.5 - 2.5 * np.log10(gdz_tab['FLUX_R'])
                    xmin,xmax=np.min(xval),np.max(xval)
            # Labels
            ax.set_xlabel(lbl)
            ax.set_ylabel(ylbl)
            ax.set_xlim(xmin,xmax)
            v_ylim = ylim * 3e5  # redshift to km/s
            ax.set_ylim(-v_ylim+yoff, v_ylim+yoff)

            # Points
            ax.plot([xmin,xmax], [0.,0], '--', color='gray')
            #ax.scatter(xval, yval, marker='o', s=1, label=objtype,
            #    color=sty_otype[objtype]['color'])
            cm = plt.get_cmap(sty_otype[objtype]['pcolor'])
            if objtype == 'ELG':
                xbins = 10**np.linspace(np.log10(xmin), np.log10(xmax), 20)
            else:
                xbins = np.linspace(xmin, xmax, 20)
            ybins = np.linspace(-v_ylim+yoff, v_ylim+yoff, 40) # km/s
            #import pdb; pdb.set_trace()
            counts, xedges, yedges = np.histogram2d(xval, yval * 3e5, bins=(xbins, ybins))
            max_c = np.max(counts)
            #if kk == 3:
            ax.pcolormesh(xedges, yedges, counts.transpose(), cmap=cm, vmin=0, vmax=max_c/5.)

            #ax.hist2d(xval, yval, bins=20, cmap=cm)
            #ax.scatter(xval, yval, marker='o', s=1, label=objtype,
            #           color=sty_otype[objtype]['color'])

    # Finish
    plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.3)
    plt.subplots_adjust(top=0.92)
    if outfile is not None:
        plt.savefig(outfile, dpi=700)
        plt.close()
        print("Wrote {:s}".format(outfile))


def summ_fig(simz_tab, summ_tab, meta, outfile=None):
    """Generate summary summ_fig
    :param simz_tab:
    :param summ_tab:
    :param meta:
    :param outfile:
    :return:
    """
    # Plot
    sty_otype = get_sty_otype()
    fig = plt.figure(figsize=(8, 5.0))
    gs = gridspec.GridSpec(3,2)

    # Redshifts analyzed
    zobj_tab = slice_simz(simz_tab, z_analy=True)
    otypes = ['ELG','LRG','QSO_L','QSO_T']

    # z vs. z plot
    jj=0
    ax= plt.subplot(gs[0:2,jj])

    # Catastrophic
    cat_tab = slice_simz(simz_tab, survey=True, catastrophic=True)
    ax.scatter(cat_tab['TRUEZ'], cat_tab['Z'],
        marker='x', s=9, label='CAT', color='red')

    notype = []
    for otype in otypes:
        gd_o = np.where(zobj_tab['TEMPLATETYPE']==otype)[0]
        notype.append(len(gd_o))
        ax.scatter(zobj_tab['TRUEZ'][gd_o], zobj_tab['Z'][gd_o],
            marker='o', s=1, label=sty_otype[otype]['lbl'], color=sty_otype[otype]['color'])
    ax.set_ylabel(r'$z_{\rm red}$')
    ax.set_xlabel(r'$z_{\rm true}$')
    ax.set_xlim(-0.1, 1.02*np.max(np.array([np.max(zobj_tab['TRUEZ']),
        np.max(zobj_tab['Z'])])))
    ax.set_ylim(-0.1, np.max(np.array([np.max(zobj_tab['TRUEZ']),
        np.max(zobj_tab['Z'])])))
    # Legend
    legend = ax.legend(loc='upper left', borderpad=0.3,
                        handletextpad=0.3, fontsize='small')

    # Zoom
    jj=1
    ax= plt.subplot(gs[0:2,jj])

    for otype in otypes:
        # Grab
        gd_o = np.where(zobj_tab['TEMPLATETYPE']==otype)[0]
        # Stat
        dz = calc_dz(zobj_tab[gd_o])
        ax.scatter(zobj_tab['TRUEZ'][gd_o], dz, marker='o',
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
    ax.text(xlbl, ylbl, 'SPECPROD: {:s}'.format(meta['SPECPROD']), transform=ax.transAxes, ha='left')
    yoff=0.15
    for key in meta:
        if key == 'SPECPROD':
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
        gd_o = simz_tab['TEMPLATETYPE']==otype
        ax.text(xlbl+0.1, ylbl, sty_otype[otype]['lbl']+': {:d} ({:d})'.format(np.sum(gd_o),notype[jj]),
            transform=ax.transAxes, ha='left', fontsize='small')

    # Finish
    plt.tight_layout(pad=0.1,h_pad=0.0,w_pad=0.1)
    if outfile is not None:
        plt.savefig(outfile, dpi=700)
        plt.close()



def summ_stats(simz_tab):
    '''Generate summary stats

    Parameters
    ----------
    simz_tab : Table
      Table summarizing redshifts

    Returns
    -------
    lis
      List of summary stat dicts
    '''
    otypes = ['ELG','LRG', 'QSO_L', 'QSO_T', 'BGS', 'MWS']  # WILL HAVE TO DEAL WITH QSO_TRACER vs QSO_LYA
    summ_dict = {}

    summ_dict['A_Legend'] = {}
    summ_dict['A_Legend']['CAT_RATE'] = 'Catastrohic Redshift failure rate'
    summ_dict['A_Legend']['EFF'] = 'Redshift Effeciency'
    summ_dict['A_Legend']['MEAN_DZ'] = 'Average redshift offset between measured and truth'
    summ_dict['A_Legend']['MEDIAN_DZ'] = 'Median redshift offset between measured and truth'
    summ_dict['A_Legend']['RMS_DZ'] = 'RMS of the redshift offsets'
    summ_dict['A_Legend']['N_CAT'] = 'Number of Catastropic failures (bad z and ZWARN=0)'
    summ_dict['A_Legend']['N_GOODZ'] = 'Number of targets with a correct redshift (within tolerance, in survey, and ZWARN=0)'
    summ_dict['A_Legend']['N_zA'] = 'Number of targets analyzed by the redshift code'
    summ_dict['A_Legend']['N_SURVEY'] = 'Number of targets included in the survey (ELGs with sufficient [OII] flux)'
    summ_dict['A_Legend']['N_TARG'] = 'Number of targets of the object type (irrespective of Survey design)'
    summ_dict['A_Legend']['N_ZWARN0'] = 'Number of redshifts with ZWARN == 0'
    summ_dict['A_Legend']['PURITY'] = 'Fraction of redshifts with ZWARN == 0 that are correct'
    summ_dict['A_Legend']['REQ_FINAL'] = 'Did the reduction pass all Requirements?'
    summ_dict['A_Legend']['REQ_INDIV'] = 'Did the reduction pass these individual requirements?'

    for otype in otypes:
        # Calculate stats
        stat_dict = calc_obj_stats(simz_tab, otype)
        summ_dict[otype] = stat_dict
        # Check requirements
        summ_dict[otype]['REQ_INDIV'], passf = obj_requirements(stat_dict,otype)
        summ_dict[otype]['REQ_FINAL'] = passf

    # Return
    return summ_dict


def plot_slices(x, y, ok, bad, x_lo, x_hi, y_cut, num_slices=5, min_count=100,
                axis=None):
    """Scatter plot with 68, 95 percentiles superimposed in slices.

    Requires that the matplotlib package is installed.

    Parameters
    ----------
    x : array of float
        X-coordinates to scatter plot.  Points outside [ `x_lo`, `x_hi` ] are
        not displayed.
    y : array of float
        Y-coordinates to scatter plot.  Y values are assumed to be roughly
        symmetric about zero.
    ok : array of bool
        Array of booleans that identify which fits are considered good.
    bad : array of bool
        Array of booleans that identify which fits have failed catastrophically.
    x_lo : float
        Minimum value of `x` to plot.
    x_hi : float
        Maximum value of `x` to plot.
    y_cut : float
        The target maximum value of :math:`|y|`.  A dashed line at this value is
        added to the plot, and the vertical axis is clipped at
        :math:`|y| = 1.25 \times y_{cut}` (but values outside this range are included in
        the percentile statistics).
    num_slices : int
        Number of equally spaced slices to divide the interval [ `x_lo`, `x_hi` ]
        into.
    min_count : int
        Do not use slices with fewer points for superimposed percentile
        statistics.
    axis : matplotlib axis object or None
        Uses the current axis if this is None.
    """
    #import matplotlib.pyplot as plt
    log = get_logger()

    if axis is None:
        axis = plt.gca()

    x_bins = np.linspace(x_lo, x_hi, num_slices + 1)
    x_i = np.digitize(x, x_bins) - 1
    limits = []
    counts = []
    for s in range(num_slices):
        # Calculate percentile statistics for ok fits.
        y_slice = y[ok & (x_i == s)]
        counts.append(len(y_slice))
        if counts[-1] > 0:
            limits.append(np.percentile(y_slice, (2.5, 16, 50, 84, 97.5)))
        else:
            limits.append((0., 0., 0., 0., 0.))
    limits = np.array(limits)
    counts = np.array(counts)

    # Plot scatter of all fits.
    axis.scatter(x[ok], y[ok], s=15, marker='.', lw=0, color='b', alpha=0.5)
    axis.scatter(x[~ok], y[~ok], s=15, marker='x', lw=0, color='k', alpha=0.5)

    # Plot quantiles in slices with enough fits.
    stepify = lambda y: np.vstack([y, y]).transpose().flatten()
    y_m2 = stepify(limits[:, 0])
    y_m1 = stepify(limits[:, 1])
    y_med = stepify(limits[:, 2])
    y_p1 = stepify(limits[:, 3])
    y_p2 = stepify(limits[:, 4])
    xstack = stepify(x_bins)[1:-1]
    for i in range(num_slices):
        s = slice(2 * i, 2 * i + 2)
        if counts[i] >= min_count:
            axis.fill_between(
                xstack[s], y_m2[s], y_p2[s], alpha=0.15, color='red')
            axis.fill_between(
                xstack[s], y_m1[s], y_p1[s], alpha=0.25, color='red')
            axis.plot(xstack[s], y_med[s], 'r-', lw=2.)

    # Plot cut lines.
    axis.axhline(+y_cut, ls=':', color='k')
    axis.axhline(0., ls='-', color='k')
    axis.axhline(-y_cut, ls=':', color='k')

    # Plot histograms of of not ok and catastrophic fits.
    rhs = axis.twinx()

    weights = np.ones_like(x[bad]) / len(x[ok])
    if len(weights) > 0:
        try:
            try:
                rhs.hist(
                    x[bad], range=(x_lo, x_hi), bins=num_slices, histtype='step',
                    weights=weights, color='k', cumulative=True)
            except:
                import pdb; pdb.set_trace()
        except UnboundLocalError:
            log.warning('All values lie outside the plot range')

    weights = np.ones_like(x[~ok]) / len(x)
    if len(weights) > 0:
        try:
            rhs.hist(
                x[~ok], range=(x_lo, x_hi), bins=num_slices, histtype='step',
                weights=weights, color='k', ls='dashed', cumulative=True)
        except UnboundLocalError:
            log.warning('All values lie outside the plot range')

    axis.set_ylim(-1.25 * y_cut, +1.25 * y_cut)
    axis.set_xlim(x_lo, x_hi)

    return axis, rhs


def dz_summ(simz_tab, outfile=None, pdict=None, min_count=20):
    """Generate a summary figure comparing zfind to ztruth.

    Parameters
    ----------
    simz_tab : Table
      Table of redshift information.
    pp : PdfPages object
      This parameter is not documented.
    pdict : dict
      Guides the plotting parameters
    min_count : int, optional
      This parameter is not documented.
    """
    log = get_logger()

    # INIT
    nrows = 2
    objtype = ['ELG', 'LRG', 'QSO_T', 'QSO_L']
    fluxes = ['OIIFLUX','ZMAG','GMAG','GMAG']
    ncols = len(objtype)
    #title = r'$\Delta v$ vs. z'

    # Plotting dicts
    if pdict is None:
        pdict = dict(ELG={'TRUEZ': { 'n': 15, 'min': 0.6, 'max': 1.6, 'label': 'redshift', 'overlap': 1 },
                          'RMAG': {'n': 12, 'min': 21.0, 'max': 23.4, 'label': 'r-band magnitude', 'overlap': 0},
                          'OIIFLUX': {'n': 10, 'min': 0.0, 'max': 5.0e-16, 'label': '[OII] flux', 'overlap': 2}},
                     LRG={'TRUEZ': {'n': 12, 'min': 0.5, 'max': 1.0, 'label': 'redshift', 'overlap': 2 },
                          'ZMAG': {'n': 15, 'min': 19.0, 'max': 21.0, 'label': 'z-band magnitude', 'overlap': 1 }},
                     QSO_T={'TRUEZ': {'n': 12, 'min': 0.5, 'max': 2.1, 'label': 'redshift', 'overlap': 1 },
                          'GMAG': {'n': 15, 'min': 19.0, 'max': 24.0, 'label': 'g-band magnitude', 'overlap': 1 }},
                     QSO_L={'TRUEZ': {'n': 12, 'min': 2.1, 'max': 4.0, 'label': 'redshift', 'overlap': 1 },
                            'GMAG': {'n': 15, 'min': 19.0, 'max': 24.0, 'label': 'g-band magnitude', 'overlap': 1 }},
        )

    # Initialize a new page of plots.
    plt.clf()
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(11, 8.5), facecolor='white',
        sharey=True)
    #figure.suptitle(title)

    # True Redshift
    row = 0
    ptype = 'TRUEZ'
    for row in range(nrows):
        for i,otype in enumerate(objtype):
            if row == 0:
                ptype = 'TRUEZ'
            else:
                ptype = fluxes[i]

            # Grab the set of measurements
            survey = slice_simz(simz_tab, objtype=otype, z_analy=True, survey=True)

            # Simple stats
            ok = survey['ZWARN'] == 0
            dv = calc_dz(survey)*3e5 # dz/1+z
            bad = dv > catastrophic_dv(otype)
            #if i==2:
            #    pdb.set_trace()

            # Plot the truth distribution for this variable.
            if ptype == 'TRUEZ':
                x = survey['TRUEZ']
            elif ptype == 'OIIFLUX':
                x = survey['OIIFLUX']
            else:
                if ptype == 'GMAG':
                    x = 22.5 - 2.5*np.log10(survey['FLUX_G'])
                elif ptype == 'RMAG':
                    x = 22.5 - 2.5*np.log10(survey['FLUX_R'])
                elif ptype == 'ZMAG':
                    x = 22.5 - 2.5*np.log10(survey['FLUX_Z'])
                else:
                    raise ValueError('unknown ptype {}'.format(ptype))

            nslice, x_min, x_max = pdict[otype][ptype]['n'], pdict[otype][ptype]['min'], pdict[otype][ptype]['max']
            rhs = None
            max_dv = 1000.
            max_frac = 0.1
            overlap = pdict[otype][ptype]['overlap']

            # axis
            col = i
            axis = axes[row][col]

            if len(survey) < 100:
                log.warning("Insufficient objects of type {:s}.  Skipping slice QA".format(otype))
                continue
            try:
                lhs, rhs = plot_slices(x=x.compressed(), y=dv[~x.mask].compressed(),
                                       ok=ok[~x.mask].compressed(), bad=bad[~x.mask].compressed(),
                                       x_lo=x_min, x_hi=x_max, num_slices=nslice,
                                       y_cut=max_dv, axis=axis, min_count=min_count)
            except:
                import pdb; pdb.set_trace()
            # Add a label even if the fitter has no results.
            xy = (0.5, 0.98)
            coords = 'axes fraction'
            axis.annotate(
                otype, xy=xy, xytext=xy, xycoords=coords,
                textcoords=coords, horizontalalignment='center',
                verticalalignment='top', size='large', weight='bold')

            rhs.set_ylim(0., max_frac)
            if col < ncols - 1:
                plt.setp([rhs.get_yticklabels()], visible=False)
            else:
                # Hide the last y-axis label except on the first row.
                if row > 0:
                    plt.setp([rhs.get_yticklabels()[-2:]], visible=False)
                rhs.set_ylabel('zwarn, catastrophic cummulative fraction')

            if col > 0:
                plt.setp([axis.get_yticklabels()], visible=False)
            else:
                axis.set_ylabel('Redshift fit residual $\Delta v$ [km/s]')

            #if row < nrows - 1:
            #    plt.setp([axis.get_xticklabels()], visible=False)
            #else:
            axis.set_xlabel('{0} {1}'.format(otype, ptype))
            axis.set_xlim(x_min, x_max)

            # Hide overlapping x-axis labels except in the bottom right.
            if overlap and (col < ncols - 1):
                axis.set_xticks(axis.get_xticks()[0:-overlap])

        figure.subplots_adjust(
            left=0.1, bottom=0.07, right=0.9, top=0.95,
            hspace=0.2, wspace=0.05)

    if outfile is not None:
        plt.savefig(outfile, dpi=700)
        print("Wrote: {:s}".format(outfile))
    plt.close()
