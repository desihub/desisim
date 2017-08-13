"""
desisim.spec_qa.s2n
=========================
Module to examine S/N in object spectra
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

from desiutil.log import get_logger, DEBUG
from desispec.io import get_exposures, findfile, read_fibermap, read_frame
from desisim.spec_qa.utils import get_sty_otype

log = get_logger()

def load_s2n_values(objtype, nights, channel, sub_exposures=None):
    fdict = dict(waves=[], s2n=[], fluxes=[], exptime=[], OII=[])
    for night in nights:
        if sub_exposures is not None:
            exposures = sub_exposures
        else:
            exposures = get_exposures(night)#, raw=True)
        for exposure in exposures:
            fibermap_path = findfile(filetype='fibermap', night=night, expid=exposure)
            fibermap_data = read_fibermap(fibermap_path)
            flavor = fibermap_data.meta['FLAVOR']
            if flavor.lower() in ('arc', 'flat', 'bias'):
                log.debug('Skipping calibration {} exposure {:08d}'.format(flavor, exposure))
                continue
            # Load simspec
            simspec_file = fibermap_path.replace('fibermap', 'simspec')
            sps_hdu = fits.open(simspec_file)
            sps_tab = Table(sps_hdu['TRUTH'].data,masked=True)
            sps_hdu.close()
            objs = sps_tab['TEMPLATETYPE'] == objtype
            if np.sum(objs) == 0:
                continue

            # Load spectra (flux or not fluxed; should not matter)
            for ii in range(10):
                camera = channel+str(ii)
                cframe_path = findfile(filetype='cframe', night=night, expid=exposure, camera=camera)
                try:
                    cframe = read_frame(cframe_path)
                except:
                    log.warn("Cannot find file: {:s}".format(cframe_path))
                    continue
                # Calculate S/N
                iobjs = objs[cframe.fibers]
                if np.sum(iobjs) == 0:
                    continue
                s2n = cframe.flux[iobjs,:] * np.sqrt(cframe.ivar[iobjs,:])
                # Save
                fdict['waves'].append(cframe.wave)
                fdict['s2n'].append(s2n)
                fdict['fluxes'].append(sps_tab['MAG'][cframe.fibers[iobjs]])
                if objtype == 'ELG':
                    fdict['OII'].append(sps_tab['OIIFLUX'][cframe.fibers[iobjs]])
                fdict['exptime'].append(cframe.meta['EXPTIME'])
    # Return
    return fdict

def obj_s2n_wave(s2n_dict, wv_bins, flux_bins, otype, outfile=None, ax=None):
    """Generate QA of S/N for a given object type
    """
    logs = get_logger()
    nwv = wv_bins.size
    nfx = flux_bins.size
    s2n_sum = np.zeros((nwv-1,nfx-1))
    s2n_N = np.zeros((nwv-1,nfx-1)).astype(int)
    # Loop on exposures+wedges  (can do just once if these are identical for each)
    for jj, wave in enumerate(s2n_dict['waves']):
        w_i = np.digitize(wave, wv_bins) - 1
        m_i = np.digitize(s2n_dict['fluxes'][jj], flux_bins) - 1
        mmm = []
        for ll in range(nfx-1): # Only need to do once
            mmm.append(m_i == ll)
        #
        for kk in range(nwv-1):
            all_s2n = s2n_dict['s2n'][jj][:,w_i==kk]
            for ll in range(nfx-1):
                if np.any(mmm[ll]):
                    s2n_sum[kk, ll] += np.sum(all_s2n[mmm[ll],:])
                    s2n_N[kk, ll] += np.sum(mmm[ll]) * all_s2n.shape[1]

    sty_otype = get_sty_otype()

    # Plot
    if ax is None:
        fig = plt.figure(figsize=(6, 6.0))
        ax= plt.gca()
    # Title
    fig.suptitle('{:s}: Summary'.format(sty_otype[otype]['lbl']),
        fontsize='large')

    # Plot em up
    wv_cen = (wv_bins + np.roll(wv_bins,-1))/2.
    lstys = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    mxy = 1e-9
    for ss in range(nfx-1):
        if np.sum(s2n_N[:,ss]) == 0:
            continue
        lbl = 'MAG = [{:0.1f},{:0.1f}]'.format(flux_bins[ss], flux_bins[ss+1])
        ax.plot(wv_cen[:-1], s2n_sum[:,ss]/s2n_N[:,ss], linestyle=lstys[ss],
                label=lbl, color=sty_otype[otype]['color'])
        mxy = max(mxy, np.max(s2n_sum[:,ss]/s2n_N[:,ss]))

    ax.set_xlabel('Wavelength (Ang)')
    #ax.set_xlim(-ylim, ylim)
    ax.set_ylabel('Mean S/N per pixel in bins of 20A')
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim(0.1, mxy*1.1)

    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                      handletextpad=0.3, fontsize='medium', numpoints=1)

    # Finish
    plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.3)
    plt.subplots_adjust(top=0.92)
    if outfile is not None:
        plt.savefig(outfile, dpi=600)
        print("Wrote: {:s}".format(outfile))


def obj_s2n_z(s2n_dict, z_bins, flux_bins, otype, outfile=None, ax=None):
    """Generate QA of S/N for a given object type vs. z (mainly for ELG)
    """
    logs = get_logger()
    nz = z_bins.size
    nfx = flux_bins.size
    s2n_sum = np.zeros((nz-1,nfx-1))
    s2n_N = np.zeros((nz-1,nfx-1)).astype(int)
    # Loop on exposures+wedges  (can do just once if these are identical for each)
    for jj, wave in enumerate(s2n_dict['waves']):
        # Turn wave into z
        zELG = wave / 3728. - 1.
        z_i = np.digitize(zELG, z_bins) - 1
        m_i = np.digitize(s2n_dict['OII'][jj]*1e17, flux_bins) - 1
        mmm = []
        for ll in range(nfx-1): # Only need to do once
            mmm.append(m_i == ll)
        #
        for kk in range(nz-1):
            all_s2n = s2n_dict['s2n'][jj][:,z_i==kk]
            for ll in range(nfx-1):
                if np.any(mmm[ll]):
                    s2n_sum[kk, ll] += np.sum(all_s2n[mmm[ll],:])
                    s2n_N[kk, ll] += np.sum(mmm[ll]) * all_s2n.shape[1]

    sty_otype = get_sty_otype()

    # Plot
    if ax is None:
        fig = plt.figure(figsize=(6, 6.0))
        ax= plt.gca()
    # Title
    fig.suptitle('{:s}: Redshift Summary'.format(sty_otype[otype]['lbl']),
                 fontsize='large')

    # Plot em up
    z_cen = (z_bins + np.roll(z_bins,-1))/2.
    lstys = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    mxy = 1e-9
    for ss in range(nfx-1):
        if np.sum(s2n_N[:,ss]) == 0:
            continue
        lbl = 'OII(1e-17) = [{:0.1f},{:0.1f}]'.format(flux_bins[ss], flux_bins[ss+1])
        ax.plot(z_cen[:-1], s2n_sum[:,ss]/s2n_N[:,ss], linestyle=lstys[ss],
                label=lbl, color=sty_otype[otype]['color'])
        mxy = max(mxy, np.max(s2n_sum[:,ss]/s2n_N[:,ss]))

    ax.set_xlabel('Redshift')
    ax.set_xlim(z_bins[0], z_bins[-1])
    ax.set_ylabel('Mean S/N per pixel in dz bins')
    ax.set_yscale("log", nonposy='clip')
    ax.set_ylim(0.1, mxy*1.1)

    legend = plt.legend(loc='lower right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='medium', numpoints=1)

    # Finish
    plt.tight_layout(pad=0.2,h_pad=0.2,w_pad=0.3)
    plt.subplots_adjust(top=0.92)
    if outfile is not None:
        plt.savefig(outfile, dpi=600)
        print("Wrote: {:s}".format(outfile))




# Command line execution
if __name__ == '__main__':
    import desispec.io
    from astropy.table import Table
    from astropy.io import fits

    # Test obj_s2n method
    if False:
        nights = ['20190901']
        exposures = [65+i for i in range(6)]
        s2n_values = load_s2n_values('ELG', nights, 'b', sub_exposures=exposures)
        wv_bins = np.arange(3570., 5950., 20.)
        obj_s2n_wave(s2n_values, wv_bins, np.arange(19., 25., 1.0), 'ELG', outfile='tst.pdf')

    # Test obj_s2n_z
    if True:
        nights = ['20190901']
        exposures = [65+i for i in range(6)]
        s2n_values = load_s2n_values('ELG', nights, 'z', sub_exposures=exposures)
        z_bins = np.linspace(1.0, 1.6, 100) # z camera
        oii_bins = np.array([1., 6., 10., 30., 100., 1000.])
        obj_s2n_z(s2n_values, z_bins, oii_bins, 'ELG', outfile='tstz.pdf')



