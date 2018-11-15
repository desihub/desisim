"""
desisim.eboss
===========

Functions and methods for mimicking eBOSS survey.

"""
from __future__ import division, print_function
import numpy as np
import os
import healpy
from pkg_resources import resource_filename
import astropy.io.fits as pyfits


def create_sdss2desi_redshift_distribution_ratio(sdss_cat,desi_cat,out,dz=0.04,mjd_min=55000):
    """Create an ascii file giving the subsample fraction as a function of redshift
        to get the same redshift distribution in DESI as in SDSS

    Args:
        sdss_cat (path): Input SDSS quasar catalog
        desi_cat (path): Input DESI quasar catalog
        out (path): Output ascii file
        dz (float)L Step in redshift (default=0.04)
        mjd_min (int): Minimum MJD (default=55000)
    Returns:
        None

    """

    bins = np.arange(0.,10.,dz)

    ## SDSS
    sdss = {}
    h = pyfits.open(sdss_cat)
    hh = h[1].data
    w = hh['MJD']>mjd_min
    sdss['Z'] = hh['Z'][w]
    h.close()
    sdss['HIST'], zhist = np.histogram(sdss['Z'],bins=bins,density=True)
    sdss['ZHIST'] = np.array([ zhist[i]+(zhist[i+1]-zhist[i])/2. for i in range(zhist.size-1) ])

    ## DESI
    desi = {}
    h = pyfits.open(desi_cat)
    hh = h[1].data
    desi['Z'] = hh['Z_QSO_RSD']
    h.close()
    desi['HIST'], zhist = np.histogram(desi['Z'],bins=bins,density=False)
    desi['ZHIST'] = np.array([ zhist[i]+(zhist[i+1]-zhist[i])/2. for i in range(zhist.size-1) ])

    ## Ratio
    w = (sdss['HIST']>0.) & (desi['HIST']>0.)
    w &= (sdss['ZHIST']>1.9) & (desi['ZHIST']>1.9)
    w &= (sdss['ZHIST']<3.75) & (desi['ZHIST']<3.75)
    coef = (1.*desi['HIST'][w]/sdss['HIST'][w]).min()
    ratio = {}
    ratio['ZHIST'] = sdss['ZHIST'].copy()
    ratio['HIST'] = 1.*sdss['HIST'].copy()*coef

    w = desi['HIST']>0.
    ratio['HIST'][w] /= desi['HIST'][w]
    ratio['HIST'][~w] = 1.
    ratio['HIST'][ratio['HIST']>1.] = 1.

    ## Save
    toSave = np.zeros( (ratio['ZHIST'].size, 2) )
    toSave[:,0] = ratio['ZHIST']
    toSave[:,1] = ratio['HIST']
    header = 'Fraction to downsample DESI mocks redshift distribution to SDSS redshift distribution\n\n'
    header += 'SDSS catalog: {}\n'.format(sdss_cat)
    header += 'DESI catalog: {}\n'.format(desi_cat)
    header += 'Removing SDSS MJD: MJD_min = {}\n'.format(mjd_min)
    header += 'Step of histogram: dz = {}\n\n'.format(dz)
    header += 'z fraction\n\n'
    np.savetxt(out, toSave, fmt='%.5e %.5e', header=header)

    return
def create_sdss_footprint(sdss_cat,out,mjd_min=55000,zmin=1.8,nside=16,nest=True):
    """Create an ascii file giving for each HEALPix index the
        density of quasars in number per square degree

    Args:
        sdss_cat (path): Input SDSS quasar catalog
        out (path): Output ascii file
        mjd_min (int): Minimum MJD (default=55000)
        zmin (float): Minimum redshift (default==1.8)
        nside (int): NSIDE for the sky (default==16)
        nest (bool): NEST for the sky (default==True)
    Returns:
        None

    """

    ## SDSS
    h = pyfits.open(sdss_cat)
    hh = h[1].data
    w = (hh['MJD']>mjd_min) & (hh['Z']>zmin)
    ra = hh['RA'][w]
    dec = hh['DEC'][w]
    z = hh['Z'][w]
    h.close()

    phi = ra*np.pi/180.
    th = np.pi/2.-dec*np.pi/180.
    pix = healpy.ang2pix(nside,th,phi,nest=nest)

    unique_pix = np.unique(pix)
    bincounts_pix = np.bincount(pix)

    area_pix = healpy.pixelfunc.nside2pixarea(nside, degrees=True)
    density = bincounts_pix[unique_pix]/area_pix

    ## Save
    toSave = np.zeros( (density.size, 2) )
    toSave[:,0] = unique_pix
    toSave[:,1] = density
    header = 'SDSS footprint\n\n'
    header += 'SDSS catalog: {}\n'.format(sdss_cat)
    header += 'Removing SDSS MJD: MJD_min = {}\n'.format(mjd_min)
    header += 'Removing low redshift quasars: z_min = {}\n'.format(zmin)
    header += 'NSIDE = {}\n'.format(nside)
    header += 'NEST = {}\n\n'.format(nest)
    header += 'pixel_index density [nb/deg^{2}]\n\n'
    np.savetxt(out, toSave, fmt='%u %.5e', header=header)

    return

class FootprintEBOSS(object):
    """ Class to store eBOSS footprint and provide useful functions. """

    def __init__(self,nside=16):
        self.nside=nside
        print('in eboss_footprint constructor, nside',nside)
        data=read_sdss_footprint(self.nside)
        print('got data from file')
        self.eboss_pix=data[0]
        self.eboss_dens=data[1]

    def read_sdss_footprint(nside):
        if not nside==16:
            raise ValueError('add eBOSS footprint for nside='+str(nside))
        fname=resource_filename('desisim','data/eboss_footprint_nside_16.txt')
        print('in read_sdss_footprint from file',fname)
        if not os.path.isfile(fname):
            print('eBOSS footprint file',fname)
            raise ValueError('file with eBOSS footprint does not exist')
        data=np.loadtxt(fname)
        pix=data[:,0].astype(int)
        dens=data[:,1]
        return pix,dens

    def highz_density(self,ra,dec):
        pixs = healpy.ang2pix(self.nside, np.pi/2.-dec*np.pi/180.,
                    ra*np.pi/180.,nest=True)
        dens=np.zeros_like(ra)
        for p,d in zip(self.eboss_pix,self.eboss_dens):
            dens[pixs==p]=d
        return dens

class redshiftDistributionEBOSS(object):
    """ Class to store eBOSS redshift distribution fraction to DESI redshift distribution
     and provide useful functions.

    """

    def __init__(self,dz=0.04):
        self.zmin = 0.
        self.zmax = 10.
        self.dz = dz
        print('In eboss_footprint constructor, dz = {}'.format(dz))
        z,f = read_sdss_redshift_distribution(self.dz)
        print('got data from file')
        self.eboss_z = z
        self.eboss_frac = f

    def read_sdss_redshift_distribution(dz):
        if not dz==0.04:
            raise ValueError('add eBOSS redshift distribution fraction for dz = {}'.format(dz))

        fname = resource_filename('desisim','data/eboss_redshift_distributon_fraction_dz_004.txt')
        print('in read_sdss_redshift_distribution from file {}'.format{fname})
        if not os.path.isfile(fname):
            print('eBOSS redshift distribution fraction file'.format{fname})
            raise ValueError('file with eBOSS redshift distribution fraction does not exist')

        data = np.loadtxt(fname)
        z = data[:,0]
        f = data[:,1]

        return z,f

    def redshift_fraction(self,z):
        bins = ( (z-self.zmin)/(self.zmax-self.zmin)/dz ).astype(np.int64)
        return self.eboss_frac[bins]

def sdss_subsample(ra,dec,input_highz_density,eboss_footprint):
    """ Downsample input list of angular positions based on SDSS footprint
        and input density of high-z quasars (z>1.8).
    Args:
        ra (ndarray): Right ascension (degrees)
        dec (ndarray): Declination (degrees)
        input_highz_density (float): Input density of high-z quasars per sq.deg.
    Returns:
        selection (ndarray): mask to apply to downsample input list
    """
    debug=True
    # figure out expected SDSS density, in quasars / sq.deg., above z=1.8
    N=len(ra)
    density = eboss_footprint.highz_density(ra,dec)
    #density = sdss_highz_density(ra,dec)
    if debug: print(np.min(density),'<density<',np.max(density))
    fraction = density/input_highz_density
    if debug: print(np.min(fraction),'<fraction<',np.max(fraction))
    selection = np.where(np.random.uniform(size=N)<fraction)[0]
    if debug: print(len(selection),'selected out of',N)

    return selection

