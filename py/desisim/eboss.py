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
        self.nside = nside
        print('in eboss_footprint constructor, nside',nside)
        self.eboss_pix,self.eboss_dens = self.read_sdss_footprint(self.nside)
        print('got data from file')

        return

    @staticmethod
    def read_sdss_footprint(nside):
        if not nside==16:
            raise ValueError('add eBOSS footprint for nside='+str(nside))

        fname=resource_filename('desisim','data/eboss_footprint_nside_16.txt')
        print('in read_sdss_footprint from file',fname)
        if not os.path.isfile(fname):
            print('eBOSS footprint file',fname)
            raise ValueError('file with eBOSS footprint does not exist')
        data = np.loadtxt(fname)

        pix = data[:,0].astype(int)
        dens = data[:,1]

        return pix,dens

    def highz_density(self,ra,dec):
        pixs = healpy.ang2pix(self.nside, np.pi/2.-dec*np.pi/180.,
                    ra*np.pi/180.,nest=True)
        dens=np.zeros_like(ra)
        for p,d in zip(self.eboss_pix,self.eboss_dens):
            dens[pixs==p]=d
        return dens

def sdss_subsample(ra,dec,input_highz_density,eboss_footprint):
    """ Downsample input list of angular positions based on SDSS footprint
        and input density of all quasars .
    Args:
        ra (ndarray): Right ascension (degrees)
        dec (ndarray): Declination (degrees)
        input_highz_density (float): Input density of all quasars per sq.deg.
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




def create_sdss2desi_redshift_distribution_ratio(sdss_cat,desi_cat,out,dz=0.04,mjd_min=55000,nside=16,nest=True,zminDensity=1.8,densityCut=30.):
    """Create an ascii file giving the subsample fraction as a function of redshift
        to get the same redshift distribution in DESI as in SDSS

    Args:
        sdss_cat (path): Input SDSS quasar catalog
        desi_cat (path): Input DESI quasar catalog
        out (path): Output ascii file
        dz (float): Step in redshift (default=0.04)
        mjd_min (int): Minimum MJD (default=55000)
        nside (int): NSIDE for the sky (default==16)
        nest (bool): NEST for the sky (default==True)
        zminDensity (float): Redshift minimum cut to compute the density (default=1.8)
        densityCut (float): Density criterium for low vs. high (default=30.)
    Returns:
        None

    """

    zmin = 0.
    zmax = 10.
    bins = np.arange(zmin,zmax,dz)

    ## SDSS
    h = pyfits.open(sdss_cat)
    hh = h[1].data
    w = hh['MJD']>mjd_min

    sdss = {}
    sdss['Z'] = hh['Z'][w]
    sdss['RA'] = hh['RA'][w]
    sdss['DEC'] = hh['DEC'][w]

    w = sdss['Z']>zminDensity
    ra = sdss['RA'][w]
    dec = sdss['DEC'][w]
    h.close()

    phi = ra*np.pi/180.
    th = np.pi/2.-dec*np.pi/180.
    pix = healpy.ang2pix(nside,th,phi,nest=nest)

    unique_pix = np.unique(pix)
    bincounts_pix = np.bincount(pix)
    area_pix = healpy.pixelfunc.nside2pixarea(nside, degrees=True)
    density = bincounts_pix[unique_pix]/area_pix

    phi = sdss['RA']*np.pi/180.
    th = np.pi/2.-sdss['DEC']*np.pi/180.
    pix = healpy.ang2pix(nside,th,phi,nest=nest)

    sdss['LOWD'] = {}
    pixLowD = unique_pix[density<densityCut]
    w = np.in1d(pix,pixLowD)
    sdss['LOWD']['HIST'], zhist = np.histogram(sdss['Z'][w],bins=bins,density=True)
    sdss['LOWD']['ZHIST'] = np.array([ zhist[i]+(zhist[i+1]-zhist[i])/2. for i in range(zhist.size-1) ])
    sdss['LOWD']['PIXS'] = pixLowD

    sdss['HIGHD'] = {}
    pixHighD = unique_pix[density>=densityCut]
    w = np.in1d(pix,pixHighD)
    sdss['HIGHD']['HIST'], zhist = np.histogram(sdss['Z'][w],bins=bins,density=True)
    sdss['HIGHD']['ZHIST'] = np.array([ zhist[i]+(zhist[i+1]-zhist[i])/2. for i in range(zhist.size-1) ])
    sdss['HIGHD']['PIXS'] = pixHighD

    ## DESI
    desi = {}
    h = pyfits.open(desi_cat)
    hh = h[1].data
    desi['Z'] = hh['Z_QSO_RSD']
    h.close()
    desi['HIST'], zhist = np.histogram(desi['Z'],bins=bins,density=False)
    desi['ZHIST'] = np.array([ zhist[i]+(zhist[i+1]-zhist[i])/2. for i in range(zhist.size-1) ])

    ## Ratio
    ratio = {}
    for k in ['LOWD','HIGHD']:
        ratio[k] = {}
        w = (sdss[k]['ZHIST']>zmin) & (desi['ZHIST']>zmin)
        w &= (sdss[k]['ZHIST']<zmax) & (desi['ZHIST']<zmax)
        w &= (sdss[k]['ZHIST']>1.9) & (desi['ZHIST']>1.9)
        w &= (sdss[k]['ZHIST']<3.75) & (desi['ZHIST']<3.75)
        coef = (1.*desi['HIST'][w]/sdss[k]['HIST'][w]).min()
        ratio[k]['ZHIST'] = sdss[k]['ZHIST'].copy()
        ratio[k]['HIST'] = 1.*sdss[k]['HIST'].copy()*coef
        ratio[k]['PIXS'] = sdss[k]['PIXS'].copy()

        w = desi['HIST']>0.
        ratio[k]['HIST'][w] /= desi['HIST'][w]
        ratio[k]['HIST'][~w] = 1.
        ratio[k]['HIST'][ratio[k]['HIST']>1.] = 1.

    ## Save

    f = open(out,'w')
    f.write('# Fraction to downsample DESI mocks redshift distribution to SDSS redshift distribution\n')
    f.write('#\n')
    f.write('# SDSS catalog: {}\n'.format(sdss_cat))
    f.write('# DESI catalog: {}\n'.format(desi_cat))
    f.write('# Removing SDSS MJD: MJD_min = {}\n'.format(mjd_min))
    f.write('# NSIDE = {}\n'.format(nside))
    f.write('# NEST = {}\n'.format(nest))
    f.write('# Density low redshift quasars: z_min = {}\n'.format(zminDensity))
    f.write('# Density criteria: density_cut = {}\n'.format(densityCut))
    f.write('# Step of histogram: dz = {}\n'.format(dz))

    f.write('#\n')
    f.write('# Low density HEALPix list\n')
    f.write('# ')
    for el in ratio['LOWD']['PIXS']:
        f.write(str(el)+' ')
    f.write('\n')

    f.write('#\n')
    f.write('# High density HEALPix list\n')
    f.write('# ')
    for el in ratio['HIGHD']['PIXS']:
        f.write(str(el)+' ')
    f.write('\n')

    toSave = np.zeros( (ratio['LOWD']['ZHIST'].size, 3) )
    toSave[:,0] = ratio['LOWD']['ZHIST']
    toSave[:,1] = ratio['LOWD']['HIST']
    toSave[:,2] = ratio['HIGHD']['HIST']
    f.write('#\n')
    f.write('# z fraction_low_density fraction_high_density\n')
    f.write('#\n')
    for i in range(ratio['LOWD']['ZHIST'].size):
        f.write('%.5e %.5e %.5e\n' % (toSave[i,0],toSave[i,1],toSave[i,2]) )
    f.close()

    return

class RedshiftDistributionEBOSS(object):
    """ Class to store eBOSS redshift distribution fraction to DESI redshift distribution
     and provide useful functions.

    """

    def __init__(self,dz=0.04,nside=16):
        self.zmin = 0.
        self.zmax = 10.
        self.dz = dz
        self.nside = nside
        self.nest = True
        print('In eboss_footprint constructor, dz = {}'.format(dz))
        self.hist = self.read_sdss_redshift_distribution(self.dz,self.nside)
        print('got data from file')
        self.nz = self.hist['LOW_DENSITY']['Z'].size

        return

    @staticmethod
    def read_sdss_redshift_distribution(dz,nside):
        """Read the SDSS redshift fraction ascii file

        Args:
            dz (float): Step in redshift
            nside (int): NSIDE for the sky
        Returns:
            hist (dic): redshift fraction histogram dictionnary

        """
        if not dz==0.04:
            raise ValueError('add eBOSS redshift distribution fraction for dz = {}'.format(dz))
        if not nside==16:
            raise ValueError('add eBOSS footprint for nside = {}'.format(nside))

        fname = resource_filename('desisim','data/eboss_redshift_distributon_fraction_dz_004_nside_16.txt')
        print('in read_sdss_redshift_distribution from file {}'.format(fname))
        if not os.path.isfile(fname):
            print('eBOSS redshift distribution fraction file'.format(fname))
            raise ValueError('file with eBOSS redshift distribution fraction does not exist')

        data = np.loadtxt(fname)
        hist = {'LOW_DENSITY':None, 'HIGH_DENSITY':None}
        hist['LOW_DENSITY'] = {'PIX':None, 'Z':data[:,0], 'HIST':data[:,1] }
        hist['HIGH_DENSITY'] = {'PIX':None, 'Z':data[:,0], 'HIST':data[:,2] }

        f = open(fname,'r')
        nextLine = False
        for l in f:
            if nextLine:
                if hist['LOW_DENSITY']['PIX'] is None:
                    l = l.split()
                    hist['LOW_DENSITY']['PIX'] = np.array([ int(el) for el in l[1:] ])
                else:
                    l = l.split()
                    hist['HIGH_DENSITY']['PIX'] = np.array([ int(el) for el in l[1:] ])
                nextLine = False
            if 'Low density HEALPix list' in l or 'High density HEALPix list' in l:
                nextLine = True
        f.close()

        return hist

    def redshift_fraction(self,ra,dec,z):
        """Get the associated fraction for randoms at a given redshift slice

        Args:
            ra (float, array of float): Right Ascension
            dec (float, array of float): Declination
            z (float, array of float): redshift
        Returns:
            fraction (float, array of float): fraction to use to select randoms

        """
        phi = ra*np.pi/180.
        th = np.pi/2.-dec*np.pi/180.
        pix = healpy.ang2pix(self.nside,th,phi,nest=self.nest)
        bins = ( (z-self.zmin)/(self.zmax-self.zmin)*self.nz+0.5 ).astype(np.int64)

        frac = np.ones(ra.size)
        for k in ['LOW_DENSITY','HIGH_DENSITY']:
            w = np.in1d(pix,self.hist[k]['PIX'])
            frac[w] = self.hist[k]['HIST'][bins[w]]

        return frac

def sdss_subsample_redshift(ra,dec,z,eboss_redshift):
    """Get the array of selection to get the redshift distribution of SDSS

    Args:
        ra (float, array of float): Right Ascension
        dec (float, array of float): Declination
        z (array of float): redshift
    Returns:
        selection (array of bool): Array for selection

    """
    frac = eboss_redshift.redshift_fraction(ra,dec,z)
    selection = np.random.uniform(size=z.size)<frac
    return selection
