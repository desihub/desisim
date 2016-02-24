"""
Utility functions for working with simulated targets
"""

import os
import sys
import string

import numpy as np
import yaml

from desimodel.focalplane import FocalPlane
import desimodel.io

from desispec import brick
from desispec.io.fibermap import empty_fibermap

from desisim import io

def sample_objtype(nobj, flavor):
    """
    Return a random sampling of object types (dark, bright, MWS, BGS, ELG, LRG, QSO, STD, BAD_QSO)

    Args:
        nobj : number of objects to generate

    Returns:
        (true_objtype, target_objtype)

    where
        true_objtype   : array of what type the objects actually are
        target_objtype : array of type they were targeted as

    Notes:
    - Actual fiber assignment will result in higher relative fractions of
      LRGs and QSOs in early passes and more ELGs in later passes.
    - Also ensures at least 2 sky and 1 stdstar, even if nobj is small
    """

    #- Load target densities
    #- TODO: what about nobs_boss (BOSS-like LRGs)?
    #- TODO: This function should be using a desimodel.io function instead of opening desimodel directly.
    fx = open(os.environ['DESIMODEL']+'/data/targets/targets.dat')
    tgt = yaml.load(fx)
    fx.close()

    # DARK or BRIGHT have a combination of targets
    if string.lower(flavor) == 'dark':
        ntgt = float(tgt['nobs_lrg'] + tgt['nobs_elg'] + tgt['nobs_qso'] + tgt['nobs_lya'] + tgt['ntarget_badqso'])
    elif string.lower(flavor) == 'bright':
        ntgt = float(tgt['nobs_BG'] + tgt['nobs_MWS'])

    # initialize so we can ask for 0 of some kinds of survey targets later
    nlrg = nqso = nelg = nmws = nbgs = nbgs = nmws = 0
    #- Fraction of sky and standard star targets is guaranteed
    nsky = int(tgt['frac_sky'] * nobj)
    nstd = int(tgt['frac_std'] * nobj)

    #- Assure at least 2 sky and 1 std
    if nobj >= 3:
        if nstd < 1:
            nstd = 1
        if nsky < 2:
            nsky = 2

    #- Number of science fibers available
    nsci = nobj - (nsky+nstd)
    true_objtype  = ['SKY']*nsky + ['STD']*nstd
        
    if (string.lower(flavor) == 'mws'):
        true_objtype  +=  ['MWS_STAR']*nsci
    elif (string.lower(flavor) == 'qso'):
        true_objtype  +=  ['QSO']*nsci
    elif (string.lower(flavor) == 'elg'):
        true_objtype  +=  ['ELG']*nsci    
    elif (string.lower(flavor) == 'lrg'):
        true_objtype  +=  ['LRG']*nsci
    elif (string.lower(flavor) == 'std'):
        true_objtype  +=  ['STD']*nsci
    elif (string.lower(flavor) == 'bgs'):
        true_objtype  +=  ['BGS']*nsci
    elif (string.lower(flavor) == 'bright'):
        #- BGS galaxies and MWS stars
        nbgs = min(nsci, np.random.poisson(nsci * tgt['nobs_BG'] / ntgt))
        nmws = nsci - nbgs
        true_objtype += ['BGS']*nbgs
        true_objtype += ['MWS_STAR']*nmws
    elif (string.lower(flavor) == 'dark'):
        #- LRGs ELGs QSOs
        nlrg = np.random.poisson(nsci * tgt['nobs_lrg'] / ntgt)

        nqso = np.random.poisson(nsci * (tgt['nobs_qso'] + tgt['nobs_lya']) / ntgt)
        nqso_bad = np.random.poisson(nsci * (tgt['ntarget_badqso']) / ntgt)

        nelg = nobj - (nlrg+nqso+nqso_bad+nsky+nstd)

        true_objtype += ['ELG']*nelg
        true_objtype += ['LRG']*nlrg
        true_objtype += ['QSO']*nqso + ['QSO_BAD']*nqso_bad
    else:
        raise ValueError("Do not know the objtype mix for flavor "+ flavor)

    assert len(true_objtype) == nobj, \
        'len(true_objtype) mismatch for flavor {} : {} != {}'.format(\
        flavor, len(true_objtype), nobj)
    np.random.shuffle(true_objtype)

    target_objtype = list()
    for x in true_objtype:
        if x == 'QSO_BAD':
            target_objtype.append('QSO')
        else:
            target_objtype.append(x)

    target_objtype = np.array(target_objtype)
    true_objtype = np.array(true_objtype)

    return true_objtype, target_objtype

def get_targets(nspec, flavor, tileid=None):
    """
    Returns:
        fibermap
        truth table

    TODO (@moustakas): Deal with the random seed correctly.

    TODO: document this better
    """
    if tileid is None:
        tile_ra, tile_dec = 0.0, 0.0
    else:
        tile_ra, tile_dec = io.get_tile_radec(tileid)

    #- Get distribution of target types
    true_objtype, target_objtype = sample_objtype(nspec, flavor)
    
    #- Get DESI wavelength coverage
    wavemin = desimodel.io.load_throughput('b').wavemin
    wavemax = desimodel.io.load_throughput('z').wavemax
    dw = 0.2
    wave = np.arange(round(wavemin, 1), wavemax, dw)
    nwave = len(wave)

    truth = dict()
    truth['FLUX'] = np.zeros( (nspec, len(wave)) )
    truth['REDSHIFT'] = np.zeros(nspec, dtype='f4')
    truth['TEMPLATEID'] = np.zeros(nspec, dtype='i4')
    truth['OIIFLUX'] = np.zeros(nspec, dtype='f4')
    truth['D4000'] = np.zeros(nspec, dtype='f4')
    truth['VDISP'] = np.zeros(nspec, dtype='f4')
    truth['OBJTYPE'] = np.zeros(nspec, dtype='S10')
    #- Note: unlike other elements, first index of WAVE isn't spectrum index
    truth['WAVE'] = wave

    if flavor == 'BGS' or flavor == 'BRIGHT':
        truth['HBETAFLUX'] = np.zeros(nspec, dtype='f4')

    fibermap = empty_fibermap(nspec)

    for objtype in set(true_objtype):
        ii = np.where(true_objtype == objtype)[0]
        nobj = len(ii)

        fibermap['OBJTYPE'][ii] = target_objtype[ii]
        truth['OBJTYPE'][ii] = true_objtype[ii]

        # Simulate spectra
        if objtype == 'SKY':
            continue

        elif objtype == 'ELG':
            from desisim.templates import ELG
            elg = ELG(wave=wave)
            simflux, wave1, meta = elg.make_templates(nmodel=nobj)

        elif objtype == 'LRG':
            from desisim.templates import LRG
            lrg = LRG(wave=wave)
            simflux, wave1, meta = lrg.make_templates(nmodel=nobj)

        elif objtype == 'BGS':
            from desisim.templates import BGS
            bgs = BGS(wave=wave)
            simflux, wave1, meta = bgs.make_templates(nmodel=nobj)

        elif objtype == 'QSO':
            from desisim.templates import QSO
            qso = QSO(wave=wave)
            simflux, wave1, meta = qso.make_templates(nmodel=nobj)

        # For a "bad" QSO simulate a normal star without color cuts, which isn't
        # right. We need to apply the QSO color-cuts to the normal stars to pull
        # out the correct population of contaminating stars.
        elif objtype == 'QSO_BAD':
            from desisim.templates import STAR
            star = STAR(wave=wave)
            simflux, wave1, meta = star.make_templates(nmodel=nobj)

        elif objtype == 'STD':
            from desisim.templates import STAR
            star = STAR(wave=wave,FSTD=True)
            simflux, wave1, meta = star.make_templates(nmodel=nobj)

        elif objtype == 'MWS_STAR':
            from desisim.templates import STAR
            star = STAR(wave=wave)
            # todo: mag ranges for different flavors of STAR targets should be in desimodel
            simflux, wave1, meta = star.make_templates(nmodel=nobj,rmagrange=(15.0,20.0))
            
        truth['FLUX'][ii] = 1e17 * simflux
        truth['UNITS'] = '1e-17 erg/s/cm2/A'
        truth['TEMPLATEID'][ii] = meta['TEMPLATEID']
        truth['REDSHIFT'][ii] = meta['REDSHIFT']

        # Pack in the photometry.  TODO: Include WISE.
        magg = meta['GMAG']
        magr = meta['RMAG']
        magz = meta['ZMAG']
        fibermap['MAG'][ii, 0:3] = np.vstack( [magg, magr, magz] ).T
        fibermap['FILTER'][ii, 0:3] = ['DECAM_G', 'DECAM_R', 'DECAM_Z']

        if objtype == 'ELG':
            truth['OIIFLUX'][ii] = meta['OIIFLUX']
            truth['D4000'][ii] = meta['D4000']
            truth['VDISP'][ii] = meta['VDISP']

        if objtype == 'LRG':
            truth['D4000'][ii] = meta['D4000']
            truth['VDISP'][ii] = meta['VDISP']

        if objtype == 'BGS':
            truth['HBETAFLUX'][ii] = meta['HBETAFLUX']
            truth['D4000'][ii] = meta['D4000']
            truth['VDISP'][ii] = meta['VDISP']
            
    #- Load fiber -> positioner mapping and tile information
    fiberpos = desimodel.io.load_fiberpos()

    #- Where are these targets?  Centered on positioners for now.
    x = fiberpos['X'][0:nspec]
    y = fiberpos['Y'][0:nspec]
    fp = FocalPlane(tile_ra, tile_dec)
    ra = np.zeros(nspec)
    dec = np.zeros(nspec)
    for i in range(nspec):
        ra[i], dec[i] = fp.xy2radec(x[i], y[i])

    #- Fill in the rest of the fibermap structure
    fibermap['FIBER'] = np.arange(nspec, dtype='i4')
    fibermap['POSITIONER'] = fiberpos['POSITIONER'][0:nspec]
    fibermap['SPECTROID'] = fiberpos['SPECTROGRAPH'][0:nspec]
    fibermap['TARGETID'] = np.random.randint(sys.maxint, size=nspec)
    fibermap['TARGETCAT'] = np.zeros(nspec, dtype='|S20')
    fibermap['LAMBDAREF'] = np.ones(nspec, dtype=np.float32)*5400
    fibermap['TARGET_MASK0'] = np.zeros(nspec, dtype='i8')
    fibermap['RA_TARGET'] = ra
    fibermap['DEC_TARGET'] = dec
    fibermap['X_TARGET'] = x
    fibermap['Y_TARGET'] = y
    fibermap['X_FVCOBS'] = fibermap['X_TARGET']
    fibermap['Y_FVCOBS'] = fibermap['Y_TARGET']
    fibermap['X_FVCERR'] = np.zeros(nspec, dtype=np.float32)
    fibermap['Y_FVCERR'] = np.zeros(nspec, dtype=np.float32)
    fibermap['RA_OBS'] = fibermap['RA_TARGET']
    fibermap['DEC_OBS'] = fibermap['DEC_TARGET']
    fibermap['BRICKNAME'] = brick.brickname(ra, dec)

    return fibermap, truth


#-------------------------------------------------------------------------
#- Currently unused, but keep around for now
def sample_nz(objtype, n):
    """
    Given `objtype` = 'LRG', 'ELG', 'QSO', 'STAR', 'STD'
    return array of `n` redshifts that properly sample n(z)
    from $DESIMODEL/data/targets/nz*.dat
    """
    #- TODO: should this be in desimodel instead?

    #- Stars are at redshift 0 for now.  Could consider a velocity dispersion.
    if objtype in ('STAR', 'STD'):
        return np.zeros(n, dtype=float)

    #- Determine which input n(z) file to use
    targetdir = os.getenv('DESIMODEL')+'/data/targets/'
    objtype = objtype.upper()
    if objtype == 'LRG':
        infile = targetdir+'/nz_lrg.dat'
    elif objtype == 'ELG':
        infile = targetdir+'/nz_elg.dat'
    elif objtype == 'QSO':
        #- TODO: should use full dNdzdg distribution instead
        infile = targetdir+'/nz_qso.dat'
    else:
        raise ValueError("objtype {} not recognized (ELG LRG QSO STD STAR)".format(objtype))

    #- Read n(z) distribution
    zlo, zhi, ntarget = np.loadtxt(infile, unpack=True)[0:3]

    #- Construct normalized cumulative density function (cdf)
    cdf = np.cumsum(ntarget, dtype=float)
    cdf /= cdf[-1]

    #- Sample that distribution
    x = np.random.uniform(0.0, 1.0, size=n)
    return np.interp(x, cdf, zhi)

class TargetTile(object):
    """
    Keeps the relevant information for targets on a tile.
    Attributes:
         The properties initialized in the __init__ procedure:
         ra (float): array for the target's RA
         dec (float): array for the target's dec
         type (string): array for the type of target
         id (int): array of unique IDs for each target
         tile_ra (float): RA identifying the tile's center
         tile_dec (float) : dec identifying the tile's center
         tile_id (int): ID identifying the tile's ID
         n_target (int): number of targets stored in the object
         filename (string): original filename from which the info was loaded
         x (float): array of positions on the focal plane, in mm
         y (float): array of positions on the focal plane, in mm
         fiber_id (int): array of fiber_id to which the target is assigned
    """
    def __init__(self, filename):

        hdulist = fits.open(filename)
        self.filename = filename
        self.ra = hdulist[1].data['RA']
        self.dec = hdulist[1].data['DEC']
        self.type = hdulist[1].data['OBJTYPE']
        self.id = np.int_(hdulist[1].data['TARGETID'])
        self.tile_ra = hdulist[1].header['TILE_RA']
        self.tile_dec = hdulist[1].header['TILE_DEC']
        self.tile_id = hdulist[1].header['TILE_ID']
        self.n = np.size(self.ra)
        self.x, self.y = radec2xy(self.ra, self.dec, self.tile_ra, self.tile_dec)

        # this is related to the fiber assignment
        self.fiber = -1.0 * np.ones(self.n, dtype='i4')

        # This section is related to the number of times a galaxy has been observed,
        # the assigned redshift and the assigned type
        self.n_observed = np.zeros(self.n, dtype='i4')
        self.assigned_z = -1.0 * np.ones(self.n)
        self.assigned_type =  np.chararray(self.n, itemsize=8)
        self.assigned_type[:] = 'NONE'

    def set_fiber(self, target_id, fiber_id):
        """
        Sets the field .fiber[] (in the target_id  location) to fiber_uid
        Args:
            target_id (int): the target_id expected to be in self.id to modify
                 its corresponding .fiber[] field
            fiber_id (int): the fiber_id to be stored for the corresponding target_id
        """
        loc = np.where(self.id==target_id)
        if(np.size(loc)!=0):
            loc = loc[0]
            self.fiber[loc]  = fiber_id
        else:
            raise ValueError('The fiber with %d ID does not seem to exist'%(fibers_id))

    def reset_fiber(self, target_id):
        """
        Resets the field .fiber[] (in the target_id  location) to fiber_uid
        Args:
            target_id (int): the target_id expected to be in self.id to modify
                 its corresponding .fiber[] field
        """
        loc = np.where(self.id==target_id)
        if(np.size(loc)!=0):
            loc = loc[0]
            self.fiber[loc]  = -1
        else:
            raise ValueError('The fiber with %d ID does not seem to exist'%(fibers_id))


    def reset_all_fibers(self):
        """
        Resets the field .fiber[] for all fibers.
        """
        self.fiber = -1.0 * np.ones(self.n, dtype='i4')


    def write_results_to_file(self, targets_file):
        """
        Writes the section associated with the results to a fits file
        Args:
            targets_file (string): the name of the corresponding targets file
        """

        results_file = targets_file.replace("Targets_Tile", "Results_Tile")
        if(os.path.isfile(results_file)):
            os.remove(results_file)

        c0=fits.Column(name='TARGETID', format='K', array=self.id)
        c1=fits.Column(name='NOBS', format='I', array=self.n_observed)
        c2=fits.Column(name='ASSIGNEDTYPE', format='8A', array=self.assigned_type)
        c3=fits.Column(name='ASSIGNEDZ', format='D', array=self.assigned_z)

        cat=fits.ColDefs([c0,c1,c2,c3])
        table_targetcat_hdu=fits.TableHDU.from_columns(cat)

        table_targetcat_hdu.header['TILE_ID'] = self.tile_id
        table_targetcat_hdu.header['TILE_RA'] = self.tile_ra
        table_targetcat_hdu.header['TILE_DEC'] = self.tile_dec

        hdu=fits.PrimaryHDU()
        hdulist=fits.HDUList([hdu])
        hdulist.append(table_targetcat_hdu)
        hdulist.verify()
        hdulist.writeto(results_file)

    def load_results(self, targets_file):
        """
        Loads results from the FITS file to update the arrays n_observed, assigned_z
        and assigned_type.
        Args:
            tile_file (string): filename with the target information
        """
        results_file = targets_file.replace("Targets_Tile", "Results_Tile")
        try:
            fin = fits.open(results_file)
            self.n_observed = fin[1].data['NOBS']
            self.assigned_z = fin[1].data['ASSIGNEDZ']
            self.assigned_type =  fin[1].data['ASSIGNEDTYPE']
        except Exception, e:
            import traceback
            print 'ERROR in get_tiles'
            traceback.print_exc()
            raise e

    def update_results(self, fibers):
        """
        Updates the results of each target in the tile given the
        corresponding association with fibers.

        Args:
            fibers (object class FocalPlaneFibers): only updates the results if a target
                is assigned to a fiber.
        Note:
            Right now this procedure only opdates by one the number of observations.
            It should also updated the redshift and the assigned type (given some additional information!)
        """
        for i in range(fibers.n_fiber):
            t = fibers.target[i]
            if(t != -1):
                if((t in self.id)):
                    index = np.where(t in self.id)
                    index = index[0]
                    self.n_observed[index]  =  self.n_observed[index] + 1
                    # these two have to be updated as well TOWRITE
                    # self.assigned_z[index]
                    # self.assigned_type[index]
                else:
                    raise ValueError('The target associated with fiber_id %d does not exist'%(fibers.id[i]))




class TargetSurvey(object):
    """
    Keeps basic information for all the targets in all tiles.
    Attributes:
        The properties initialized in the __init__ procedure are:
        type (string): array describing the type of target.
        id (int): 1D array of unique IDs.
        n_observed (int)
        assigned_type (string): array describing the assigned type
        assigned_z (float): number of times this target has been observed
        tile_names (string): list of list keeping track of all the tiles where this target is present.
    """
    def __init__(self, filename_list):
        n_file = np.size(filename_list)
        for i_file in np.arange(n_file):
            print('Adding %s to build TargetSurvey %d files to go'%(filename_list[i_file], n_file - i_file))
            tmp = TargetTile(filename_list[i_file])
            # The first file is a simple initialization
            if(i_file==0):
                self.type = tmp.type.copy()
                self.id = tmp.id.copy()
                self.n_observed = tmp.n_observed.copy()
                self.assigned_type = tmp.assigned_type.copy()
                self.assigned_z = tmp.assigned_z.copy()
                self.tile_names= []
                for i in np.arange(np.size(self.id)):
                    self.tile_names.append([filename_list[i_file]])
            else: # the other files have to take into account the overlap
                mask = np.in1d(self.id, tmp.id)

                if((len(self.tile_names)!=np.size(self.id))):
                    raise ValueError('Building TargetSurvey the numer of items in the filenames is not the same as in the ids.')
                for i in np.arange(np.size(self.id)):
                    if(mask[i]==True):
                        self.tile_names[i].append(filename_list[i_file])

                mask = np.in1d(tmp.id, self.id, invert=True)
                n_new = np.size(np.where(mask==True))
                self.id = np.append(self.id, tmp.id[mask])
                self.type = np.append(self.type, tmp.type[mask])
                self.n_observed = np.append(self.n_observed, tmp.n_observed[mask])
                self.assigned_type = np.append(self.assigned_type, tmp.assigned_type[mask])
                self.assigned_z = np.append(self.assigned_z, tmp.assigned_z[mask])
                for i in np.arange(n_new):
                    self.tile_names.append([filename_list[i_file]])

        self.n_targets = np.size(self.id)
