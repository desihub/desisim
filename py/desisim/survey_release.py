"""
desisim.survey_release
=============

Functions and methods for mimicking a DESI release footprint and object density with adequate exposure times.

"""
from __future__ import division, print_function
import os
import desisim
import numpy as np
import healpy as hp
import fitsio
from astropy.table import Table, vstack
from desiutil.log import get_logger
from scipy.stats import rv_discrete
from desitarget.targetmask import desi_mask
from desimodel.io import load_tiles
from desimodel.footprint import tiles2pix, is_point_in_desi, radec2pix


log = get_logger()
class SurveyRelease(object):
    """ Class object to preprocess a master catalog to mimic a DESI release footprint and object density with adequate exposure times.

    Args:
        mastercatalog (str): path to the master catalog.
        data_file (str): path to the data catalog.
        include_nonqso_targets (bool): if True, will keep non QSO targets on data catalog and generate them in mock catalog.
        seed (int): seed for random number generation.
        invert (bool): if True, will invert the random number generation for redshift and magnitude distributions.
    """
    def __init__(self,mastercatalog,data_file=None,include_nonqso_targets=False,seed=None,invert=False):
        self.mockcatalog=Table.read(mastercatalog,hdu=1) # Assumes master catalog is in HDU1
        log.info(f"There are {len(self.mockcatalog)} targets in {mastercatalog} master catalog.")
        if 'Z' not in self.mockcatalog.colnames and 'Z_QSO_RSD' in self.mockcatalog.colnames:
            self.mockcatalog.rename_column('Z_QSO_RSD', 'Z')
        else:
            raise Exception("Mock catalog Z column not found")
        self.invert=invert
        self.include_nonqso_targets = include_nonqso_targets
        np.random.seed(seed)

        self.data = None
        if data_file is not None:
            self.data = self.prepare_data_catalog(data_file,zmin=min(self.mockcatalog['Z']),zmax=max(self.mockcatalog['Z']),
                                                  include_nonqso_targets=self.include_nonqso_targets)
        elif self.include_nonqso_targets:
            raise ValueError("The include_nonqso_targets option was set to True. This requires an input data catalog to extract distributions.")
            
    @staticmethod
    def prepare_data_catalog(cat_filename,zmin=None,zmax=None,include_nonqso_targets=False):
        """Prepare the data catalog for the quickquasars pipeline.
        Args:
            cat_filename (str): path to the data catalog.
            zmin (float): minimum redshift of the distribution
            zmax (float): maximum redshift of the distribution
            include_nonqso_targets (bool): if True, will keep non QSO targets on data catalog and generate them in mock catalog.
        returns:
            cat (astropy.table.Table): data catalog preprocessed in the redshift range of the mock master catalog.
        """
        log.info(f"Reading data catalog: {cat_filename}")
        cat = Table.read(cat_filename)
        log.info(f"Found {len(cat)} targets in data catalog.")
        w_z =(cat['Z']>=zmin)&(cat['Z']<=zmax)
        log.info(f"Keeping {np.sum(w_z)} targets in redshift range ({zmin},{zmax}) in data catalog.")
        cat=cat[w_z]
        
        qso_targets_mask = get_qsotarget_mask(cat)
        log.info(f"Keeping {np.sum(qso_targets_mask)} ({np.sum(qso_targets_mask)/np.sum(w_z):.2%}) QSO targets in data catalog.")
        if not include_nonqso_targets:
            log.info(f"Excluding {np.sum(~qso_targets_mask)} ({np.sum(~qso_targets_mask)/np.sum(w_z):.2%}) non QSO targets from data catalog.")
            cat=cat[qso_targets_mask]
        else:
            log.info(f"Keeping {np.sum(~qso_targets_mask)} ({np.sum(~qso_targets_mask)/np.sum(w_z):.2%}) non QSO targets in data catalog.")
        return cat
    
    def apply_redshift_dist(self,zmin=0,zmax=10,dz=0.1,distribution='SV'):
        """Apply a redshift distribution to the catalog.

        Args:
            zmin (float): minimum redshift of the distribution
            zmax (float): maximum redshift of the distribution
            dz (float): redshift bin width
            distribution (str): 'SV' or 'target_selection' for the moment.
        """
        if distribution=='SV':
            filename = os.path.join(os.path.dirname(desisim.__file__),'data/dn_dzdM_EDR.fits')
            with fitsio.FITS(filename) as fts:
                dn_dzdm=fts[3].read()
                zcenters=fts[1].read()

            dndz=np.sum(dn_dzdm,axis=1)
            dz = zcenters[1] - zcenters[0]
            zmin = max(zcenters[0]-0.5*dz,zmin)
            zmax = min(zcenters[-1]+0.5*dz,zmax)
            w_z = (zcenters-0.5*dz > zmin) & (zcenters+0.5*dz <= zmax)
            dndz=dndz[w_z]
            zcenters=zcenters[w_z]
            zbins = np.linspace(zmin,zmax,len(zcenters)+1)
        elif distribution=='target_selection':
            dist = Table.read(os.path.join(os.path.dirname(desisim.__file__),'data/redshift_dist_chaussidon2022.ecsv')) 
            dz_dist=dist['z'][1]-dist['z'][0]
            factor=0.1/dz_dist # Account for redshift bin width difference.
            
            zbins = np.linspace(zmin,zmax,int((zmax-zmin)/dz)+1)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            dndz=factor*np.interp(zcenters,dist['z'],dist['dndz_23'],left=0,right=0)
        elif distribution=='from_data':
            raise NotImplementedError(f"Option {distribution} is not implemented yet")
        else:
            raise ValueError(f"Distribution option {distribution} not in available options: SV, target_selection, from_data")
        
        selection = self.sample_redshift(dndz,zbins=zbins)
        selected_ids = list(self.mockcatalog['MOCKID'][selection])
        self.mockcatalog['IS_QSO_TARGET'] = np.isin(self.mockcatalog['MOCKID'],selected_ids)
        log.info(f"Selecting {sum(selection)} mock QSO targets following the distribution: {distribution}")
        if self.include_nonqso_targets:
            avail_ids = self.mockcatalog['MOCKID'][~selection]
            dndz_data_nonqso = self.get_nonqso_dndz(expected_dist=dist,zbins=zbins) 
            selected_nonqso = self.sample_redshift(dndz_data_nonqso,zbins=zbins,mask=~selection)
            selected_nonqso_ids = avail_ids[selected_nonqso]
            log.info(f"Selecting {sum(selected_nonqso)} mock non QSO targets following the distribution of data.")
            self.are_nonqso_targets = np.isin(self.mockcatalog['MOCKID'],selected_nonqso_ids)
            selection |= self.are_nonqso_targets  
        log.info(f"Selected {sum(selection)} mock targets out of {len(self.mockcatalog)}")
        self.mockcatalog=self.mockcatalog[selection]
                                 
    def get_nonqso_dndz(self,expected_dist, zbins=np.linspace(0,10,100+1)):
        """Get the redshift distribution of non QSO targets in the data catalog.

        Args:
            expected_dist (Table): Table with expected redshift distribution.
            zbins (array): redshift bins.

        Returns:    
            dndz_data_nonqso (array): redshift distribution of non QSO targets in the data catalog.
        """
        log.info(f"Sampling mock targets from data non QSO targets. Assuming data QSO targets already follow expected distribution.")
        targetmask = get_qsotarget_mask(self.data)
        nonqso_targets = self.data[~targetmask]
        dndz_data_nonqso = np.histogram(nonqso_targets['Z'],bins=zbins)[0]
        # Figure out the area of the catalog by checking number of Lya. 
        # This assumes that data lya qso targets already follow the expected distribution.
        qso_targets = self.data[targetmask]
        lyabins = np.linspace(2.1,3.8,18)
        dndz_data_qso = np.histogram(qso_targets['Z'],bins=lyabins)[0]
        dz_dist=expected_dist['z'][1]-expected_dist['z'][0]
        factor=0.1/dz_dist # Account for redshift bin width difference.
        zcenters=0.5*(lyabins[1:]+lyabins[:-1])
        expected_dndz=factor*np.interp(zcenters,expected_dist['z'],expected_dist['dndz_23'],left=0,right=0)
        area = np.sum(dndz_data_qso)/np.sum(expected_dndz)
        return dndz_data_nonqso/area
            
    def sample_redshift(self,target_dndz, zbins = np.linspace(0,10,100+1), mask=None):
        """Randomly sample mockcatalog to follow target redshift distribution. 
        
        Args:
            target_dndz (array): target redshift distribution.
            zbins (array): redshift bins.
            mask (array): mask to apply to the catalog before sampling.
            
        Returns:
            selection (array): boolean array with True where the targets were selected."""
        mock_area=get_catalog_area(self.mockcatalog,nside=16)
        mockz = self.mockcatalog['Z']
        if mask is not None:
            mockz = mockz[mask]
        dndz_mock=np.histogram(mockz,bins=zbins,
                                   weights=np.repeat(1/mock_area,len(mockz)))[0]
        fractions=np.divide(target_dndz,dndz_mock,where=dndz_mock>0,out=np.zeros_like(dndz_mock))
        fractions[fractions>1]=1

        bin_index = np.digitize(mockz, zbins) - 1
        rand = np.random.uniform(size=mockz.size)
        if self.invert:
            rand = 1-rand
        selection = rand < fractions[bin_index]
        return selection
        
    def apply_data_geometry(self, release='Y5',tilefile=None):
        """ Apply the geometry of a given data release to the mock catalog.
        
        Args:
            release (str): data release name. release='Y5' will simulate DESI-Y5 footprint without NPASS downsampling.
            tilefile (str): path to a tile file. If None, will use the tiles from the release.
        """
        mock=self.mockcatalog
        if tilefile is None:
            log.info(f"Getting Lya tiles for {release.upper()} release.")
            tiles=get_lya_tiles(release)
        else:
            log.info(f"Reading tiles from file: {tilefile}")
            tiles=Table.read(tilefile)
        mask_footprint=is_point_in_desi(tiles,mock['RA'],mock['DEC'])
        if sum(mask_footprint)==0:
            raise ValueError("There are no mock targets available in the tiles region")
            
        mock=mock[mask_footprint]
        log.info(f"Keeping {sum(mask_footprint)} mock targets in tiles footprint.")

        # If 'Y5' is selected there is no need to downsample by NPASS the mock catalog.
        if release!='Y5':
            log.info(f"Downsampling by NPASSES fraction in tiles")
            self.npass_pixmap=npass_pixmap_from_tiles(tiles,nside=1024)
            mock_pixels = radec2pix(ra=mock['RA'], dec=mock['DEC'], nside=1024)
            if self.data is None:
                raise ValueError("No data catalog was provided.")
            if 'TARGET_DEC' in self.data.colnames and 'TARGET_RA' in self.data.colnames:
                data_pixels = radec2pix(dec=self.data['TARGET_DEC'], ra=self.data['TARGET_RA'], nside=1024)
            elif 'DEC' in self.data.colnames and 'RA' in self.data.colnames: 
                data_pixels = radec2pix(dec=self.data['TARGET_DEC'], ra=self.data['TARGET_RA'], nside=1024)
            else:
                raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in data catalog.")

            data_passes = self.npass_pixmap[data_pixels]['NPASS']
            mock_passes = self.npass_pixmap[mock_pixels]['NPASS']
            data_pass_counts = np.bincount(data_passes,minlength=8) # Minlength = 7 passes + 1 (for bining)
            mock_pass_counts = np.bincount(mock_passes,minlength=8)
            mock['NPASS'] = mock_passes
            downsampling=np.divide(data_pass_counts,mock_pass_counts,out=np.zeros(8),where=mock_pass_counts>0)
            rand = np.random.uniform(size=len(mock))
            if self.invert:
                rand = 1-rand
            selection = rand<downsampling[mock_passes]
            log.info(f"Keeping {len(mock[selection])} mock targets out of {len(mock)}.")
            mock = mock[selection]
        self.mockcatalog=mock  

    def assign_rband_magnitude(self,from_data=False):
        """Assign r-band magnitude and include FLUX_R to the catalog according to the distribution.
        
        Args:
            from_data (bool): if True, will use the data catalog to generate the distribution. Otherwise, will use the default distribution as expected by SV.
        """
        if self.include_nonqso_targets and not from_data:
            log.warning("Setting from_data = True as the include_nonqso_targets requires it.") 
            from_data = True
        mock_mask = self.mockcatalog['IS_QSO_TARGET']
        self.mockcatalog['FLUX_R'] = 0. # Initiate FLUX_R values
        if not from_data:
            filename=os.path.join(os.path.dirname(desisim.__file__),'data/dn_dzdM_EDR.fits')
            with fitsio.FITS(filename) as fts:
                zcenters=fts['Z_CENTERS'].read()
                rmagcenters=fts['RMAG_CENTERS'].read()
                dn_dzdm=fts['dn_dzdm'].read()
            log.info(f"Generating random magnitudes according to distribution: {filename}")
        else:
            if self.data is None:
                raise ValueError("No data catalog was provided") 
            data_mask = get_qsotarget_mask(self.data)
            log.info(f"Computing dN/dzdm from data QSO targets")
            dn_dzdm, zbins, rmagbins = get_catalog_dndzdm(self.data,mask=data_mask)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            rmagcenters=0.5*(rmagbins[1:]+rmagbins[:-1])
            
            if self.include_nonqso_targets:
                log.info(f"Computing dN/dzdm from data non QSO targets")
                dn_dzdm_nonqso, zbins, rmagbins = get_catalog_dndzdm(self.data, mask=~data_mask)
                log.info(f"Assigning r-band magnitude to {sum(~mock_mask)} non QSO targets.")
                self.mockcatalog['FLUX_R'][~mock_mask] = generate_random_fluxes(dn_dzdm_nonqso, self.mockcatalog['Z'][~mock_mask],
                                                                              zcenters, rmagcenters, self.invert) 
                
        log.info(f"Assigning r-band magnitude to {sum(mock_mask)} QSO targets.")
        self.mockcatalog['FLUX_R'][mock_mask] = generate_random_fluxes(dn_dzdm, self.mockcatalog['Z'][mock_mask], zcenters, rmagcenters, self.invert)
        assert np.all(self.mockcatalog['FLUX_R']!=0)
            
    def assign_exposures(self,exptime=None):
        """ Assign exposures to the catalog according to the distribution in the data release.
        
        Args:
            exptime (float, optional): if not None, will assign the given exposure time in seconds to all objects in the mock catalog.
        """
        if exptime is not None:
            log.info(f"Assigning uniform exposures {exptime} seconds")
            self.mockcatalog['EXPTIME']=exptime
        else:
            if self.data is None:
                raise ValueError("No data catalog nor exptime was provided. Please provide one of them")
            log.info("Assigning exposures")
            mock=self.mockcatalog
            if 'TARGET_DEC' in self.data.colnames and 'TARGET_RA' in self.data.colnames:
                data_pixels = radec2pix(ra=self.data['TARGET_RA'], dec=self.data['TARGET_DEC'], nside=1024)
            elif 'DEC' in self.data.colnames and 'RA' in self.data.colnames: 
                data_pixels = radec2pix(ra=self.data['RA'], dec=self.data['DEC'], nside=1024)
            else:
                raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in data catalog")
                
            is_lya_data=self.data['Z']>2.1
            is_data_qsotgt=get_qsotarget_mask(self.data)
            data_mask = is_lya_data&is_data_qsotgt
            if 'TSNR2_LRG' in self.data.colnames:
                log.info('Getting effective exposure time in data catalog by 12.15*TSNR2_LRG.')
                exptime_data = 12.15*self.data['TSNR2_LRG']
            else:
                log.warning('Effective exposure time TSNR2_LRG column not found in observed data catalog.') 
                log.warning('Will compute effective exposure time from alternative templates.')
                if 'TSNR2_LYA' in self.data.colnames:
                    log.info('Getting effective exposure time in data catalog by 11.8*TSNR2_LYA.')
                    exptime_data = 11.8*self.data['TSNR2_LYA']
                elif 'TSNR2_QSO' in self.data.colnames:
                    log.info('Getting effective exposure time in data catalog by 33.61*TSNR2_QSO.')
                    exptime_data = 33.61*self.data['TSNR2_QSO']
                else: 
                    raise ValueError("Can't compute effective exposure time. Data catalog columns should include TSNR2_LRG, TSNR2_LYA or TSNR2_QSO.")
            exptime_mock = np.zeros(len(mock))
            is_mock_qsotgt = mock['IS_QSO_TARGET']
            is_lya_mock = mock['Z']>2.1
            mock_mask = is_lya_mock&is_mock_qsotgt
            exptime_mock[~mock_mask]=1000
            for tile_pass in range(1,8):
                w=self.npass_pixmap[data_pixels]['NPASS'][data_mask] == tile_pass
                pdf=np.histogram(exptime_data[data_mask][w]/1000,bins=np.arange(.5,9.5,1),density=True)[0]
                random_variable = rv_discrete(values=(np.arange(1,len(pdf)+1),pdf))
                is_pass = mock['NPASS'] == tile_pass
                exptime_mock[mock_mask&is_pass]=1000*random_variable.rvs(size=np.sum(is_pass&mock_mask))
            self.mockcatalog['EXPTIME']=exptime_mock

def get_qsotarget_mask(cat):
    """Get boolean mask for QSO and non QSO targets in the data catalog.

    Args:
        cat (astropy.table.Table): data catalog.

    Returns:
        qso_targets_mask (array): boolean mask for QSO targets.
    """
    mask = desi_mask.mask('QSO|QSO_HIZ|QSO_NORTH|QSO_SOUTH')
    qso_targets_mask = ((cat["DESI_TARGET"]&mask)>0)
    return qso_targets_mask
            
def get_catalog_area(catalog, nside=256):
    """Return the area of the catalog in square degrees.

    Args:
        nside (int): HEALPix nside parameter

    Returns:
        area (float): area of the catalog in square degrees.
    """
    if 'DEC' in catalog.colnames and 'RA' in catalog.colnames: 
        pixels = radec2pix(ra=catalog['RA'], dec=catalog['DEC'], nside=nside)
    elif 'TARGET_DEC' in catalog.colnames and 'TARGET_RA' in catalog.colnames:
        pixels = radec2pix(ra=catalog['TARGET_RA'], dec=catalog['TARGET_DEC'], nside=nside)
    else:
        raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in catalog")
    pixarea = hp.pixelfunc.nside2pixarea(nside, degrees=True)
    npix = len(np.unique(pixels))
    return npix*pixarea

def get_lya_tiles(release='Y5'):
    """Return the tiles that have been observed in a given release.

    Args:
        release (str,optional): release name. If release=='Y5', will return all DESI DARK tiles.

    Returns:
        tiles (astropy.table.Table): Table with the tiles observed in the specified release. 
    """
    all_tiles = load_tiles()
    dark_tiles = all_tiles['PROGRAM']=='DARK'
    tiles = all_tiles[dark_tiles]
    if release!='Y5':
        if release.upper() not in ['FUJI','FUGU']:
            surveys = ['main']
            redux_tiles_filename = os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'{release}/tiles-{release}.fits')
            redux_tiles=Table.read(redux_tiles_filename)
        else:
            fuji_tiles_filename= os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'fuji/tiles-fuji.fits')
            fuji_tiles=Table.read(fuji_tiles_filename)
            if release.upper()=='FUJI':
                surveys = ['sv1','sv3']
            elif release.upper()=='FUGU':
                surveys = ['main','sv1','sv3']
                guadalupe_tiles_filename=os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'guadalupe/tiles-guadalupe.fits')
                guadalupe_tiles=Table.read(guadalupe_tiles_filename)
                redux_tiles=vstack([guadalupe_tiles,fuji_tiles])
            # All tiles in surveys (Main and/or SV)
            tile_arr = [load_tiles(tilesfile = tiles_filename.format(survey)) for survey in surveys]                
            tiles = vstack(tile_arr)
        
        # Only get DARK tiles in spectro redux
        dark_tiles = redux_tiles['PROGRAM']=='dark'
        tiles_in_surveys = np.isin(np.char.decode(redux_tiles['SURVEY']),surveys)
        redux_tiles = redux_tiles = redux_tiles[dark_tiles&tiles_in_surveys]
        
        tiles_in_redux = np.isin(tiles['TILEID'],redux_tiles['TILEID'])
        tiles = tiles[tiles_in_redux]
    return tiles
    
        
def get_catalog_dndzdm(cat,zbins=np.linspace(0,10,100+1), rmagbins=np.linspace(15,25,100+1),mask=None):
    """Get the redshift and magnitude distributions of a catalog.

    Args:
        cat (astropy.table.Table): data catalog.
        zbins (array): redshift bins.
        rmagbins (array): r-band magnitude bins.
        mask (array): mask to apply to the catalog before computing the distributions.

    Returns:
        dn_dzdm (array): redshift and magnitude distribution.
        zbins (array): redshift bins.
        rmagbins (array): r-band magnitude bins.
    """
    z=cat['Z']
    if 'RMAG' in cat.colnames:
        rmag = cat['RMAG']
    elif 'FLUX_R' in cat.colnames:
        rmag = 22.5-2.5*np.log10(cat['FLUX_R'])
    else:
        raise ValueError("Magnitude information could not be retrieved from data catalog")
    if mask is not None:
        z=z[mask]
        rmag = rmag[mask]
    dn_dzdm=np.histogram2d(z,rmag,bins=(zbins,rmagbins))[0]
    return dn_dzdm, zbins, rmagbins

def generate_random_fluxes(dist, z, zcenters, rmagcenters,invert=False):
    cdf=np.cumsum(dist,axis=1)
    cdf = np.divide(cdf,cdf[:,-1][:,None],where=cdf[:,-1][:,None]>0,out=np.zeros_like(cdf))
    dz = zcenters[1]-zcenters[0]
    mags=np.zeros(len(z))

    for i,z_bin in enumerate(zcenters):
        w_z = (z > z_bin-0.5*dz) & (z <= z_bin+0.5*dz)
        if np.sum(w_z)==0: continue
        rand = np.random.uniform(size=np.sum(w_z))
        if invert:
            rand = 1-rand
        mags[w_z]=np.interp(rand,cdf[i],rmagcenters)  
    if np.sum(mags==0)!=0:
        raise ValueError(f"Randomly generated magnitudes contain zeros.")
    return 10**((22.5-mags)/2.5)

def npass_pixmap_from_tiles(tiles,nside=1024):
    """Create a pixmap with the number of passes in each pixel.

    Args:
        tiles (astropy.table.Table): Table with the tiles.
        nside (int): HEALPix nside parameter.

    Returns:
        pixmap (astropy.table.Table): Table with the number of passes in each HEALPix pixel.
    """
    pixmap = Table()
    npix = hp.nside2npix(nside)
    pixmap["HPXPIXEL"] = np.arange(npix, dtype=int)
    thetas, phis = hp.pix2ang(nside, np.arange(npix), nest=True)
    pixmap["RA"], pixmap["DEC"] = np.degrees(phis), 90.0 - np.degrees(thetas)

    passes = np.unique(tiles["PASS"])
    npass = len(passes)
    # listing the area covered by 1, 2, ..., npass passes
    pixmap["TILEIDS"] = np.zeros((npix, npass), dtype=int)
    for i in range(len(tiles)):
        ipixs = tiles2pix(nside, tiles=Table(tiles[i]))
        pixmap["TILEIDS"][ipixs, tiles["PASS"][i]] = tiles["TILEID"][i]
    pixmap["NPASS"] = (pixmap["TILEIDS"] != 0).sum(axis=1)
    return pixmap