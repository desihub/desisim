"""
desisim.survey_release
=============

Functions and methods for mimicking a DESI release footprint and object density with adequate exposure times.

"""
from __future__ import division, print_function
from pkg_resources import resource_filename
import os
import desisim
import numpy as np
import healpy as hp
import fitsio
from astropy.io import fits
from astropy.table import Table, vstack
from desiutil.log import get_logger
from scipy.stats import rv_discrete
from desitarget.targetmask import desi_mask
import desimodel.footprint


log = get_logger()

class SurveyRelease(object):
    """ Generate a master catalog mimicking a DESI release footprint and object density with adequate exposure times.
    To be inputted to quickquasars.

    Args:
        mastercatalog (str): path to the master catalog.
        data_file (str): path to the data catalog.
        qso_only (bool): if True, will only keep QSO targets.
        seed (int): random seed for reproducibility
        invert (bool): if True, will shuffle the random number generation by 1-random_number.
    """
    def __init__(self,mastercatalog,data_file=None,qso_only=True,seed=None,invert=False):
        self.mockcatalog=Table.read(mastercatalog,hdu=1) # Assumes master catalog is in HDU1
        log.info(f"Obtained {len(self.mockcatalog)} objects from {mastercatalog} master catalog")
        self.mockcatalog['TARGETID']=self.mockcatalog['MOCKID']
        self.mockcatalog['Z']=self.mockcatalog['Z_QSO_RSD']
        self.invert=invert
        np.random.seed(seed)

        self.data = None
        if data_file is not None:
            self.data = self.prepare_data_catalog(data_file,zmin=min(self.mockcatalog['Z']),zmax=max(self.mockcatalog['Z']),qsos_only=qso_only)
        
    @staticmethod
    def get_catalog_area(catalog, nside=256):
        """Return the area of the catalog in square degrees.
        
        Args:
            nside (int): HEALPix nside parameter
            
        Returns:
            float: area of the catalog in square degrees
        """
        if 'DEC' in catalog.colnames and 'RA' in catalog.colnames: 
            pixels = hp.ang2pix(nside, np.radians(90-catalog['DEC']),np.radians(catalog['RA']),nest=True)
        elif 'TARGET_DEC' in catalog.colnames and 'TARGET_RA' in catalog.colnames:
            pixels = hp.ang2pix(nside, np.radians(90-catalog['TARGET_DEC']),np.radians(catalog['TARGET_RA']),nest=True)
        else:
            raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in catalog")
        pixarea = hp.pixelfunc.nside2pixarea(nside, degrees=True)
        npix = len(np.unique(pixels))
        return npix*pixarea

    @staticmethod
    def prepare_data_catalog(cat_filename,zmin=None,zmax=None,qsos_only=True):
        """Prepare the data catalog for the quickquasars pipeline.
        Args:
            cat_filename (str): path to the data catalog.
            zmin (float): minimum redshift of the distribution
            zmax (float): maximum redshift of the distribution
            qsos_only (bool): if True, will only keep QSO targets.
        """
        log.info(f"Reading data catalog {cat_filename}")
        cat = Table.read(cat_filename)
        log.info(f"Found {len(cat)} targets in catalog")
        w_z =(cat['Z']>=zmin)&(cat['Z']<=zmax)
        log.info(f"Keeping {np.sum(w_z)} QSOs in redshift range ({zmin},{zmax}) in data catalog")
        cat=cat[w_z]
        if qsos_only: 
            mask = desi_mask.mask('QSO|QSO_HIZ|QSO_NORTH|QSO_SOUTH')
            qso_targets = ((cat["DESI_TARGET"]&mask)>0)
            log.info(f"Keeping {np.sum(qso_targets)} ({np.sum(qso_targets)/len(cat):.2%}) QSO targets in data catalog")
            log.info(f"{np.sum(~qso_targets)} ({np.sum(~qso_targets)/np.sum(w_z):.2%}) non QSO targets will be excluded from data catalog")
            cat=cat[qso_targets]
        return cat

    def apply_redshift_dist(self,zmin=0,zmax=10,distribution='SV'):
        """Apply a redshift distribution to the catalog.

        Args:
            zmin (float): minimum redshift of the distribution
            zmax (float): maximum redshift of the distribution
            distribution (str): 'SV' or 'target_selection' for the moment.
        """
        if distribution=='SV':
            filename = os.path.join(os.path.dirname(desisim.__file__),'data/dn_dzdM_EDR.fits')
            with fitsio.FITS(filename) as fts:
                dn_dzdr=fts[3].read()
                zcenters=fts[1].read()

            dndz=np.sum(dn_dzdr,axis=1)
            dz = zcenters[1] - zcenters[0]
            zmin = max(zcenters[0]-0.5*dz,zmin)
            zmax = min(zcenters[-1]+0.5*dz,zmax)
            w_z = (zcenters-0.5*dz > zmin) & (zcenters+0.5*dz <= zmax)
            dndz=dndz[w_z]
            zcenters=zcenters[w_z]
            zbins = np.linspace(zmin,zmax,len(zcenters)+1)
        elif distribution=='target_selection':
            dist = Table.read(os.path.join(os.path.dirname(desisim.__file__),'data/redshift_dist_chaussidon2022.ecsv')) 
            dz=dist['z'][1]-dist['z'][0]
            factor=0.1/dz # Account for redshift bin width difference.
            zbins = np.linspace(0,10,100+1)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            dndz=factor*np.interp(zcenters,dist['z'],dist['dndz_23'],left=0,right=0)
        elif distribution=='from_data':
            raise NotImplementedError(f"Option {distribution} is not implemented yet")
        
            if self.data is None:
                raise ValueError("No data catalog was provided")
            dz = 0.1
            zbins = np.linspace(zmin,zmax,int((zmax-zmin)/dz)+1)
            data_area = SurveyRelease.get_catalog_area(self.data,nside=256)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            dndz=np.histogram(self.data['Z'],bins=zbins,weights=np.repeat(1/data_area,len(self.data)))[0]
        else:
            raise ValueError(f"Distribution option {distribution} not in available options: SV, target_selection, from_data")
        mock_area=SurveyRelease.get_catalog_area(self.mockcatalog,nside=16)
        dndz_mock=np.histogram(self.mockcatalog['Z_QSO_RSD'],bins=zbins,weights=np.repeat(1/mock_area,len(self.mockcatalog)))[0]
        fractions=np.divide(dndz,dndz_mock,where=dndz_mock>0,out=np.zeros_like(dndz_mock))
        fractions[fractions>1]=1

        local_z = self.mockcatalog['Z_QSO_RSD']
        bin_index = np.digitize(local_z, zbins) - 1
        rand = np.random.uniform(size=local_z.size)
        if self.invert:
            rand = 1-rand
        selection = rand < fractions[bin_index]
        log.info(f"Keeping {sum(selection)} mock QSOs following the distribution in {distribution}")
        self.mockcatalog=self.mockcatalog[selection]
        
    def apply_data_geometry(self, release='iron'):
        """ Apply the geometry of a given data release to the mock catalog.
        
        Args:
            data (astropy.table.Table): data catalog.
            release (str,optional): release name. If None, will use the whole DESI footprint.
        """
        mock=self.mockcatalog
        tiles=self.get_lya_tiles(release)
        mask_footprint=desimodel.footprint.is_point_in_desi(tiles,mock['RA'],mock['DEC'])
        mock=mock[mask_footprint]
        log.info(f"Keeping {sum(mask_footprint)}  mock QSOs in footprint TILES")

        # If 'Y5' is selected there is no need to downsample the mock catalog.
        if release!='Y5':
            if release=='iron':
                # TODO: Add submodules to generate the pixelmaps from data.
                log.info(f"Downsampling by NPASSES fraction in {release} release")
                # TODO: Implement this in desimodel instead of local path
                pixmap=Table.read('/global/cfs/cdirs/desi/users/hiramk/desi/quickquasars/sampling_tests/npass_pixmap.fits')
                mock_pixels = hp.ang2pix(1024, np.radians(90-mock['DEC']),np.radians(mock['RA']),nest=True)
                if self.data is None:
                    raise ValueError("No data catalog was provided")
                if 'TARGET_DEC' in self.data.colnames and 'TARGET_RA' in self.data.colnames:
                    data_pixels = hp.ang2pix(1024,np.radians(90-self.data['TARGET_DEC']),np.radians(self.data['TARGET_RA']),nest=True)
                elif 'DEC' in self.data.colnames and 'RA' in self.data.colnames: 
                    data_pixels = hp.ang2pix(1024,np.radians(90-self.data['DEC']),np.radians(self.data['RA']),nest=True)
                else:
                    raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in data catalog")
                
                data_passes = pixmap[data_pixels]['NPASS']
                mock_passes = pixmap[mock_pixels]['NPASS']
                data_pass_counts = np.bincount(data_passes,minlength=8) # Minlength = 7 passes + 1 (for bining)
                mock_pass_counts = np.bincount(mock_passes,minlength=8)
                mock['NPASS'] = mock_passes
                downsampling=np.divide(data_pass_counts,mock_pass_counts,out=np.zeros(8),where=mock_pass_counts>0)
                rand = np.random.uniform(size=len(mock))
                if self.invert:
                    rand = 1-rand
                selection = rand<downsampling[mock_passes]
                log.info(f"Keeping {len(mock[selection])} mock QSOs out of {len(mock)}")
                mock = mock[selection]
            else:
                raise NotImplementedError(f"Release {release} is not implemented yet")
        self.mockcatalog=mock  

    def assign_rband_magnitude(self,from_data=False):
        """Assign r-band magnitudes to the catalog according to the distribution
        
        Args:
            from_data (bool): if True, will use the data catalog to generate the distribution. Otherwise, will use the distribution in the default file.
        """
        if not from_data:
            filename=os.path.join(os.path.dirname(desisim.__file__),'data/dn_dzdM_EDR.fits')
            with fitsio.FITS(filename) as fts:
                zcenters=fts['Z_CENTERS'].read()
                rmagcenters=fts['RMAG_CENTERS'].read()
                dn_dzdm=fts['dn_dzdm'].read()
            dz = zcenters[1]-zcenters[0]
            log.info(f"Generating random magnitudes according to distribution in {filename}")
        else:
            if self.data is None:
                raise ValueError("No data catalog was provided")
            dz = 0.1
            zbins = np.linspace(0,10,int(10/dz)+1)
            zcenters=0.5*(zbins[1:]+zbins[:-1])
            rmagbins = np.linspace(15,25,100+1)
            rmagcenters=0.5*(rmagbins[1:]+rmagbins[:-1])
            log.info("Generating random magnitudes according to distribution in data catalog")
            if 'RMAG' in self.data.colnames:
                data_rmag = self.data['RMAG']
            elif 'FLUX_R' in self.data.colnames:
                data_rmag = 22.5-2.5*np.log10(self.data['FLUX_R'])
            else:
                raise ValueError("No magnitude information in data catalog")
            dn_dzdm=np.histogram2d(self.data['Z'],data_rmag,bins=(zbins,rmagbins))[0]
        cdf=np.cumsum(dn_dzdm,axis=1)
        cdf_norm=cdf/cdf[:,-1][:,None]
        mags=np.zeros(len(self.mockcatalog))
        for i,z_bin in enumerate(zcenters):
            w_z = (self.mockcatalog['Z'] > z_bin-0.5*dz) & (self.mockcatalog['Z'] <= z_bin+0.5*dz)
            if np.sum(w_z)==0: continue
            rand = np.random.uniform(size=np.sum(w_z))
            if self.invert:
                rand = 1-rand
            mags[w_z]=np.interp(rand,cdf_norm[i],rmagcenters)  
        if np.sum(mags==0)!=0:
            print(self.mockcatalog[mags==0])
            raise ValueError(f"Generated magnitudes contain zeros")
        self.mockcatalog['FLUX_R'] = 10**((22.5-mags)/2.5)

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
            filename='/global/cfs/cdirs/desi/users/hiramk/desi/quickquasars/sampling_tests/npass_pixmap.fits'
            pixmap=Table.read(filename)
            mock=self.mockcatalog
            if 'TARGET_DEC' in self.data.colnames and 'TARGET_RA' in self.data.colnames:
                data_pixels = hp.ang2pix(1024,np.radians(90-self.data['TARGET_DEC']),np.radians(self.data['TARGET_RA']),nest=True)
            elif 'DEC' in self.data.colnames and 'RA' in self.data.colnames: 
                data_pixels = hp.ang2pix(1024,np.radians(90-self.data['DEC']),np.radians(self.data['RA']),nest=True)
            else:
                raise ValueError("No RA,DEC or TARGET_RA,TARGET_DEC columns in data catalog")
                
            is_lya_data=self.data['Z']>2.1    
            effective_exptime = lambda tsnr2_lrg: 12.25*tsnr2_lrg
            exptime_data = effective_exptime(self.data['TSNR2_LRG'])
            
            exptime_mock = np.zeros(len(mock))
            is_lya_mock = mock['Z']>2.1
            exptime_mock[~is_lya_mock]=1000
            for tile_pass in range(1,8):
                w=pixmap[data_pixels]['NPASS'][is_lya_data] == tile_pass
                pdf=np.histogram(exptime_data[is_lya_data][w]/1000,bins=np.arange(.5,7.5,1),density=True)[0]
                random_variable = rv_discrete(values=(np.arange(1,len(pdf)+1),pdf))
                is_pass = self.mockcatalog['NPASS'] == tile_pass
                exptime_mock[is_lya_mock&is_pass]=1000*random_variable.rvs(size=np.sum(is_pass&is_lya_mock))
            self.mockcatalog['EXPTIME']=exptime_mock

    @staticmethod
    def get_lya_tiles(release='Y5'):
        """Return the tiles that have been observed in a given release.
        
        Args:
            release (str,optional): release name. If None, will return the tiles in the whole DESI footprint.
            
        Returns:
            astropy.table.Table: Table containing the tiles.
        """
        if release !='Y5':
            surveys=["main"]
            if not release.upper()=='FUGU':
                tiles_filename = os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'{release}/tiles-{release}.fits')
                tiles=Table.read(tiles_filename)

            else:
                tiles_filename=os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'guadalupe/tiles-guadalupe.fits')
                tiles=Table.read(tiles_filename)
                sv_tiles_filename= os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'fuji/tiles-fuji.fits')
                sv_tiles=Table.read(sv_tiles_filename)
                tiles=vstack([tiles,sv_tiles])
                surveys+=["sv1","sv3"]
            mask_survey=np.isin(np.char.decode(tiles['SURVEY']),np.array(surveys))
            tiles = tiles [mask_survey]
        else:
            from desimodel.io import load_tiles
            tiles = load_tiles()
        
        # Depending on the table version, the column names are different and need to be decoded.
        try: program = np.char.decode(tiles['PROGRAM'])
        except AttributeError: program = tiles['PROGRAM']
        mask_program=(np.char.lower(program)=='dark')
        tiles=tiles[mask_program]

        # RENAME OTHERWISE THE IN_DESI_FOOTPRINT FUNCTION WONT WORK
        if 'TILERA' in tiles.dtype.names:
            tiles.rename_column('TILERA','RA')
        if 'TILEDEC' in tiles.dtype.names:
            tiles.rename_column('TILEDEC','DEC')
        return tiles
    
            
