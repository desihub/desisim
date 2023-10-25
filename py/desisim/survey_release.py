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
from astropy.io import fits
from astropy.table import Table, vstack
from desiutil.log import get_logger
import desimodel.footprint


log = get_logger()

def get_lya_tiles(release):
    """
    Release observed DARK TILES containing Lyman-alpha.
    
    Args:
        release (str): DESI's release to get the tiles from. 
    Returns:
        tiles: Astropy table containing the observed tiles
    """
    surveys=[b"main"]
    if release.upper() != 'FUGU':
        ifile=get_tiles_filename(release.lower())
        tiles=Table.read(ifile)

    else:
        ifile_guadalupe=get_tiles_filename('guadalupe')
        ifile_fuji=get_tiles_filename('fuji')
        tiles=Table.read(ifile_guadalupe)
        sv_tiles=Table.read(ifile_fuji)
        tiles=vstack([tiles,sv_tiles])
        surveys+=[b"sv1",b"sv3"]

    mask_program=(tiles['PROGRAM']=='dark')
    mask_survey=np.isin(tiles['SURVEY'],np.array(surveys))
    tiles=tiles[mask_program&mask_survey]

    # RENAME OTHERWISE THE IN_DESI_FOOTPRINT FUNCTION WONT WORK
    tiles.rename_column('TILERA','RA')
    tiles.rename_column('TILEDEC','DEC')
    return tiles

def get_tiles_filename(release):
    filename=os.path.join(os.environ['DESI_SPECTRO_REDUX'],f'{release}/tiles-{release}.fits')
    return filename
    



class SurveyRelease(object):
    """
    Class with functions to reproduce a DESI release.
    In terms of footprint, object density and exposures.
    
    Args:
        release (str): DESI's release to be reproduced.
        pixel (int): Input HPXPIXEL to get density and number of observations from.
        nside (int): NSIDE for the sky.
        hpxnest (int): NEST for the sky.
    """
    def __init__(self,release,nside,hpxnest):
        survey_releases_file=os.path.join(os.path.dirname(desisim.__file__),'data/releases_pixmap.fits')
        self.fname = survey_releases_file
        self.release = release
        self.nside_input = nside
        self.pixarea_input = hp.nside2pixarea(self.nside_input,degrees=True)
        self.nest_input = hpxnest
        self.tiles = get_lya_tiles(release)
        self.target_density_map,self.numobs_prob = self.release_info()
        
    def release_info(self):
        log.info(f'Reading {self.release} density pixel map from file {self.fname}')
        hdul = fits.open(self.fname)
        if self.release.upper() not in hdul:
            raise ValueError(f'{self.release} not in an HDU of {self.fname}')
        data = Table(hdul[self.release.upper()].data)
        hdr = hdul[self.release.upper()].header
        if 'NEST' not in hdr.keys():
            log.warning(f"NEST not specified by file {self.fname}, assuming nested")
            self.nest_file=True
        else:
            self.nest_file=hdr['NEST']
            
        if self.nest_input != self.nest_file:
            raise ValueError(f'NEST option from file {self.fname} does not match NEST from transmission files')
            
        if 'NSIDE' not in hdr.keys():
            log.warning(f"NSIDE not specified by file {self.fname}, getting from pixel map")
            npix=len(data)
            self.nside_file=hp.npix2nside(npix)
        else:
            self.nside_file=hdr['NSIDE']
            
        if self.nside_input!=self.nside_file:
            # Nsides are not necesary the same, log if its not to be aware of this.
            log.warning(f'Transmission nside={self.nside_input}, is different from density map nside={self.nside_file}')
            
        self.pix_area_file = hp.nside2pixarea(self.nside_file,degrees=True)
        density = data['z<2.1','z>2.1']
        prob_numobs = data['NUMOBS_PROBABILITY']
        return density,prob_numobs
    
    @staticmethod
    def get_pixels(ra,dec,nside,nest=True):
        """
        Get the pixels covered by the mock in the nside of the density map file
        
        Args:
            ra (ndarray): Objects right ascension.
            dec (ndarray): Objects declination.
            nside (int): Healpix nside
            nest (bool): Healpix nest
        Returns:
            pixels (ndarray): Pixels array in density map file nside 
        """
        pixels = hp.ang2pix(nside, np.radians(90-dec),np.radians(ra),nest=nest)
        return pixels
    
    @staticmethod
    def _zmask_dict(z):
        is_lya = z>=2.1
        zmask_dict = {'z>2.1':is_lya,
                       'z<2.1':~is_lya}
        return zmask_dict
    
    @staticmethod
    def get_density_map(data,nside,nest=True):
        """
        Generate pixel density map
        
        Args:
            data (ndarray): Table of objects containing RA, DEC and Z.
            nside (int): Healpix nside
            nest (bool): Healpix nest
        Returns:
            pixel_map (ndarray): Pixels array in density map file nside 
        """
        num_pixels = hp.nside2npix(nside)
        
        pixel_map={'HPXPIXEL':np.arange(num_pixels),
                   'z<2.1':np.zeros(num_pixels),
                   'z>2.1':np.zeros(num_pixels)}
        pixel_area = hp.pixelfunc.nside2pixarea(nside, degrees=True)
        pixels = SurveyRelease.get_pixels(data['RA'],data['DEC'],nside=nside,nest=nest)
        zmask_dict=SurveyRelease._zmask_dict(data['Z'])
        
        for zbin, zmask in zmask_dict.items():
            unique_pixels,pixel_count=np.unique(pixels[zmask],return_counts=True)
            density=pixel_count/pixel_area 
            pixel_map[zbin][unique_pixels]=density

        return pixel_map
    
    def apply_geometry(self,metadata):
        """ Selects QSOs to reproduce a data release.
        Args:
            metadata (ndarray): Metadata of objects. Containing RA, DEC and Z.
        Returns:
            selection (ndarray): mask to apply to downsample input list
        """
        ids_initial = metadata['MOCKID']
        in_tiles = desimodel.footprint.is_point_in_desi(self.tiles,metadata['RA'],metadata['DEC'])
        if np.count_nonzero(in_tiles)==0: 
            log.warning(f"No intersection with {self.release} footprint")
            return in_tiles # Return all false to exit QSO generation cycle.
        
        metadata_intiles = metadata[in_tiles]
        # Get pixel in density map nside for zbin
        mock_pixels = SurveyRelease.get_pixels(metadata_intiles['RA'],metadata_intiles['DEC'],
                                               nside=self.nside_file,nest=self.nest_file)
        unique_pixels = np.unique(mock_pixels)
        log.info(f"Selecting QSOs in pixels index {unique_pixels} of nside={self.nside_file}")
        target_density_map = self.target_density_map   
        
        # Downsample objects 
        selection=list()
        zmask_dict = SurveyRelease._zmask_dict(metadata_intiles['Z'])
        for z_mask_name, z_mask in zmask_dict.items():
            N_targets = target_density_map[z_mask_name] 
            if N_targets==0: continue
            indices=np.where(z_mask)[0]
            N_avail = np.count_nonzero(z_mask)
            density_avail  = N_avail/self.pixarea_input
            downsampling_fraction = N_targets/density_avail
            selected_indices=np.random.uniform(size=N_avail)<downsampling_fraction[mock_pixels][z_mask]
            selection.extend(indices[selected_indices])
            
        # Make sure objects are not repeated
        assert len(selection)==len(np.unique(selection))
        ids_selected = metadata_intiles['MOCKID'][selection]
        # Get the selected objects from the original metadata
        selection = np.isin(ids_initial,ids_selected)
        return selection
                
    def assign_exposures(self,metadata):
        
        """
        Assign one exposure to z<2.1 QSOs and random exposure times to z>2.1 QSOs based on survey release.
        Args:
            metadata (ndarray): Metadata of objects. Containing RA, DEC and Z.
        Returns:
            exptime (ndarray): array of exposure time for each quasar.
        """

        exptime = np.full(len(metadata),1000)
        z=metadata['Z']
        is_lya = z>=2.1
        num_lya = np.count_nonzero(is_lya)
        if num_lya!=0:
            mock_pixels = SurveyRelease.get_pixels(metadata['RA'],metadata['DEC'],
                                          nside=self.nside_file,nest=self.nest_file)
            unique_pixels = np.unique(mock_pixels)
            for pixel in unique_pixels:
                numobs_distribution = self.numobs_prob[pixel]
                
                isnan_numobs = np.isnan(numobs_distribution)
                if isnan_numobs.sum()!=0:
                    # Avoid NaNs (for some reason) in distribution
                    log.warning(f"Probability distribution in pixel {pixel} of nside={self.nside_file} has NaNs, setting to zero.")
                    numobs_distribution= np.nan_to_num(numobs_distribution, nan=0)
                 
      
                if numobs_distribution.sum()==0:
                    log.warning(f"All probabilities in pixel {pixel} of nside={self.nside_file} are zero, assigning one exposure")
                    numobs_distribution[0]=1.
            
                numobs = np.arange(1,len(numobs_distribution)+1)
                in_pixel = mock_pixels==pixel
                
                num_lya_pixel = np.count_nonzero(is_lya&in_pixel)
                log.info(f'Assigning {numobs} exposures with probabilities {numobs_distribution} to {num_lya_pixel} z>2.1 qsos')
                exptime_lya = 1000*np.random.choice(numobs,size=num_lya_pixel,p=numobs_distribution)
                exptime[is_lya&in_pixel]=exptime_lya
        return exptime
