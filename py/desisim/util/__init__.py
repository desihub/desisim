
"""
Tools for setting up the survey
"""

import configuration
from . import positioner
from astropy.io import fits
import numpy as np
import shapely as shape
import shapely.geometry as shapeg
import descartes as desc
import os.path

def plate_dist(theta):
    """
    Returns the radial distance on the plate (mm) given the angle (radians).
    
    Input:
        theta (float): angle from the center of the plate (radians)

    Returns:
        radius (float): position from the center of the plate in mm.
        
    Note:
        This is a fit to the provided data
    """
    p = np.array([8.297E5,-1750.0,1.394E4,0.0])
    radius = 0.0
    for i in range(4):
        radius = theta*radius + p[i]
    return radius
    
def radec2xy(object_ra, object_dec, tile_ra, tile_dec):
    """
    Returns the x,y coordinats of an object on the plate.

    Input:
        object_ra (float) : 1D array, RA coordinates of the object (degrees)
        object_dec (float) : 1D array, dec coordinates of the object (degrees)
        tile_ra (float) : RA position of the center of the tile
        tile_dec (float) : dec position of the center of the tile
    Returns:
        x (float) : 1D array, x position on the focal plane (in mm)
        y (float) : 1D array, y position on the focal plane (in mm)
        
    It takes as an input the ra,dec coordinates ob the object 
    and the ra,dec coordinates of the plate's center.
    """
    object_theta = (90.0 - object_dec)*np.pi/180.0
    object_phi = object_ra*np.pi/180.0
    o_hat0 = np.sin(object_theta)*np.cos(object_phi)
    o_hat1 = np.sin(object_theta)*np.sin(object_phi)
    o_hat2 = np.cos(object_theta)
    
    tile_theta = (90.0 - tile_dec)*np.pi/180.0
    tile_phi = tile_ra*np.pi/180.0
    t_hat0 = np.sin(tile_theta)*np.cos(tile_phi)
    t_hat1 = np.sin(tile_theta)*np.sin(tile_phi)
    t_hat2 = np.cos(tile_theta)
    
    
    #we make a rotation on o_hat, so that t_hat ends up aligned with 
    #the unit vector along z. This is composed by a first rotation around z
    #of an angle pi/2 - phi and a second rotation around x by an angle theta, 
    #where theta and phi are the angles describin t_hat.
    
    costheta = t_hat2
    sintheta = np.sqrt(1.0-costheta*costheta) + 1E-10
    cosphi = t_hat0/sintheta
    sinphi = t_hat1/sintheta
    
    #First rotation, taking into account that cos(pi/2 -phi) = sin(phi) and sin(pi/2-phi)=cos(phi)
    n_hat0 = sinphi*o_hat0 - cosphi*o_hat1
    n_hat1 = cosphi*o_hat0 + sinphi*o_hat1
    n_hat2 = o_hat2
    
    #Second rotation
    nn_hat0 = n_hat0
    nn_hat1 = costheta*n_hat1 - sintheta*n_hat2
    nn_hat2 = sintheta*n_hat1 + costheta*n_hat2
    
    #Now find the radius on the plate
    theta = np.sqrt(nn_hat0*nn_hat0 + nn_hat1*nn_hat1)
    radius = plate_dist(theta)
    x = radius * nn_hat0/theta
    y = radius * nn_hat1/theta
    
    return x,y

def rot_displ_shape(shape_coords, angle=0.0, radius=0.0):
    """
    Rotates a set of points
    Input:
        shape_coords (float) : 2D array. shape_coords[:,0] is X shape_coords[:,1] is Y.
        angle (float): rotation angle around the origin, in degrees.
        radius (float): displacement of the origin, in same units as shape_coords.
    Returns:
       A new array with the coordinates rotated.
    """
    tmp = shape_coords.copy()
    tmp[:,0] = shape_coords[:,0]*np.cos(np.deg2rad(angle)) - shape_coords[:,1]*np.sin(np.deg2rad(angle))
    tmp[:,1] = shape_coords[:,0]*np.sin(np.deg2rad(angle)) + shape_coords[:,1]*np.cos(np.deg2rad(angle))
    tmp[:,0] = tmp[:,0] + radius*np.cos(np.deg2rad(angle))
    tmp[:,1] = tmp[:,1] + radius*np.sin(np.deg2rad(angle))
    return tmp

class Positioner(object):
    """
    Holds the shape and position information for a single positioner.
    
    Atttributes:
        Ferrule_radius (float): in mm
        R1 (float) : distance from central axis to eccentric axis, in mm
        R2 (float) : distance from eccentric axis to ferrule axis, in mm
        Ei (float) : inner clear rotation envelope, in mm
        Eo (float) : outer clear rotation envelope, in mm
        Theta (float): angle of the inner arm, in degrees
        Phi (float): angle of the outer arm, in degrees
    """
    def __init__(self, offset_x = 0.0, offset_y=0.0, Theta=0.0, Phi=0.0, id=0):
        """        
        Args:
            offset_x (float): position on the focal plane in mm.
            offset_y (float): position on the focal plane in mm.
            Theta (float): angle of the inner arm in degrees.
            Phi (float): angle of the outer arm in degrees.
            id (int) : positioner ID
        Note:
            Coordinates are taken from
            https://desi.lbl.gov/trac/browser/code/focalplane/positioner_control/trunk/anticollision/pos_geometry.m
        """
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.ferrule_radius = 1.250/2.0 # mm
        self.R1 = 3.000 # distance from central axis to eccentric axis
        self.R2 = 3.000 # distance from eccentric axis to ferrule axis
        self.Ei = 6.800 # inner clear rotation envelope
        self.Eo = 9.990 # outer clear rotation envelope
        self.Theta = Theta
        self.Phi = Phi
        self.id = id
        
        
        self.lower_pos = np.array(((0.387, 0.990), (0.967,0.410), (0.967, -0.410), (0.387, -0.990), (-0.649, -0.990), 
                    (-1.000, -0.639), (-1.000, 0.639), (-0.649, 0.990)))
        self.lower_pos[:,0]=self.lower_pos[:,0] + self.R1 
        self.lower_pos[:,1]=self.lower_pos[:,1] 
        
        self.upper_pos = np.array(((0.387, -0.990), (0.967, -0.410), (0.967,0.410), (0.387,0.990), (-2.235,0.990), 
                   (-2.668, 1.240), (-3.514, 1.240), (-4.240,0.514), (-4.240,-0.514), (-3.682,-1.072), 
                  (-2.994,-1.339), (-2.944,-1.922), (-2.688, -2.015 ), (-1.981,-1.757 ), (-1.844, -0.990)))
        self.upper_pos[:,0]=self.upper_pos[:,0] + self.R1 
        self.upper_pos[:,1]=self.upper_pos[:,1] 
        
        self.central_pos = np.array((( 4.358 , -2.500), (5.095,-0.474),(5.095,0.605),(4.348,1.792), 
                    (3.000,2.180), (1.652, 1.792), (0.905, 0.605), (0.905 ,-0.356), 
                    (1.759, -2.792), (2.771, -2.500)))
        self.central_pos[:,0] = self.central_pos[:,0] 
        self.central_pos[:,1] = self.central_pos[:,1] 
        
        self.Eo_circ_resn  = 32;
        self.env_pos = np.zeros((self.Eo_circ_resn,2))
        self.env_pos[:,0] = self.Eo/2*np.cos(np.linspace(0,2*np.pi,self.Eo_circ_resn))
        self.env_pos[:,1] = self.Eo/2*np.sin(np.linspace(0,2*np.pi,self.Eo_circ_resn))
        
        #move to Theta and Phi
        #first rotate phi
        self.upper_pos = rot_displ_shape(self.upper_pos, angle=self.Phi) 
        self.lower_pos = rot_displ_shape(self.lower_pos, angle=self.Phi)
        #offset the central axis 
        self.upper_pos = rot_displ_shape(self.upper_pos, angle=0, radius=self.R1)
        self.lower_pos = rot_displ_shape(self.lower_pos, angle=0, radius=self.R1)

        #now rotathe theta
        self.upper_pos = rot_displ_shape(self.upper_pos, angle=self.Theta) 
        self.lower_pos = rot_displ_shape(self.lower_pos, angle=self.Theta)
        self.central_pos = rot_displ_shape(self.central_pos, angle=self.Theta)
        
        #final offset
        self.upper_pos[:,0]=self.upper_pos[:,0] + self.offset_x
        self.upper_pos[:,1]=self.upper_pos[:,1] + self.offset_y
        self.central_pos[:,0]=self.central_pos[:,0] + self.offset_x
        self.central_pos[:,1]=self.central_pos[:,1] + self.offset_y
        self.lower_pos[:,0]=self.lower_pos[:,0] + self.offset_x
        self.lower_pos[:,1]=self.lower_pos[:,1] + self.offset_y
        self.env_pos[:,0]=self.env_pos[:,0] + self.offset_x
        self.env_pos[:,1]=self.env_pos[:,1] + self.offset_y

    def add_plot_positioner(self, ax=None): 
        """
        Adds a plot of the positioner to the plotting axis defined by ax.
        """
        up_poly = shapeg.Polygon(positioner.upper_pos)
        central_poly= shapeg.Polygon(positioner.central_pos)
        low_poly= shapeg.Polygon(positioner.lower_pos)
        env_poly = shapeg.Polygon(positioner.env_pos)
    
        patch_u = desc.patch.PolygonPatch(up_poly, facecolor='yellow', edgecolor='yellow', alpha=0.5, zorder=2)
        patch_c = desc.patch.PolygonPatch(central_poly, facecolor='blue', edgecolor='blue', alpha=0.5, zorder=2)
        patch_l = desc.patch.PolygonPatch(low_poly, facecolor='red', edgecolor='red', alpha=0.5, zorder=2)
        patch_e = desc.patch.PolygonPatch(env_poly, facecolor='white', edgecolor='black', alpha=0.2, zorder=2)

        ax.add_patch(patch_e)
        ax.add_patch(patch_u)
        ax.add_patch(patch_c)
        ax.add_patch(patch_l)


class FocalPlaneFibers(object):
    """
    Keeps the relevant information to position fibers on the focal plane

    Attributes:
        The properties initialized in the __init__ procedure:
        x_focal (float) : array for the x_positions on the focal plane, in mm
        y_focal (float) : array for the y_positions on the focal plane, in mm        
        z_focal (float) : array for the y_positions on the focal plane, in mm        
        fiber_id (int) :
        positioner_id (int) : 
        spectrograph_id (int) : 
        neighbors (int) : 2D array of shape (n_fibers, 6) holding the fiber of the 6 nearest fibers.
        n_fiber (int) : total number of fibers
    """

    def __init__(self, filename):
        hdulist = fits.open(filename)        
        self.filename = filename
        self.x_focal = hdulist[1].data['x']
        self.y_focal = hdulist[1].data['y']
        self.z_focal = hdulist[1].data['z']
        self.fiber_id = hdulist[1].data['fiber']
        self.positioner_id = hdulist[1].data['positioner']
        self.spectrograph_id = hdulist[1].data['spectrograph']
        self.neighbors = np.zeros((np.size(self.x_focal), 6), dtype='i4') 
        self.n_fiber = np.size(self.x_focal)

        for i in range(self.n_fiber):
            x = self.x_focal[i]
            y = self.y_focal[i]
            radius = np.sqrt((self.x_focal -x )** 2 + (self.y_focal - y)**2)
            ids = radius.argsort()
            self.neighbors[i,:] = ids[1:7]
        

        # This section is related to targets
        self.available_targets = [None] * self.n_fiber
        self.n_targets = np.zeros(self.n_fiber)
        self.target = -1 * np.ones(self.n_fiber, dtype=np.int64)

        # We use this object to import all the positioner geometry variable
        self.positioner = Positioner()

    def set_available(self, position, ID_list):
        """
        Set-up the list and number of available targets to this positioner
         
        Args:
             position (int): position in the list to be updated
             ID_list (int): array of available galaxies
             
        """
        self.available_targets[position] = ID_list.copy()
        self.n_targets[position] = np.size(self.available_targets)

    def reset_available(self, position):
        """
        Reset the list and number of available targets to this positioner
         
        Args:
             position (int): position in the list to be updated
        """
        self.available_targets[position] = None
        self.n_targets[position] = 0

    def reset_all_available(self):
        """
        Resets the list and number of available targets to this positioner
        """
        self.available_targets = [None] * self.n_fiber
        self.n_targets = np.zeros(self.n_fiber)

    def set_target(self, position, target_id):
        """
        Sets the id of the target assigned to this positioner
        Args:
             position (int): position in the list to be updated
            target_id (int): id of the target assigned to this positioner
        """
        self.target[position]  = target_id


    def reset_target(self, position):
        """
        resets the id of the target assigned to this positioner
        Args:
             position (int): position in the list to be updated
            target_id (int): id of the target assigned to this positioner
        """
        self.target[position]  = -1

    def reset_all_targets(self):
        """
        resets the id of the target assigned to this positioner
        """
        self.target = -1 * np.ones(self.n_fiber,  dtype=np.int64)



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
