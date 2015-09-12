"""
Functions for working on simulated nights of observation. 
"""


import ephem as eph
import numpy as np
import desimodel.io

_observer = None

def init_observer(long=-111.5967, lat=31.9583, date_time='2018/1/2 2:00'):
    """
    Initializes the observer position and date. Returns an
    ephen.Observer() object.  
    
    Args:
        long (float), longitude in degrees.
        lat (float), latitude in degrees.        
        date_time (string), date and time. default value is '2018/1/2 2:00'
        Default values for long and lat correspond to Kitt Peak.
    Returns:
        ephem.Observer() object.
        
    """
    global _observer
    _observer = eph.Observer()
    _observer.long = str(long)
    _observer.lat = str(lat)
    _observer.date = date_time
    return _observer

def radec_zenith():
    """
    Return RA,Dec at zenith.    
    Returns:
        RA (ephem.Angle) 
        Dec (ephem.Angle): 
    """
    altitude = 90.0
    azimuth = 0.0
    ra_zenith , dec_zenith = _observer.radec_of(str(azimuth), str(altitude))
    return ra_zenith, dec_zenith

def days_to_next_full_moon():
    """
    Compute the number of days to the next full moon
    Returns:
        n_days (float): number of days.
    """
    d1 = eph.next_new_moon(_observer.date)
    n_days = d1 - _observer.date 
    return n_days

def separation_azimuth_moon():
    """
    Compute the angle between azimuth and the Moon.
    Returns:
        a (ephem.Angle): angle between Moon and Azimuth.
    """
    up_ra, up_dec = radec_zenith()
    moon = eph.Moon()
    moon.compute(_observer.date)
    a = eph.separation([up_ra,up_dec], [moon.ra, moon.dec])
    return a

def points_inside_azimuth_cone(ra_list, dec_list, id_list, angle_to_zenith):
    """
    Selects a subset of points close to the zenith
    
    Args:
        ra_list (float): np.array with RA angles in degrees.
        dec_list (float): np.array with Dec angles in degrees.
        id_list (int): np.array with integers indexing the RA, DEC values
        angle_to_zenith (float): angle in degrees to zenith below which the list points are selected.
    Returns:
        The subset of points is represented by three arrays.
        ra_sublist (float): np.array with RA angles in degrees.
        dec_sublist (float): np.array with Dec angles in degrees.
        id_sublist (int): np.array with integers indexing the RA, DEC values.
    """
    assert angle_to_zenith >= 0.0, "Angle to zenith must be positive"
    assert angle_to_zenith <= 360.0, "Angle to zenith must be less than 360 degrees"
    
    ra_sublist = np.empty((0))
    dec_sublist = np.empty((0))
    id_sublist = np.empty((0), dtype='int')
    n_tiles = np.size(id_list)

    ra_list_radians = ra_list * np.pi/180.0
    dec_list_radians = dec_list * np.pi/180.0
    
    ra_up, dec_up = radec_zenith()
    for i_tile in np.arange(n_tiles):
        a = eph.separation([ra_up,dec_up], [ra_list_radians[i_tile], dec_list_radians[i_tile]])
        if(float(a) < (angle_to_zenith*np.pi/180.0)):
            id_sublist = np.append(id_sublist, id_list[i_tile])
            ra_sublist = np.append(ra_sublist, ra_list[i_tile])
            dec_sublist = np.append(dec_sublist, dec_list[i_tile])
    return ra_sublist, dec_sublist, id_sublist
