"""
Functions for working on simulated nights of observation. 
"""

import ephem as eph
import numpy as np

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
