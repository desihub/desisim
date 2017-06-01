"""
desisim.util
============

Utility functions for desisim.  These may belong elsewhere?
"""

from __future__ import print_function, division
import numpy as np


#- Experimental
class _FakeMPIComm(object):
    '''
    Provides a fake MPI communicator with a very small subset of the actual
    functions that mpi4py.MPI.COMM_WORLD provides.  This can be used by
    programs that want to gracefully fall back from MPI to serial code if they
    only need to get rank, size, and do barriers.
    '''
    def __init__(self):
        '''creates a fake MPI communicator'''
        self._size = 1
        self._rank = 0

    @property
    def size(self):
        return self._size

    @property
    def rank(self):
        return self._rank

    def Get_size(self):
        '''mimics a real MPI communicator; returns 1'''
        return self._size

    def Get_rank(self):
        '''mimics a real MPI communicator; returns 0'''
        return self._rank

    def Barrier(self):
        '''mimics a real MPI communicator, but doesn't do anything here'''
        pass

    def Abort(self):
        '''Stop hard'''
        import sys
        sys.exit(1)

def spline_medfilt2d(image, kernel_size=201):
    '''
    Returns a 2D spline interpolation of a median filtered input image
    '''
    if 3*kernel_size > min(image.shape):
        raise ValueError(
            'kernel_size {} must be < min image shape {}//3'.format(
                kernel_size, min(image.shape)))

    from scipy.interpolate import RectBivariateSpline
    n = kernel_size // 2
    xx = np.arange(n, image.shape[1], kernel_size)
    yy = np.arange(n, image.shape[0], kernel_size)
    zz = np.zeros((len(yy), len(xx)))
    for i,x in enumerate(xx):
        for j,y in enumerate(yy):
            xy = np.s_[y-n:y+n+1, x-n:x+n+1]
            zz[j,i] = np.median(image[xy])

    #- adjust spline order for small test data
    kx = min(3, len(xx)-1)
    ky = min(3, len(yy)-1)
    s = RectBivariateSpline(xx, yy, zz, kx=kx, ky=ky)
    background = s(np.arange(image.shape[0]), np.arange(image.shape[1]))

    return background

def medxbin(x,y,binsize,minpts=20,xmin=None,xmax=None):
    """
    Compute the median (and other statistics) in fixed bins along the x-axis.
    """
    import numpy as np
    from scipy import ptp

    # Need an exception if there are fewer than three arguments.

    if xmin==None:
        xmin = x.min()
    if xmax==None:
        xmax = x.max()
    #print(xmin,xmax)

    nbin = int(ptp(x)/binsize)
    bins = np.linspace(xmin,xmax,nbin)
    idx  = np.digitize(x,bins)
    #print(nbin, bins, xmin, xmax)

    stats = np.zeros(nbin,[('median','f8'),('sigma','f8'),('iqr','f8')])
    for kk in np.arange(nbin):
        npts = len(y[idx==kk])
        if npts>minpts:
            stats['median'][kk] = np.median(y[idx==kk])
            stats['sigma'][kk] = np.std(y[idx==kk])
            stats['iqr'][kk] = np.subtract(*np.percentile(y[idx==kk],[75, 25]))

    # Remove bins with too few points.
    good = np.nonzero(stats['median'])
    stats = stats[good]

    return bins[good], stats

#- TODO: move to desiutil
def dateobs2night(dateobs):
    '''
    Convert UTC dateobs to KPNO YEARMMDD night string

    Args:
        dateobs:
            float -> interpret as MJD
            str -> interpret as ISO 8601 YEAR-MM-DDThh:mm:ss.s string
            astropy.time.Time -> UTC
            python datetime.datetime -> UTC

    TODO: consider adding format option to pass to astropy.time.Time without
        otherwise questioning the dateobs format
    '''
    import astropy.time
    import datetime
    if isinstance(dateobs, float):
        dateobs = astropy.time.Time(dateobs, format='mjd')
    elif isinstance(dateobs, datetime.datetime):
        dateobs = astropy.time.Time(dateobs, format='datetime')
    elif isinstance(dateobs, str):
        dateobs = astropy.time.Time(dateobs, format='isot')
    elif not isinstance(dateobs, astropy.time.Time):
        raise ValueError('dateobs must be float, str, datetime, or astropy time object')

    import astropy.units as u
    kpno_time = dateobs - 7*u.hour

    #- "night" rolls over at local noon, not midnight, so subtract another 12 hours
    yearmmdd = (kpno_time - 12*u.hour).isot[0:10].replace('-', '')
    assert len(yearmmdd) == 8

    return yearmmdd
