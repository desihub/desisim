# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np

"""
Utility functions for desisim.  These may belong elsewhere..?
"""

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
            
    s = RectBivariateSpline(xx, yy, zz)
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

    nbin = long(ptp(x)/binsize)
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
