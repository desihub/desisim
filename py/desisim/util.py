# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Utility functions for desisim.  These may belong elsewhere..?
"""

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
