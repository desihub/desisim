# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals


"""
Utility functions for desisim.  These may belong elsewhere..?
"""

def medxbin(x,y,binsize,minpts=20,xmin=None,xmax=None):
    """
    Compute the median (and other statistics) in fixed bins along the x-axis. 
    """
    import numpy as np
    import scipy as sci

    # Need an exception if there are fewer than three arguments.

    if xmin==None:
        xmin = x.min()
    if xmax==None:
        xmax = x.max()
    print(xmin,xmax)

    nbin = long(sci.ptp(x)/binsize)
    bins = np.linspace(xmin,xmax, nbin)
    idx  = np.digitize(x,bins)
    #print(nbin, bins, idx)

    stats = {'median': np.zeros(nbin), 'sigma': np.zeros(nbin),
             'iqr': np.zeros(nbin)}
    #med = np.zeros(nbin)
    for kk in np.arange(nbin-1)+1:
        npts = len(y[idx==kk])
        if npts>minpts:
            stats['median'][kk] = np.median(y[idx==kk])
            stats['sigma'][kk] = np.std(y[idx==kk])
            stats['iqr'][kk] = np.subtract(*np.percentile(y[idx==kk],
                                                          [75, 25]))

    # Remove bins with too few points.
    good = np.nonzero(stats['median'])
    stats['median'] = stats['median'][good]
    stats['sigma'] = stats['sigma'][good]
    stats['iqr'] = stats['iqr'][good]

    return bins[good], stats

