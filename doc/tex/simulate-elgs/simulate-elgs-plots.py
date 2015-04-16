#!/usr/bin/env python

"""
Document me.
"""

from __future__ import division, print_function

import os
import sys
import scipy as sci
import triangle
import numpy as np

from desisim.io import read_base_templates
from desisim.util import medxbin
import matplotlib.pyplot as plt

qadir = os.path.join(os.getenv('DESISIM'),'doc','tex',
                     'simulate-elgs','figures')

# Read the ELGs metadata table and continuum spectra.
obsflux, obswave, obsmeta = read_base_templates()
cflux, cwave, cmeta = read_base_templates(observed=True)

# 
d4000 = cmeta['D4000']
ewoii = np.log10(cmeta['OII_3727_EW'])
bins, stats = medxbin(d4000,ewoii,0.05,minpts=100)
coeff = sci.polyfit(bins,stats['median'],2)
print(stats['median'], stats['sigma'])

# build the plot

plt.ioff()
fig = plt.figure(figsize=(8,7))
hist2d = triangle.hist2d(d4000, ewoii)
plt.xlim([1.0,1.8])
plt.xlabel('D$_{n}$(4000)',fontsize=18)
plt.ylabel('EW([O II] $\lambda\lambda3726,29$) ($\AA$, rest-frame)',
           fontsize=18)
plt.errorbar(bins,stats['median'],yerr=stats['sigma'],fmt='bo',markersize=8)
plt.plot(bins,sci.polyval(coeff,bins),color='red')
#hist2d = triangle.hist2d(np.array([d4000,ewoii]).transpose())
#plt.plot(d4000,ewoii,'bo')
fig.savefig(qadir+'/d4000_ewoii.pdf')
