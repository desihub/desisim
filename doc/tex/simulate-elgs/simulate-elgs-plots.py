#!/usr/bin/env python

"""
Document me.
"""

from __future__ import division, print_function

import os
import sys
import numpy as np

from desisim.io import read_base_templates
import matplotlib.pyplot as plt

qapath = os.getenv('DESISIM')

# Read the ELGs metadata table and continuum spectra.
obsflux, obswave, obsmeta = read_base_templates()
cflux, cwave, cmeta = read_base_templates(observed=True)

plt.plot(cmeta['D4000'],cmeta['OII_3727_EW'],'bo')
plt.savefig(

