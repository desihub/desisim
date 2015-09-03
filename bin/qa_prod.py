#!/usr/bin/env python

"""
Generate QA for a DESI spectrograph production run that was 
generated from simulated DESI images.

JXP, UCSC
September 2015
"""

import sys
import os
import numpy as np
import optparse
import random
import time

from desisim.spec_qa import high_level
from desispec.log import get_logger
log = get_logger()

#- Parse options
parser = optparse.OptionParser(
    usage = "%prog [options]",
    epilog = "See $SPECTER_DIR/doc/datamodel.md for input format details"
    )
        
parser.add_option("--no_highlevel", type=int, help='Suppress high-level QA generation')

opts, args = parser.parse_args()

#- Check environment
envOK = True
for envvar in ('DESI_SPECTRO_REDUX', 'PRODNAME', 'DESI_SPECTRO_DATA'):
    if envvar not in os.environ:
        log.fatal("${} is required".format(envvar))
        envOK = False
if not envOK:
    print "Set those environment variable(s) and then try again"
    sys.exit(1)

if opts.no_highlevel is None: 
    high_level.main()
    sys.exit(0)

