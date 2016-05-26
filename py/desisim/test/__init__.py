from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

def test_suite():
    """Returns unittest.TestSuite of desisim tests for use by setup.py"""

    #- DEBUG Travis test failures
    return unittest.defaultTestLoader.loadTestsFromNames([
        # 'desisim.test.test_batch',      #- OK
        # 'desisim.test.test_io',         #- OK
        # 'desisim.test.test_obs',        #- OK
        'desisim.test.test_pixsim',
        # 'desisim.test.test_quickcat',   #- OK
        # 'desisim.test.test_targets',    #- OK
        # 'desisim.test.test_templates',  #- OK
        # 'desisim.test.test_top_level',  #- OK
        ])
    #- DEBUG Travis test failures

    # from os.path import dirname
    # desisim_dir = dirname(dirname(__file__))
    # print(desisim_dir)
    # return unittest.defaultTestLoader.discover(desisim_dir,
    #     top_level_dir=dirname(desisim_dir))
