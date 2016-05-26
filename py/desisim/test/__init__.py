from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

def test_suite():
    """Returns unittest.TestSuite of desisim tests for use by setup.py"""

    #- DEBUG Travis test failures
    return unittest.defaultTestLoader.loadTestsFromNames(
        ['desisim.test.test_io', 'desisim.test.test_obs']
    )
    #- DEBUG Travis test failures

    # from os.path import dirname
    # desisim_dir = dirname(dirname(__file__))
    # print(desisim_dir)
    # return unittest.defaultTestLoader.discover(desisim_dir,
    #     top_level_dir=dirname(desisim_dir))
