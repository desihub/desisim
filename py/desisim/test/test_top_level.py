# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
"""
test top-level desisim functions
"""
#
from __future__ import absolute_import, division, print_function
#
import unittest
import re
from .. import __version__ as theVersion
#
class TestTopLevel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.versionre = re.compile(r'([0-9]+!)?([0-9]+)(\.[0-9]+)*((a|b|rc|\.post|\.dev)[0-9]+)?')

    @classmethod
    def tearDownClass(cls):
        pass

    def test_version(self):
        """Ensure the version conforms to PEP386/PEP440.
        """
        #- python 3.5 has deprecated assertRegexpMatches in favor of
        #- assertRegexp, but that doesn't exist in python 2.7.  Keep for
        #- now until we've fully transitioned to python 3.5
        self.assertRegexpMatches(theVersion,self.versionre)

if __name__ == '__main__':
    unittest.main()
