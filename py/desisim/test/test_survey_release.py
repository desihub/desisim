import os
import unittest
from importlib import resources

import numpy as np

from desisim.survey_release import get_lya_tiles

class TestSurveyRelease(unittest.TestCase):

    def test_get_lya_tiles_y5(self):
        #- Y5 (default) just returns all DARK tiles from load_tiles(), no
        #- redux/tiles-{release}.fits file needed. Point DESI_SURVEYOPS at
        #- the bundled fixture explicitly so this test is deterministic even
        #- when a real (much larger) $DESI_SURVEYOPS snapshot is present.
        orig = os.environ.get('DESI_SURVEYOPS')
        os.environ['DESI_SURVEYOPS'] = str(resources.files('desisim').joinpath(
            'test', 'data', 'surveyops'))
        try:
            tiles = get_lya_tiles(release='Y5')
        finally:
            if orig is None:
                os.environ.pop('DESI_SURVEYOPS', None)
            else:
                os.environ['DESI_SURVEYOPS'] = orig

        self.assertTrue(np.all(tiles['PROGRAM'] == 'DARK'))
        self.assertEqual(set(tiles['TILEID']), set(range(1001, 1011)))

if __name__ == '__main__':
    unittest.main()
