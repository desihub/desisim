import unittest

import numpy as np
from astropy.table import Table

from desisim.scripts.build_lya_coverage import exptime_per_target, ntile_per_target, tile_night_lookup
from desisim.survey_release import EXPTIME_FACTOR


def _rows(targetid, tileid, tsnr2=100.):
    d = np.zeros(len(targetid), dtype=[('TARGETID', 'i8'), ('RA', 'f8'), ('DEC', 'f8'), ('TILEID', 'i4'), ('TSNR2_LRG', 'f4')])
    d['TARGETID'] = targetid; d['TILEID'] = tileid; d['RA'] = 10.; d['DEC'] = 5.; d['TSNR2_LRG'] = tsnr2
    return d


class TestBuildLyaCoverage(unittest.TestCase):

    def test_cap_rule_and_duplicates(self):
        #- target 1: 5 tiles before the rule night, 2 after: first 4 + the 2 later ones = 6 observations
        nights = {t: 20240100 + t for t in range(1, 6)}
        nights.update({6: 20250801, 7: 20250802})
        tiles = Table(dict(TILEID=list(nights), LASTNIGHT=list(nights.values())))
        rows = _rows([1] * 7, list(range(1, 8)))
        #- target 2: tile 1 appears twice (two locations), tile 2 once
        rows2 = _rows([2, 2, 2], [1, 1, 2])
        night = tile_night_lookup(tiles)
        self.assertEqual(list(ntile_per_target([rows, rows2], night)[2]), [7, 2])
        #- q = p_reobs = 1: every available tile is observed, up to the cap rule
        ra, dec, ntile, exptime, observed = exptime_per_target([rows, rows2], night, 1.0, 1.0, np.random.default_rng(1))
        self.assertEqual(list(ntile), [7, 2])
        self.assertTrue(observed.all())
        self.assertAlmostEqual(exptime[0], 6 * EXPTIME_FACTOR * 100., places=3)
        self.assertAlmostEqual(exptime[1], 2 * EXPTIME_FACTOR * 100., places=3)


if __name__ == '__main__':
    unittest.main()
