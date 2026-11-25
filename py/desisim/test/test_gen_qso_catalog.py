import unittest

from desisim.scripts.gen_qso_catalog import parse, resolve_release


class _Coverage:
    def __init__(self, release=None):
        self.meta = {} if release is None else {'RELEASE': release}


class TestGenQsoCatalogOptions(unittest.TestCase):

    base = ['-i', 'master.fits', '-o', 'out.fits', '--seed', '1']

    def test_default_is_tiles_without_coverage(self):
        #- command lines of the existing mock productions keep the tiles method and loa
        args = parse(self.base + ['--input-data', 'data.fits'])
        self.assertEqual(args.selection_method, 'tiles')
        self.assertEqual(args.release, 'loa')
        args = parse(self.base + ['--release', 'Y5'])
        self.assertEqual((args.selection_method, args.release), ('tiles', 'Y5'))

    def test_default_is_randoms_with_coverage(self):
        args = parse(self.base + ['--lya-coverage', 'cov.fits'])
        self.assertEqual(args.selection_method, 'randoms')
        self.assertIsNone(args.release)

    def test_randoms_needs_coverage(self):
        with self.assertRaises(SystemExit):
            parse(self.base + ['--selection-method', 'randoms'])

    def test_release_from_coverage(self):
        self.assertEqual(resolve_release(None, _Coverage('matterhorn')), 'matterhorn')
        self.assertEqual(resolve_release('matterhorn', _Coverage('matterhorn')), 'matterhorn')
        self.assertEqual(resolve_release('matterhorn', _Coverage()), 'matterhorn')
        with self.assertRaises(ValueError):
            resolve_release('loa', _Coverage('matterhorn'))
        with self.assertRaises(ValueError):
            resolve_release(None, _Coverage())


if __name__ == '__main__':
    unittest.main()
