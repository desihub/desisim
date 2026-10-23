"""
Generate the quasar catalog to input quickquasars: select from a master catalog of mock quasars the targets of a DESI release.

The redshifts of the QSO targets follow the target-selection distribution of Chaussidon et al. 2022, not the data.
The observed data catalog (--input-data) gives the r-band magnitude distribution (dN/dz dM), the redshift distribution of the
non-QSO targets (--include-nonqso-targets) and, with --selection-method tiles, the number of targets and the exposure times per NPASS class.

Footprint, downsampling and exposure times come from the Lya coverage of the release (--selection-method randoms, the default; the
coverage is built once per release with build_lya_coverage) or from the tiles of the release (--selection-method tiles, the previous
method and the default for --release Y5, which has no data).
"""
import argparse

from desiutil.log import get_logger
from desisim.survey_release import SurveyRelease, LyaCoverage


def parse(options=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input-master", type=str, required=True,
                        help="Input Master catalog")

    parser.add_argument("--input-data", type=str, required=False,
                        default=None, help="Input observed data catalog: magnitude distribution, redshift distribution of the non-QSO targets "
                                           "and, for --selection-method tiles, counts and exposure times per NPASS class. Not needed for --release Y5")

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path of output catalog")

    parser.add_argument("--seed", type=int, default=None, required=True,
                        help='Mock seed')

    parser.add_argument("--invert", action='store_true', default=False,
                        help='Invert the selection of the mock catalog')

    parser.add_argument("--exptime", type=float, default=None, required=False,
                        help='Exposure time to assign to all targets in the mock catalog')

    parser.add_argument("--zmin", type=float, default=1.7, required=False,
                        help='Minimum redshift of the QSO targets')

    parser.add_argument("--zmax", type=float, default=10.0, required=False,
                        help='Maximum redshift of the QSO targets')

    parser.add_argument("--release", type=str, default='loa', required=False,
                        help='DESI survey release to reproduce (tile file of --selection-method tiles)')

    parser.add_argument("--include-nonqso-targets", action='store_true', default=False,
                        help='Also select non-QSO targets (quasars targeted as other classes, e.g. ELG), with the redshift and magnitude '
                             'distributions of the non-QSO targets of the observed data catalog')

    parser.add_argument("--tiles-file", type=str, default=None, required=False,
                        help='Input tile file to mimic. Overrides the default tile file for the release given by --release')

    parser.add_argument("--selection-method", type=str, default=None, choices=['randoms', 'tiles'], required=False,
                        help="randoms: footprint, downsampling and exposure times from the Lya coverage of the release (--lya-coverage). "
                             "tiles: previous method, NPASS classes from the tiles of the release. Default: tiles for --release Y5, randoms otherwise")

    parser.add_argument("--lya-coverage", type=str, default=None, required=False,
                        help='Lya coverage of the release, lya_<release>_coverage.fits, made with build_lya_coverage (for --selection-method randoms)')

    parser.add_argument("--overwrite", action='store_true', default=False,
                        help='Overwrite output file if it exists')

    args = parser.parse_args(options)
    if args.selection_method is None:
        args.selection_method = 'tiles' if args.release=='Y5' else 'randoms'
    if args.selection_method=='randoms' and args.lya_coverage is None:
        parser.error("--selection-method randoms needs --lya-coverage (build it once per release with build_lya_coverage), "
                     "or use --selection-method tiles for the previous method")
    return args


def main(args=None):
    log = get_logger()
    if not isinstance(args, argparse.Namespace):
        args = parse(args)

    lya_coverage = None
    nonqso_pool_factor = 1.
    if args.selection_method=='randoms':
        lya_coverage = LyaCoverage.read(args.lya_coverage)
        # The coverage keeps only a fraction of the non-QSO candidates, so it needs more of them than the data has per area.
        nonqso_pool_factor = lya_coverage.nonqso_pool_factor()

    survey = SurveyRelease(mastercatalog=args.input_master,seed=args.seed,
                           include_nonqso_targets=args.include_nonqso_targets,
                           data_file=args.input_data,invert=args.invert,nonqso_pool_factor=nonqso_pool_factor)

    # Apply redshift distribution
    # Note: For Y1 and Y3 mocks (and probably Y5 too) the target selection redshift distribution
    # from Chaussidon et al. 2022 works better to match QSO targets Iron catalog.
    # The option distribution='from_data' should be a better option once I finish implementing it.
    survey.apply_redshift_dist(distribution='target_selection', zmin=args.zmin, zmax=args.zmax)

    # Apply footprint and downsampling: Lya coverage of the release, or NPASS geometry either from a release or a custom tiles file.
    if lya_coverage is not None:
        survey.apply_lya_coverage(lya_coverage)
    else:
        survey.apply_data_geometry(release=args.release, tilefile=args.tiles_file)

    # Assign magnitudes
    # Passes from_data = False for Y5 mocks. And from_data=True otherwise.
    survey.assign_rband_magnitude(from_data=(args.release!='Y5'))

    # Assign exposures (from the footprint step above, unless exptime is given)
    if args.release=='Y5' and args.exptime!=4000:
        log.warning('Exptime should be 4000 for Y5 mocks. Assigning exptime=4000')
        args.exptime = 4000

    survey.assign_exposures(exptime=args.exptime)  # Pass exptime = 4000 for Y5 mocks.

    # Write mock catalog, with the selection method and the coverage file in the header
    survey.mockcatalog.meta['SELMETH'] = args.selection_method
    if lya_coverage is not None:
        survey.mockcatalog.meta['LYACOV'] = args.lya_coverage.rsplit('/',1)[-1]
    survey.mockcatalog.write(args.output,overwrite=args.overwrite)
