import sys
import argparse

from desisim.survey_release import SurveyRelease


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-master", type=str, required=True,
                        help="Input Master catalog")

    parser.add_argument("--input-data", type=str, required=False,
                        default=None, help="Input observed data catalog")

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path of output catalog")

    parser.add_argument("--seed", type=int, default=None, required=True,
                        help='Mock seed')
    
    parser.add_argument("--invert", action='store_true', default=False,
                        help='Invert the selection of the mock catalog')

    parser.add_argument("--exptime", type=float, default=None, required=False,
                        help='Exposure time to assign to all targets in the mock catalog')
    
    parser.add_argument("--zmin", type=float, default=1.7, required=False,
                        help='Minimum redshift')
    
    parser.add_argument("--zmax", type=float, default=10.0, required=False,
                        help='Maximum redshift')
                    
    parser.add_argument("--release", type=str, default='loa', required=False,
                        help='DESI survey release to reproduce')
    
    parser.add_argument("--include-nonqso-targets", action='store_true', default=False, 
                        help='Include nonqso targets on observed data catalog in the redshift and magnitude distributions.')
    
    parser.add_argument("--tiles-file", type=str, default=None, required=False,
                        help='Input tile file to mimic. Overrides the default tile file for the release given by --release')
    
    parser.add_argument("--overwrite", action='store_true', default=False,
                        help='Overwrite output file if it exists')
    
    args = parser.parse_args()

    survey = SurveyRelease(mastercatalog=args.input_master,seed=args.seed, 
                           include_nonqso_targets=args.include_nonqso_targets, 
                           data_file=args.input_data,invert=args.invert)

    # Apply redshift distribution
    # Note: For Y1 and Y3 mocks (and probably Y5 too) the target selection redshift distribution
    # from Chaussidon et al. 2022 works better to match QSO targets Iron catalog.
    # The option distribution='from_data' should be a better option once I finish implementing it.
    survey.apply_redshift_dist(distribution='target_selection', zmin=args.zmin, zmax=args.zmax)

    # Apply NPASS geometry either from a release or a custom tiles file.
    survey.apply_data_geometry(release=args.release, tilefile=args.tiles_file)

    # Assign magnitudes
    # Passes from_data = False for Y5 mocks. And from_data=True otherwise.
    survey.assign_rband_magnitude(from_data=(args.release!='Y5'))

    # Assign exposures
    if args.release=='Y5' and args.exptime!=4000:
        print('Warning: Exptime should be 4000 for Y5 mocks. Assigning exptime=4000')
        args.exptime = 4000

    survey.assign_exposures(exptime=args.exptime)  # Pass exptime = 4000 for Y5 mocks.

    # Write mock catalog
    survey.mockcatalog.write(args.output,overwrite=args.overwrite)
