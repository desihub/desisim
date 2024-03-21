import sys
import argparse

from desisim.survey_release import SurveyRelease


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-master", type=str, required=True,
                        help="Input Master catalog")

    parser.add_argument("--input-data", type=str, required=False,
                        default=str(
                            '/global/cfs/cdirs/desi/science/lya/y1-kp6/iron-tests/catalogs/'
                            'QSO_cat_iron_main_dark_healpix_zlya-altbal_zwarn_cut_20230918.fits'
                        ),
                        help="Input Iron catalog")

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path of output catalog")

    parser.add_argument("--seed", type=int, default=None, required=True,
                        help='Mock seed')
    
    parser.add_argument("--invert", action='store_true', default=False,
                        help='Invert the selection of the mock catalog')

    parser.add_argument("--exptime", type=float, default=None, required=False,
                        help='Exposure time to assign to all targets in the mock catalog')
                    
    parser.add_argument("--release", type=str, default='iron', choices=['iron','Y5'], required=False,
                        help='DESI survey release to reproduce')
    
    parser.add_argument("--overwrite", action='store_true', default=False,
                        help='Overwrite output file if it exists')
    
    args = parser.parse_args()

    survey = SurveyRelease(mastercatalog=args.input_master,seed=args.seed, qso_only=True, data_file=args.input_data,invert=args.invert)

    # Apply redshift distribution
    # Note: For Y1 mocks (and probably Y5 too) the target selection redshift distribution
    # from Chaussidon et al. 2022 works better to match QSO targets Iron catalog.
    # The option distribution='from_data' should be a better option once I finish implementing it.
    survey.apply_redshift_dist(distribution='target_selection', zmin=1.8)

    # Apply NPASS geometry:
    survey.apply_data_geometry(release=args.release)  # Pass release = None for Y5 mocks.

    # Assign magnitudes
    # Pass from_data = False for Y5 mocks. Unless you want to use the Y1 magnitude distributions.
    survey.assign_rband_magnitude(from_data=True)

    # Assign exposures
    if args.release=='Y5' and args.exptime!=4000:
        print('Warning: Exptime should be 4000 for Y5 mocks. Assigning exptime=4000')
        args.exptime = 4000

    survey.assign_exposures(exptime=args.exptime)  # Pass exptime = 4000 for Y5 mocks.

    # Write mock catalog
    survey.mockcatalog.write(args.output,overwrite=args.overwrite)
