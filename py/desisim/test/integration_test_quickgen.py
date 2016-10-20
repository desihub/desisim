"""
Run integration test using quickgen output for full pipeline

python -m desisim.test.integration_test_quickgen
"""
import os
import sys
from desisim import io
import desispec.pipeline as pipe
import desispec.log as logging

desi_templates_available = 'DESI_ROOT' in os.environ
desi_root_available = 'DESI_ROOT' in os.environ

def check_env():
    """
    Check required environment variables; raise RuntimeException if missing
    """
    log = logging.get_logger()
    #- template locations
    missing_env = False
    if 'DESI_BASIS_TEMPLATES' not in os.environ:
        log.warning('missing $DESI_BASIS_TEMPLATES needed for simulating spectra'.format(name))
        missing_env = True

    if not os.path.isdir(os.getenv('DESI_BASIS_TEMPLATES')):
        log.warning('missing $DESI_BASIS_TEMPLATES directory')
        log.warning('e.g. see NERSC:/project/projectdirs/desi/spectro/templates/basis_templates/v1.0')
        missing_env = True

    for name in (
        'DESI_SPECTRO_SIM', 'DESI_SPECTRO_REDUX', 'PIXPROD', 'SPECPROD', 'DESIMODEL'):
        if name not in os.environ:
            log.warning("missing ${0}".format(name))
            missing_env = True

    if missing_env:
        log.warning("Why are these needed?")
        log.warning("    Simulations written to $DESI_SPECTRO_SIM/$PIXPROD/")
        log.warning("    Raw data read from $DESI_SPECTRO_DATA/")
        log.warning("    Spectro pipeline output written to $DESI_SPECTRO_REDUX/$SPECPROD/")
        log.warning("    Templates are read from $DESI_BASIS_TEMPLATES")

    #- Wait until end to raise exception so that we report everything that
    #- is missing before actually failing
    if missing_env:
        log.critical("missing env vars; exiting without running pipeline")
        sys.exit(1)

# Simulate raw data

def sim(night, nspec=5, clobber=False):
    """
    Simulate data as part of the integration test.

    Args:
        night (str): YEARMMDD
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist
        
    Raises:
        RuntimeError if any script fails
    """
    log = logging.get_logger()
    output_dir = os.path.join('$DESI_SPECTRO_REDUX','calib2d')

    # Create input fibermaps, spectra, and quickgen data

    for expid, flavor in zip([0,1,2], ['flat', 'arc', 'dark']):

        cmd = "newexp-desi --flavor {flavor} --nspec {nspec} --night {night} --expid {expid}".format(expid=expid, flavor=flavor, nspec=nspec, night=night)
        simspec = io.findfile('simspec', night, expid)
        fibermap = '{}/fibermap-{:08d}.fits'.format(os.path.dirname(simspec),expid)
        if pipe.runcmd(cmd, clobber=clobber) != 0:
            raise RuntimeError('newexp failed for {} exposure {}'.format(flavor, expid))

        cmd = "quickgen --simspec {} --fibermap {}".format(simspec,fibermap)
        if pipe.runcmd(cmd, clobber=clobber) != 0:
            raise RuntimeError('quickgen failed for {} exposure {}'.format(flavor, expid))

    return

def integration_test(night="20160726", nspec=25, clobber=False):
    """Run an integration test from raw data simulations through redshifts
    
    Args:
        night (str, optional): YEARMMDD, defaults to current night
        nspec (int, optional): number of spectra to include
        clobber (bool, optional): rerun steps even if outputs already exist
        
    Raises:
        RuntimeError if any script fails
      
    """
    log = logging.get_logger()
    log.setLevel(logging.DEBUG)

    flat_expid = "00000000"
    expid = "00000002"

    # check for required environment variables
    check_env()

    # simulate inputs
    sim(night, nspec=nspec, clobber=clobber)
    simdir = os.path.join('$DESI_SPECTRO_SIM','exposures','20160726')
    rawdir = os.path.join('$DESI_SPECTRO_REDUX','exposures','20160726')
    flatdir = os.path.join('$DESI_SPECTRO_REDUX','calib2d','20160726')

    # verify that quickgen output works for full pipeline
    for camera in ['b0', 'r0', 'z0']:
        com = "desi_compute_sky --infile {}/{}/frame-{}-{}.fits --fiberflat {}/fiberflat-{}-{}.fits --outfile {}/{}/sky-{}-{}_test.fits".format(rawdir,expid,camera,expid,flatdir,camera,flat_expid,rawdir,expid,camera,expid)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_compute_sky failed for camera {}'.format(camera))

    for camera in ['b0', 'r0', 'z0']:
        com = "desi_fit_stdstars --frames {}/{}/frame-{}-{}.fits --skymodels {}/{}/sky-{}-{}.fits --fiberflats {}/fiberflat-{}-{}.fits --starmodels $DESI_BASIS_TEMPLATES/star_templates_v2.1.fits --outfile {}/{}/stdstars-{}-{}.fits".format(rawdir,expid,camera,expid,rawdir,expid,camera,expid,flatdir,camera,flat_expid,rawdir,expid,camera,expid)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_fit_stdstars failed for camera {}'.format(camera))

    for camera in ['b0', 'r0', 'z0']:
        com = "desi_compute_fluxcalibration --infile {}/{}/frame-{}-{}.fits --fiberflat {}/fiberflat-{}-{}.fits --sky {}/{}/sky-{}-{}.fits --models {}/{}/stdstars-{}-{}.fits --outfile {}/{}/calib-{}-{}_test.fits".format(rawdir,expid,camera,expid,flatdir,camera,flat_expid,rawdir,expid,camera,expid,rawdir,expid,camera,expid,rawdir,expid,camera,expid)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_compute_fluxcalibration failed for camera {}'.format(camera))

    for camera in ['b0', 'r0', 'z0']:
        com = "desi_process_exposure --infile {}/{}/frame-{}-{}.fits --fiberflat {}/fiberflat-{}-{}.fits --sky {}/{}/sky-{}-{}.fits --calib {}/{}/calib-{}-{}.fits --outfile {}/{}/cframe-{}-{}_test.fits".format(rawdir,expid,camera,expid,flatdir,camera,flat_expid,rawdir,expid,camera,expid,rawdir,expid,camera,expid,rawdir,expid,camera,expid)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_process_exposure failed for camera {}'.format(camera))

    com = "desi_make_bricks --night 20160726"
    if pipe.runcmd(com, clobber=clobber) != 0:
        raise RuntimeError('desi_make_bricks failed')

if __name__ == '__main__':
    integration_test()

