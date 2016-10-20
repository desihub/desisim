"""
Run integration test using quickgen output for full pipeline

python -m desisim.test.integration_test_quickgen
"""
import os
import desisim.io
import desispec.io
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

def sim(night, nspec=25, clobber=False):
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
        simspec = desisim.io.findfile('simspec', night, expid)
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

    flat_expid = 00000000
    expid = 00000002

    # check for required environment variables and simulate inputs
    check_env()
    sim(night, nspec=nspec, clobber=clobber)

    for camera in ['b0', 'r0', 'z0']:

        # find all necessary input and output files
        framefile = desispec.io.findfile('frame', night, expid, camera)
        fiberflatfile = desispec.io.findfile('fiberflat', night, flat_expid, camera)
        skyfile = desispec.io.findfile('sky', night, expid, camera)
        skytestfile = desispec.io.findfile('sky', night, expid, camera) + 'test'
        calibfile = desispec.io.findfile('calib', night, expid, camera)
        calibtestfile = desispec.io.findfile('calib', night, expid, camera) + 'test'
        stdstarsfile = desispec.io.findfile('stdstars', night, expid, camera, spectrograph=0)
        cframetestfile = desispec.io.findfile('cframe', night, expid, camera) + 'test'

        # verify that quickgen output works for full pipeline
        com = "desi_compute_sky --infile {} --fiberflat {} --outfile {}".format(framefile, fiberflatfile, skytestfile)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_compute_sky failed for camera {}'.format(camera))

        com = "desi_fit_stdstars --frames {} --skymodels {} --fiberflats {} --starmodels $DESI_BASIS_TEMPLATES/star_templates_v2.1.fits --outfile {}".format(framefile, skyfile, fiberflatfile, stdstarsfile)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_fit_stdstars failed for camera {}'.format(camera))

        com = "desi_compute_fluxcalibration --infile {} --fiberflat {} --sky {} --models {} --outfile {}".format(framefile, fiberflatfile, skyfile, stdstarsfile, calibtestfile)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_compute_fluxcalibration failed for camera {}'.format(camera))

        com = "desi_process_exposure --infile {} --fiberflat {} --sky {} --calib {} --outfile {}".format(framefile, fiberflatfile, skyfile, calibfile, cframetestfile)
        if pipe.runcmd(com, clobber=clobber) != 0:
            raise RuntimeError('desi_process_exposure failed for camera {}'.format(camera))

    com = "desi_make_bricks --night {}".format(night)
    if pipe.runcmd(com, clobber=clobber) != 0:
        raise RuntimeError('desi_make_bricks failed')

if __name__ == '__main__':
    integration_test()

