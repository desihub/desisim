"""
Session-wide pytest configuration for desisim tests.
"""
import os
from importlib import resources

import pytest


def _surveyops_tiles_available(surveyops):
    """Does `surveyops` (a $DESI_SURVEYOPS-style path) contain tiles-main.ecsv?"""
    return (os.path.exists(os.path.join(surveyops, 'ops', 'tiles-main.ecsv')) or
            os.path.exists(os.path.join(surveyops, 'trunk', 'ops', 'tiles-main.ecsv')))


@pytest.fixture(scope='session', autouse=True)
def _ensure_desi_surveyops():
    """Point $DESI_SURVEYOPS at a bundled minimal tiles-main.ecsv fixture
    if a real surveyops snapshot isn't already available, so that tests
    exercising desimodel.io.load_tiles() (e.g. via desisim.obs.new_exposure)
    don't require downloading the real DESI_SURVEYOPS snapshot.

    A real $DESI_SURVEYOPS snapshot, if present, always takes precedence.
    """
    orig = os.environ.get('DESI_SURVEYOPS')

    have_real_snapshot = (orig is not None) and _surveyops_tiles_available(orig)
    if not have_real_snapshot:
        fixture_dir = str(resources.files('desisim').joinpath('test', 'data', 'surveyops'))
        os.environ['DESI_SURVEYOPS'] = fixture_dir

    yield

    if orig is None:
        os.environ.pop('DESI_SURVEYOPS', None)
    else:
        os.environ['DESI_SURVEYOPS'] = orig
