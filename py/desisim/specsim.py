'''
desisim.specsim
===============

DESI wrapper functions for external specsim classes.
'''

#- Cached simulators, keyed by config string
_simulators = dict()

#- Cached defaults after loading a new simulator, to be used to reset a
#- simulator back to a reference state before returning it as a cached copy
_simdefaults = dict()

import numpy as np

from specsim.config import Configuration
import desiutil.log
log = desiutil.log.get_logger()

def get_simulator(config='desi', num_fibers=1, camera_output=True):
    '''
    returns new or cached specsim.simulator.Simulator object

    Also adds placeholder for BGS fiberloss if that isn't already in the config
    '''
    if isinstance(config, Configuration):
        w = config.wavelength
        wavehash = (np.min(w), np.max(w), len(w))
        key = (config.name, wavehash, num_fibers, camera_output)
    else:
        key = (config, num_fibers, camera_output)

    if key in _simulators:
        log.debug('Returning cached {} Simulator'.format(key))
        qsim = _simulators[key]
        defaults = _simdefaults[key]
        qsim.source.focal_xy = defaults['focal_xy']
        qsim.atmosphere.airmass = defaults['airmass']
        qsim.observation.exposure_time = defaults['exposure_time']
        qsim.atmosphere.moon.moon_phase = defaults['moon_phase']
        qsim.atmosphere.moon.separation_angle = defaults['moon_angle']
        qsim.atmosphere.moon.moon_zenith = defaults['moon_zenith']

    else:
        log.debug('Creating new {} Simulator'.format(key))

        #- New config; create Simulator object
        import specsim.simulator
        qsim = specsim.simulator.Simulator(config, num_fibers,
            camera_output=camera_output)

        #- Cache defaults to reset back to original state later
        defaults = dict()
        defaults['focal_xy'] = qsim.source.focal_xy
        defaults['airmass'] = qsim.atmosphere.airmass
        defaults['exposure_time'] = qsim.observation.exposure_time
        defaults['moon_phase'] = qsim.atmosphere.moon.moon_phase
        defaults['moon_angle'] = qsim.atmosphere.moon.separation_angle
        defaults['moon_zenith'] = qsim.atmosphere.moon.moon_zenith

        _simulators[key] = qsim
        _simdefaults[key] = defaults

    return qsim

