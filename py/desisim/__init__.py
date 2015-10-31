"""
desisim
=======

Tools for DESI instrument simulations, including input templates.
It does not cover cosmology simulations.

Also see desimodel, which contains quicksim.
"""
from . import pixsim
from . import obs
from . import io
from . import targets

__version__ = '0.6.dev0'
def gitversion():
    from subprocess import Popen, PIPE
    try:
        p = Popen(['git', "describe", "--tags", "--dirty", "--always"], stdout=PIPE)
    except EnvironmentError:
        return __version__
    out = p.communicate()[0]
    if p.returncode != 0:
        return __version__
        
    return out.rstrip()

        
