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

__version__ = 'unknown'
def version():
    global __version__
    if __version__ != 'unknown':
        return __version__
    else:
        from subprocess import Popen, PIPE
        try:
            p = Popen(['git', "describe", "--tags", "--dirty", "--always"], stdout=PIPE)
        except EnvironmentError:
            return __version__
        out = p.communicate()[0]
        if p.returncode != 0:
            return __version__
            
        __version__ = out.rstrip()
        return __version__

#- This requires making a system call every time desisim is loaded.  Yuck.
# __version__ = version()
        
