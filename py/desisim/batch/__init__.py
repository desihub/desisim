"""
desisim.batch
=============

Batch scripts.  Why exactly is this sub-package different from
:mod:`desisim.scripts`?
"""
from __future__ import absolute_import, division, print_function
import math

from desispec.log import get_logger
_log = get_logger()

def calc_nodes(ntasks, tasktime, maxtime):
    '''
    Return a recommended number of nodes to use to process `ntasks` within
    `maxtime` if each one takes `tasktime` minutes.
    '''
    #- number of tasks that can be serially run within maxtime
    n = int(maxtime / tasktime)

    nodes = max(4, int(math.ceil(ntasks / n)))
    runtime = math.ceil(ntasks / nodes) * tasktime
    _log.debug('Requesting {} nodes for {} tasks'.format(nodes, ntasks, runtime))
    _log.debug('Expected runtime {} minutes'.format(runtime))
    return nodes
