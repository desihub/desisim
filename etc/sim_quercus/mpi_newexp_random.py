
from mpi4py import MPI

import sys
import os
import argparse
import traceback

import numpy as np

from desispec.util import option_list

from desispec.parallel import stdouterr_redirected

from desisim import obs

import desisim.scripts.newexp_random as newexp


flavors = ['arc', 'arc', 'arc', 
           'flat', 'flat', 'flat', 
           'dark', 'gray', 'bright'
           ]

nights = [
    '20191001',
    '20191002',
    '20191003',
    '20191004',
    '20191005'
]

ntask = len(flavors) * len(nights)

comm = MPI.COMM_WORLD
if comm.size < ntask:
    if comm.rank == 0:
        print("Communicator size ({}) too small for {} tasks".format(comm.size, ntask), flush=True)
        comm.Abort()

np.random.seed(123456)
seeds = np.random.randint(2**32, size=ntask)

expids = None
if comm.rank == 0:
    expids = obs.get_next_expid(ntask)
expids = comm.bcast(expids, root=0)

tileids = list()
if comm.rank == 0:
    for nt in nights:
        for fl in flavors:
            flavor = fl.lower()
            t = obs.get_next_tileid(program=flavor)
            tileids.append(t)
            if flavor in ('arc', 'flat'):
                obs.update_obslog(obstype=flavor, program='calib', tileid=t)
            elif flavor in ('bright', 'bgs', 'mws'):
                obs.update_obslog(obstype='science', program='bright', tileid=t)
            elif flavor in ('gray', 'grey'):
                obs.update_obslog(obstype='science', program='gray', tileid=t)
            else:
                obs.update_obslog(obstype='science', program='dark', tileid=t)

tileids = comm.bcast(tileids, root=0)

if comm.rank == 0:
    simdir = os.path.join(os.environ['DESI_SPECTRO_SIM'], 
        os.environ['PIXPROD'])
    etcdir = os.path.join(simdir, 'etc')
    if not os.path.isdir(etcdir):
        os.makedirs(etcdir)
    for nt in nights:
        ntdir = os.path.join(simdir, nt)
        if not os.path.isdir(ntdir):
            os.makedirs(ntdir)

comm.barrier()

taskproc = 1

comm_group = comm
comm_rank = None
group = comm.rank
ngroup = comm.size
group_rank = 0
if comm is not None:
    if taskproc > 1:
        ngroup = int(nproc / taskproc)
        group = int(rank / taskproc)
        group_rank = rank % taskproc
        comm_group = comm.Split(color=group, key=group_rank)
        comm_rank = comm.Split(color=group_rank, key=group)
    else:
        comm_group = MPI.COMM_SELF
        comm_rank = comm

log_root = "newexp_"

task = 0
for nt in nights:
    for fl in flavors:
        tasklog = "{}{}-{:08d}.log".format(log_root, nt, expids[task])
        if task == group:
            with stdouterr_redirected(to=tasklog, comm=comm_group):
                try:
                    options = {}
                    options["program"] = fl
                    options["night"] = nt
                    options["expid"] = expids[task]
                    options["tileid"] = tileids[task]
                    options["seed"] = seeds[task]
                    #options["nproc"] = 1
                    optarray = option_list(options)
                    args = newexp.parse(optarray)
                    newexp.main(args)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    print("".join(lines), flush=True)
        task += 1

