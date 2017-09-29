
from mpi4py import MPI

import sys
import os
import argparse

import numpy as np

from desispec.util import option_list

from desispec.parallel import stdouterr_redirected

import desisim.scripts.pixsim as pixsim


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

cams = []
for band in ['b', 'r', 'z']:
    for spec in range(10):
        cams.append('{}{}'.format(band, spec))
cameras = ','.join(cams)

ntask = len(flavors) * len(nights) * 30

comm = MPI.COMM_WORLD
if comm.size < ntask:
    if comm.rank == 0:
        print("Communicator size ({}) too small for {} tasks".format(comm.size, ntask), flush=True)
        comm.Abort()

np.random.seed(123456)
seeds = np.random.randint(2**32, size=ntask)

expids = list(range(len(flavors) * len(nights)))


taskproc = 30

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

log_root = "pixsim_"

task = 0
for nt in nights:
    for fl in flavors:
        tasklog = "{}{}-{:08d}.log".format(log_root, nt, expids[task])
        if task == group:
            with stdouterr_redirected(to=tasklog, comm=comm_group):
                try:
                    options = {}
                    options["night"] = nt
                    options["expid"] = expids[task]
                    options["cosmics"] = True
                    options["seed"] = seeds[task]
                    options["cameras"] = cameras
                    #options["ncpu"] = 1
                    options["verbose"] = True
                    optarray = option_list(options)
                    args = pixsim.parse(optarray)
                    pixsim.main(args, comm_group)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    print("".join(lines), flush=True)
        task += 1


