"""
desisim.scripts.pixsim_nights
======================

Entry point for simulating multiple nights.
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import re
import argparse
import traceback

import numpy as np

from desispec.util import option_list

from desispec.parallel import stdouterr_redirected

import desispec.io as specio

from .. import io as simio

from ..io import SimSpec

from . import pixsim


def parse(options=None):
    parser = argparse.ArgumentParser(
        description = 'Generate pixel-level simulated DESI data for one or more nights',
        )

    parser.add_argument("--nights", type=str, default=None, required=False, help="YEARMMDD,YEARMMDD,YEARMMDD")
    parser.add_argument("--verbose", action="store_true", help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw and simpix files")
    parser.add_argument("--cosmics", action="store_true", help="Add simulated cosmics")
    parser.add_argument("--preproc", action="store_true", help="Run the preprocessing")
    parser.add_argument("--seed", type=int, default=123456, required=False, help="random number seed")

    parser.add_argument("--cameras", type=str, default=None, help="cameras, e.g. b0,r5,z9")
    parser.add_argument("--expids", type=str, default=None, help="expids, e.g. 0,12,14")
    parser.add_argument("--camera_procs", type=int, default=1, help="Number "
        "of MPI processes to use per camera")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        options = [str(x) for x in options]
        args = parser.parse_args(options)

    return args


def main(args, comm=None):
    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    # Determine which nights we are using
    nights = None
    if args.nights is not None:
        nights = args.nights.split(",")
    else:
        if rank == 0:
            rawdir = os.path.abspath(specio.rawdata_root())
            nights = []
            nightpat = re.compile(r"\d{8}")
            for root, dirs, files in os.walk(rawdir, topdown=True):
                for d in dirs:
                    nightmat = nightpat.match(d)
                    if nightmat is not None:
                        nights.append(d)
                break
        if comm is not None:
            nights = comm.bcast(nights, root=0)

    # Get the list of exposures for each night
    requested_expids=None
    if args.expids is not None :
        requested_expids=list()
        vals = args.expids.split(",")
        for v in vals :
            requested_expids.append(int(v))
    
    night_expid = {}
    all_expid = []
    exp_to_night = {}
    if rank == 0:
        for nt in nights:
            if requested_expids is not None :
                night_expid[nt] = list()
                for expid in specio.get_exposures(nt, raw=True) :
                    if expid in requested_expids :
                        night_expid[nt].append(expid)
            else :    
                night_expid[nt] = specio.get_exposures(nt, raw=True) 
            
            all_expid.extend(night_expid[nt])
            for ex in night_expid[nt]:
                exp_to_night[ex] = nt
            
    
    if comm is not None:
        night_expid = comm.bcast(night_expid, root=0)
        all_expid = comm.bcast(all_expid, root=0)
        exp_to_night = comm.bcast(exp_to_night, root=0)

    expids = np.array(all_expid, dtype=np.int32)
    nexp = len(expids)

    # Get the list of cameras
    cams = None
    if args.cameras is not None:
        cams = args.cameras.split(",")
    else:
        cams = []
        for band in ['b', 'r', 'z']:
            for spec in range(10):
                cams.append('{}{}'.format(band, spec))

    # number of cameras
    ncamera = len(cams)

    # check that our communicator is an appropriate size

    if comm is not None:
        if ncamera * args.camera_procs > comm.size:
            if comm.rank == 0:
                print("Communicator size ({}) too small for {} cameras each with {} procs".format(comm.size, ncamera, args.camera_procs), flush=True)
                comm.Abort()

    # create a set of reproducible seeds for each exposure
    np.random.seed(args.seed)
    maxexp = np.max(expids)
    allseeds = np.random.randint(2**32, size=(maxexp+1))
    seeds = allseeds[-nexp:]

    taskproc = ncamera * args.camera_procs

    comm_group = comm
    comm_rank = None
    group = 0
    ngroup = 1
    if comm is not None:
        group = comm.rank
        ngroup = comm.size

    group_rank = 0
    if comm is not None:
        from mpi4py import MPI
        if taskproc > 1:
            ngroup     = int(np.ceil(float(comm.size) / taskproc))
            group = int(comm.rank / taskproc)
            group_rank = comm.rank % taskproc
            comm_group = comm.Split(color=group, key=group_rank)
            comm_rank = comm.Split(color=group_rank, key=group)
        else:
            comm_group = MPI.COMM_SELF
            comm_rank = comm

    myexpids = np.array_split(expids, ngroup)[group]

    for ex_counter,ex in enumerate(myexpids):
        nt = exp_to_night[ex]
        
        # path to raw file
        simspecfile = simio.findfile('simspec', nt, ex)
        rawfile = specio.findfile('raw', nt, ex)
        rawfile = os.path.join(os.path.dirname(simspecfile), rawfile)
        
        # Is this exposure already finished?
        done = True
        if group_rank == 0:
            if not os.path.isfile(rawfile):
                done = False
            if args.preproc:
                for c in cams:
                    pixfile = specio.findfile('pix', night=nt,
                        expid=ex, camera=c)
                    if not os.path.isfile(pixfile):
                        done = False
        if comm_group is not None:
            done = comm_group.bcast(done, root=0)
        if done and not args.overwrite:
            if group_rank == 0:
                print("Skipping completed exposure {:08d} on night {}".format(ex, nt))
            continue

        # Write per-process logs to a separate directory,
        # since there are so many of them.
        logdir = "{}_logs".format(rawfile)
        if group_rank == 0:
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
        if comm_group is not None:
            comm_group.barrier()
        tasklog = os.path.join(logdir, "pixsim")

        with stdouterr_redirected(to=tasklog, comm=comm_group):
            try:
                options = {}
                options["night"] = nt
                options["expid"] = int(ex)
                options["cosmics"] = args.cosmics
                options["seed"] = seeds[ex_counter]
                options["cameras"] = ",".join(cams)
                options["mpi_camera"] = args.camera_procs
                options["verbose"] = args.verbose
                options["preproc"] = args.preproc
                optarray = option_list(options)
                pixargs = pixsim.parse(optarray)
                pixsim.main(pixargs, comm_group)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                print("".join(lines), flush=True)


