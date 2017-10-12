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
from desiutil.log import get_logger
from desispec.parallel import stdouterr_redirected

import desispec.io as specio

from .. import io as simio

from ..io import SimSpec

from . import pixsim


def parse(options=None):
    parser = argparse.ArgumentParser(
        description = "Generate pixel-level simulated DESI data for one or more nights",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Example run with mpi
        
        export DESI_SPECTRO_SIM=...
        export PIXPROD=...
        export DESI_SPECTRO_DATA=${DESI_SPECTRO_SIM}/$PIXPROD

        # NNODE : number of nodes to be tuned depending on queue and things to do
        # For NEXP exposures to process, the number of tasks per node is NEXP*30/NNODE
        # and it takes about 10min per task.
        NNODE=50 # to be tuned.
        NCORE=32 # number of cores per node; 32 for cori haswell
        NPROC=$(( NNODE * NCORE )) # number of MPI processes
        
        srun --cpu_bind=cores -n $NPROC -N $NNODE -c 2 pixsim_nights -verbose --cosmics --preproc --camera_procs $NCORE  
        """
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
        "of MPI processes to use per camera and node")

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

    log = get_logger()

    if(args.verbose) :
        import logging
        log.setLevel(logging.DEBUG)
    
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
        
        log.info("Will simulate:")
        for nt in nights: 
            log.info("night {} expids {}".format(nt,night_expid[nt]))
    
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

    # this is the total number of tasks to do 
    # number of cameras x number of exposures
    # by default it is going to be 3*10*nexps
    ntask = ncamera*nexp
    
    # create a set of reproducible seeds for each exposure and camera
    np.random.seed(args.seed)
    seeds = np.random.randint(2**32, size=ntask)
    
    nproc = 1 
    nnode = 1
    
    if comm is not None:
        nproc = comm.size
        nnode = nproc//args.camera_procs
    
    if rank==0 :
        log.debug("number of cameras = {}".format(ncamera))
        log.debug("number of expids  = {}".format(nexp))
        log.debug("number of tasks   = {}".format(ntask))
        log.debug("number of procs   = {}".format(nproc))
        log.debug("number of nodes   = {}".format(nnode))
    
    comm_group = comm
    group = 0 # we want to have a group = a node
    group_rank = rank
    if comm is not None and nnode>1 :
        from mpi4py import MPI
        group      = comm.rank//args.camera_procs
        group_rank = comm.rank % args.camera_procs
        comm_group = comm.Split(color=group, key=group_rank)
    
    # splitting tasks on procs according to the node they belong to
    mytasks  = np.array_split(np.arange(ntask,dtype=np.int32),nnode)[group]
     
    if comm is not None and group_rank == 0 :
            log.debug("group {} size {} tasks {}".format(group,comm_group.size,mytasks))
    
    
    # now we loop on the exposures and cameras that each group has to do
    for task in mytasks :

        # from the task index, find which exposure and camera it corresponds to
        # we do first each camera of exposure before going to the next
        camera_index   = task%ncamera
        exposure_index = task//ncamera
        seed           = seeds[task]
        
        #if comm is not None:
        #    log.debug("rank {} task {} exposure_index {} camera_index {}".format(rank,task,exposure_index,camera_index))
        
        expid  = expids[exposure_index]
        camera = cams[camera_index]
        night = exp_to_night[expid]
        
        
        
        # Is this task already finished?
        # checking if already done
        
        # path to raw file
        simspecfile = simio.findfile('simspec', night, expid)
        rawfile = specio.findfile('raw', night, expid)
        rawfile = os.path.join(os.path.dirname(simspecfile), rawfile)
        pixfile = specio.findfile('pix', night=night, expid=expid, camera=camera)

        if comm_group is not None : comm_group.barrier()
        
        done = True        
        if group_rank == 0:
            done &= os.path.isfile(rawfile)
            if args.preproc: 
                done &= os.path.isfile(pixfile)
        if comm_group is not None:
            done = comm_group.bcast(done, root=0)
        
        if done and not args.overwrite :
            if group_rank == 0:
                log.info("group {} skipping completed night {} expid {} camera {}".format(group,night,expid,camera))
                sys.stdout.flush()
            continue
        
        if comm is not None and group_rank == 0 :
            log.debug("group {} doing night {} expid {} camera {}".format(group,night,expid,camera))
                
        # now we run this
        
        # Write per-process logs to a separate directory,
        # since there are so many of them.
        tasklog = pixfile.replace(".fits",".log")
        with stdouterr_redirected(to=tasklog, comm=comm_group):
            
            try:
                options = {}
                options["night"] = night
                options["expid"] = int(expid)
                options["cameras"] = camera
                options["cosmics"] = args.cosmics
                options["seed"] = seed                
                options["mpi_camera"] = args.camera_procs
                options["verbose"] = args.verbose
                options["preproc"] = args.preproc
                
                #log.debug("group {} running pixsim {}".format(group,options))
                #sys.stdout.flush()
                
                optarray = option_list(options)
                pixargs = pixsim.parse(optarray)
                pixsim.main(pixargs, comm_group)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                print("".join(lines), flush=True)


