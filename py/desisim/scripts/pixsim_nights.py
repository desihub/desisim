"""
desisim.scripts.pixsim_nights
=============================

This is a module.
"""
from __future__ import absolute_import, division, print_function

import os,sys
import os.path
import shutil

import random
from time import asctime

import numpy as np

import desimodel.io
from desiutil.log import get_logger
import desispec.io
from desispec.parallel import stdouterr_redirected

from . import pixsim

from ..pixsim import simulate_exposure
from ..pixsim import get_nodes_per_exp
from ..pixsim import mpi_count_nodes
from ..pixsim import mpi_split_by_node

from ..io import SimSpec
from .. import obs, io

import argparse

import desispec.io as specio

from .. import io as simio

log = get_logger()


def parse(options=None):
    parser = argparse.ArgumentParser(
        description = 'Generate pixel-level simulated DESI data for one or more nights',
        )

    parser.add_argument("--nights", type=str, default=None, required=False, help="YEARMMDD,YEARMMDD,YEARMMDD")
    parser.add_argument("--verbose", action="store_true", help="Include debug log info")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw and simpix files")
    parser.add_argument("--cosmics", default=None, action="store_true", required=False, help="Add simulated cosmics")
    # parser.add_argument("--seed", type=int, default=123456, required=False, help="random number seed")

    # parser.add_argument("--nspec", type=int, help="Number of spectra to simulate per camera")
    parser.add_argument("--nexp", type=int, help="Number of exposures to process")
    # parser.add_argument("--wavemin", type=float, help="Minimum wavelength to simulate")
    # parser.add_argument("--wavemax", type=float, help="Maximum wavelength to simulate")
    parser.add_argument("--cameras", type=str, default=None, help="cameras, e.g. b0,r5,z9")
    parser.add_argument("--nodes_per_exp", type=int, default=None, help="nodes per exposure")

    args = None
    if options is None:
        args = parser.parse_args()
    else:
        options = [str(x) for x in options]
        args = parser.parse_args(options)

    return args

def main(args, comm=None):
    if args.verbose:
        import logging
        log.setLevel(logging.DEBUG)
    #we do this so we can use operator.itemgetter
    import operator
    #we do this so our print statements can have timestamps
    import time

    rank = 0
    nproc = 1
    if comm is not None:
        import mpi4py
        rank = comm.rank
        nproc = comm.size

    if rank == 0:
        log.info('Starting pixsim at {}'.format(asctime()))

    #no preflight check here, too complicated.   
    #we'll assume the user knows what he or she is doing...

    # Determine which nights we are using
    nights = None
    if args.nights is not None:
        nights = args.nights.split(",")
    else:
        rawdir = os.path.abspath(specio.rawdata_root())
        nights = []
        nightpat = re.compile(r"\d{8}")
        for root, dirs, files in os.walk(rawdir, topdown=True):
            for d in dirs:
                nightmat = nightpat.match(d)
                if nightmat is not None:
                    nights.append(d)

    # Get the list of exposures for each night
    night_expid = {}
    all_expid = []
    exp_to_night = {}
    for nt in nights:
        night_expid[nt] = specio.get_exposures(nt, raw=True)

    #get a list of tuples of (night,expid) that we can evenly divide between communicators
    night_exposure_list=list()
    if comm is None or comm.rank == 0:
        for nt in nights:
            for exp in night_expid[nt]:
                rawfile = desispec.io.findfile('raw', nt, exp)
                if not os.path.exists(rawfile):
                    night_exposure_list.append([nt,exp])
                elif args.overwrite:
                    log.warning('Overwriting pre-existing {}'.format(os.path.basename(rawfile)))
                    os.remove(rawfile)
                    night_exposure_list.append([nt,exp])
                else:
                    log.info('Skipping pre-existing {}'.format(os.path.basename(rawfile)))

        if args.nexp is not None:
            night_exposure_list = night_exposure_list[0:args.nexp]

    if comm is not None:
        night_exposure_list = comm.bcast(night_exposure_list, root=0)

    if len(night_exposure_list) == 0:
        if comm is None or comm.rank == 0:
            log.error('No exposures to process')
        sys.exit(1)

    # Get the list of cameras and make sure it's in the right format
    cams = []
    if args.cameras is not None:
        entry = args.cameras.split(',')
        for i in entry:
            cams.append(i)
    else:
        #do this with band first so we can avoid re-broadcasting cosmics
        for band in ['b', 'r', 'z']:
            for spec in range(10):
                cams.append('{}{}'.format(band, spec)) 

    #are we using cosmics?
    if args.cosmics is not None:
        addcosmics = True
    else:
        addcosmics = False

    ncameras=len(cams)
    nexposures=len(night_exposure_list)
    #call ultity function to figure out how many nodes we have
    nnodes=mpi_count_nodes(comm)

    #call utility functions to divide our workload
    if args.nodes_per_exp is not None:
        user_specified_nodes=args.nodes_per_exp
    else:
        user_specified_nodes=None

    nodes_per_comm_exp=get_nodes_per_exp(nnodes,nexposures,ncameras,user_specified_nodes)    
    #also figure out how many exposure communicators we have
    num_exp_comm = nnodes // nodes_per_comm_exp 
 
    #split the communicator into exposure communicators
    comm_exp, node_index_exp, num_nodes_exp = mpi_split_by_node(comm, nodes_per_comm_exp)
    #further splitting will happen automatically in simulate_exposure

    #based on this, figure out which simspecfiles and rawfiles are assigned to each communicator
    #find all specfiles
    #find all rawfiles
    rawfile_list=[]
    simspecfile_list=[]
    night_list=[]
    expid_list=[]
    for i in range(len(night_exposure_list)):
        night_list.append(night_exposure_list[i][0])
        expid_list.append(night_exposure_list[i][1])
        rawfile_list.append(desispec.io.findfile('raw', night_list[i], expid_list[i]))
        simspecfile_list.append(io.findfile('simspec', night_list[i], expid_list[i]))
 
    #now divy the rawfiles and specfiles between node communicators
    #there is onerawfile and one specfile for each exposure
    rawfile_comm_exp=[]
    simspecfile_comm_exp=[]
    for i in range(num_exp_comm):
        if node_index_exp == i: #assign rawfile, simspec file to one communicator at a time
            rawfile_comm_exp=rawfile_list[i::num_exp_comm]
            simspecfile_comm_exp=simspecfile_list[i::num_exp_comm] 
            night_comm_exp=night_list[i::num_exp_comm]
            expid_comm_exp=expid_list[i::num_exp_comm]
    
    comm.Barrier()

    #now wrap pixsim.simulate_exposure for each exposure (in desisim.pixsim)
    if comm_exp.rank == 0:
        log.info("Starting simulate_exposure for night {} expid {}".format(night_comm_exp, expid_comm_exp))
    for i in range(len(rawfile_comm_exp)):
        simulate_exposure(simspecfile_comm_exp[i], rawfile_comm_exp[i], cameras=cams,
            ccdshape=None, simpixfile=None, addcosmics=addcosmics, comm=comm_exp)


    comm.Barrier()

    if rank == 0:
        log.info('Finished pixsim nights {}'.format(args.nights, asctime()))




