#!/usr/bin/env python
"""
vanilla survey strategy
"""

import desisim.quicksurvey as qs
import argparse 

parser = argparse.ArgumentParser(description="Quick and simplified simulation of a DESI survey")

parser.add_argument("--output_dir", "-O", help="Path to write the outputs", type=str, default="./")
parser.add_argument("--targets_dir","-T",help="Path to the targets.fits file", type=str, required=True)
parser.add_argument("--fiberassign_exec", "-f", help="Executable file for fiberassign", type=str, required=True)
parser.add_argument("--epochs_dir","-E",help="Path to the directory with the epochs files ", type=str, required=True)
parser.add_argument("--template_fiberassign","-t",help="File containing template for fiberassign", type=str, required=True)
parser.add_argument("--n_epochs","-N",help="Number of epochs to run", type=int, default=1)
args = parser.parse_args()

setup = qs.SimSetup(output_path = args.output_dir, 
                    targets_path = args.targets_dir, 
                    fiberassign_exec = args.fiberassign_exec, 
                    epochs_path = args.epochs_dir, 
                    template_fiberassign = args.template_fiberassign, 
                    n_epochs = args.n_epochs)

qs.simulate_setup(setup)

