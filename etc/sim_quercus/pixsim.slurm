#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=150
#SBATCH --time=00:30:00
#SBATCH --job-name=pixsim
#SBATCH --output=pixsim_%j.log

echo Starting slurm script at `date`

nodes=150
node_proc=6

# Set TMPDIR to be on the ramdisk
export TMPDIR=/dev/shm

cpu_per_core=2
node_cores=24
node_thread=$(( node_cores / node_proc ))
node_depth=$(( cpu_per_core * node_thread ))
procs=$(( nodes * node_proc ))

export OMP_NUM_THREADS=${node_thread}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

simdir="${SCRATCH}/desi/sim"
prod="quercus"

export DESI_SPECTRO_SIM="${simdir}"
export PIXPROD="${prod}"
export DESI_SPECTRO_DATA="${simdir}/${prod}"

run="srun --cpu_bind=cores -n ${procs} -N ${nodes} -c ${node_depth}"

com="${run} pixsim_nights --verbose --cosmics --preproc --camera_procs ${node_proc}"

echo ${com}
time ${com} 2>&1

