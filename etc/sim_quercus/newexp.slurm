#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=45
#SBATCH --time=00:30:00
#SBATCH --job-name=newexp
#SBATCH --output=newexp_%j.log

echo Starting slurm script at `date`

nodes=45
node_proc=1

# Set TMPDIR to be on the ramdisk
export TMPDIR=/dev/shm

cpu_per_core=2
node_cores=24
node_thread=$(( node_cores / node_proc ))
node_depth=$(( cpu_per_core * node_thread ))
procs=$(( nodes * node_proc ))

export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread

simdir="${SCRATCH}/desi/sim"
prod="quercus"

export DESI_SPECTRO_SIM="${simdir}"
export PIXPROD="${prod}"
#export DESI_SPECTRO_DATA="${simdir}/${prod}"

#run="srun --cpu_bind=no -n ${procs} -N ${nodes} -c ${node_depth}"
run="srun --cpu_bind=no -n ${procs} -N ${nodes}"

com="${run} python mpi_newexp_random.py"

echo ${com}
time ${com} 2>&1

