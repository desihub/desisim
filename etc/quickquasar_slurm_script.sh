#!/bin/bash -l

#SBATCH --partition=debug
#SBATCH --account=desi
#SBATCH --nodes=20
#SBATCH --time=00:30:00
#SBATCH --job-name=lyasim
#SBATCH --output=lyasim.log

# HOWTO:
# 1) copy this script  
# 2) choose your options to quickquasars (z range, downsampling, target selection, DESI footprint) in the command below
# 3) change idir outdir downsampling output nodes nthreads time  
# 4) verify nodes=XX below is same as --nodes=XX above
# 5) sbatch yourjob.sh
#
# example: v1.1.0
# the QSO density (z>2.1) = 52.4 /deg2 , so I set downsampling=1, 
# it's only one chunk , almost fully contained in DESI footprint
downsampling=1.
idir=/project/projectdirs/desi/mocks/lya_forest/v1.1/v1.1.0
outdir=/project/projectdirs/desi/users/jguy/qso-v1.1.0-3

nodes=20 # CHECK MATCHING #SBATCH --nodes ABOVE !!!!
nthreads=4 # TO BE TUNED ; CAN HIT NODE MEMORY LIMIT ; 4 is max on edison for nside=16 and ~50 QSOs/deg2

    
if [ ! -d $outdir/jobs ] ; then
    mkdir -p $outdir/jobs
fi
if [ ! -d $outdir/logs ] ; then
    mkdir -p $outdir/logs
fi
if [ ! -d $outdir/spectra-16 ] ; then
    mkdir -p $outdir/spectra-16
fi

echo "get list of skewers to run ..."

files=`\ls -1 $idir/*/*/transmission*.fits`
nfiles=`echo $files | wc -w`
nfilespernode=$((nfiles/nodes+1))

echo "n files =" $nfiles
echo "n files per node =" $nfilespernode

first=1
last=$nfilespernode
for node in `seq $nodes` ; do
    echo "starting node $node"

    # list of files to run
    if (( $node == $nodes )) ; then
	last=""
    fi
    echo ${first}-${last}
    tfiles=`echo $files | cut -d " " -f ${first}-${last}`
    first=$(( first + nfilespernode ))
    last=$(( last + nfilespernode ))
    command="srun -N 1 -n 1 -c $nthreads  quickquasars --exptime 4000. -i $tfiles --zmin 1.6 --nproc $nthreads --outdir $outdir/spectra-16 --downsampling $downsampling --zbest"
    echo $command
    echo "log in $outdir/logs/node-$node.log"
    
    $command >& $outdir/logs/node-$node.log &
    
done

wait
echo "END"



