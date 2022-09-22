#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A datascience
#COBALT --attrs filesystems=home,grand,theta-fs0
#COBALT -O jobs/logs/train-baseline-buffer

PROJECT=/home/jgouneau/projects/grand/deephyper/frnn
source $PROJECT/repos/scalable-bo/build/activate-dhenv.sh

export RANKS_PER_NODE=1
export COBALT_JOBSIZE=1
export PYTHONPATH=$PROJECT/repos/scalable-bo/build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

echo "Running: mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python train_baseline_buffer.py"
gpustat --no-color -i >> "results/gpu_comparison/gpustat_buffer.txt" &
mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python train_baseline_buffer.py