#!/bin/bash
#COBALT -n 1
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -A datascience
#COBALT --attrs filesystems=home,grand
#COBALT -O jobs/logs/test_HPS_1_gpu

PROJECT=/home/jgouneau/projects/grand/deephyper/frnn
source $PROJECT/repos/scalable-bo/build/activate-dhenv.sh

export RANKS_PER_NODE=1
export COBALT_JOBSIZE=1
export PYTHONPATH=$PROJECT/repos/scalable-bo/build/dhenv/lib/python3.8/site-packages/:$PYTHONPATH

export acq_func="qUCB"
export strategy="qUCB"
export timeout=3500
export random_state=42 
export problem="minimalistic-frnn"
export sync_val=0
export search="DBO"

if [[ "$sync_val" -eq 0 ]];
then
  export sync_str="async"
else
  export sync_str="sync"
fi

export log_dir="results/test_HPS_1_gpu";

echo "mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
--search $search \
--sync $sync_val \
--timeout $timeout \
--acq-func $acq_func \
--strategy $strategy \
--random-state $random_state \
--log-dir $log_dir \
--verbose 1";

mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -np $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) --hostfile $COBALT_NODEFILE python -m scalbo.exp --problem $problem \
    --search $search \
    --sync $sync_val \
    --timeout $timeout \
    --acq-func $acq_func \
    --strategy $strategy \
    --random-state $random_state \
    --log-dir $log_dir \
    --verbose 1 