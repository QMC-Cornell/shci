#!/bin/bash
#SBATCH -o out6
#SBATCH -N 1 -c 12 -x node10[01-32],node30[01-03],node31[01-03] -t 10-99:00:00

echo 'Running on ' $SLURM_JOB_NUM_NODES ' nodes:' $SLURM_JOB_NODELIST ', with ' $SLURM_CPUS_PER_TASK ' threads per node'
echo ''

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo 'cpus_per_node(# nodes)' $SLURM_JOB_CPUS_PER_NODE

mpirun -np $SLURM_JOB_NUM_NODES /home/cyrus/shci/shci >> out6
cat out6 >> out7
