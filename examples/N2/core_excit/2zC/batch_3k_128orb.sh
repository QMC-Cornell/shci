#!/bin/bash
#SBATCH -o out6
#SBATCH -N 1 -c 32 -x node10[01-32],node20[01-20],node32[01-05] -p hi_mem
#SBATCH -t 10-99:00:00

echo 'Running on ' $SLURM_JOB_NUM_NODES ' nodes:' $SLURM_JOB_NODELIST ', with ' $SLURM_CPUS_PER_TASK ' threads per node'
echo ''

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

HOSTFILE=/tmp/hosts.$SLURM_JOB_ID
srun hostname -s > $HOSTFILE

echo 'hostfile' $HOSTFILE
echo 'cpus_per_node(# nodes)' $SLURM_JOB_CPUS_PER_NODE

mpirun_rsh -hostfile $HOSTFILE -np $SLURM_JOB_NUM_NODES MV2_ENABLE_AFFINITY=0 /home/cyrus/shci/shci_orb128 >> out6
cat out6 >> out7
