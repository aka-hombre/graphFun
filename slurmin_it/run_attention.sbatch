#!/bin/bash
#SBATCH -N 1           # number of nodes
#SBATCH -c 8           # number of cores
#SBATCH --mem=384G     # 384BG of memory (still enough for public)
#SBATCH -t 0-05:30:00  # time in d-hh:mm:ss
#SBATCH -p public      # partition
#SBATCH -q public      # QOS
#SBATCH -G 1           # request 1 gpu    
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user="\%u@asu.edu"
#SBATCH --export=NONE

export PYTHONPATH=$PWD:$PYTHONPATH

apptainer exec --nv /home/iarodr13/projects/graphFun/mySif.sif python3 graphfun/scripts/train_attention.py