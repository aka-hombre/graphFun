#!/bin/bash

#SBATCH -N 1           # number of nodes
#SBATCH -c 8           # number of cores
#SBATCH --mem=32G      # 32GB of memory
#SBATCH -t 0-03:30:00  # time in d-hh:mm:ss
#SBATCH -p public      # partition
#SBATCH -q public      # QOS
#SBATCH -G 1           # request 1 gpu    
#SBATCH -o slurm.\%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.\%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="\%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load Mamba
module load mamba/latest
module load cuda

#Activate environment
source activate venv

#Run the python script
python3 graphfun/scripts/train_linear.py
