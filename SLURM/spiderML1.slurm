#!/bin/bash
 
# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=spiderML1
### Optional. Set the output filename.
### SLURM reads %x as the job name and %j as the job ID
#SBATCH --output=%x-%j.out
### REQUIRED. Specify the PI group for this job. Replace <PI GROUP> with your own group.
#SBATCH --account=nirav
### REQUIRED. Set the partition for your job. This is a job queue
#SBATCH --partition=gpu_standard
### REQUIRED. Set the number of nodes
#SBATCH --nodes=1
### REQUIRED. Set the number of CPUs that will be used for this job. 
#SBATCH --ntasks=2
### REQUIRED. Set the memory required for this job.
#SBATCH --mem-per-cpu=8gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
module load python/3.9
#module load aws
#module load nextflow
source ~/.bashrc
### change to your scripts directory
conda init bash
conda init
conda activate spiders

cd /xdisk/nirav/zkdeng/SpiderML
 
### Run your work
time python scripts/huggingFaceTemplate.py --dataset zkdeng/t5spiders --model facebook/convnextv2-tiny-22k-384
sleep 10

