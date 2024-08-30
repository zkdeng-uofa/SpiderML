#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Optional. Set the job name
#SBATCH --job-name=spiderML
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
#SBATCH --mem=16gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:volta:4

# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
module load python/3.9
source ~/.bashrc
conda init bash
conda init
conda activate spiders

# Function to display help message
usage() {
    echo "Usage: $0 -c config.json"
    exit 1
}

# Parse command-line arguments
while getopts ":c:" opt; do
  case $opt in
    c) config_path="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
    ;;
    :) echo "Option -$OPTARG requires an argument." >&2
       usage
    ;;
  esac
done

# Check if the required argument is provided
if [ -z "$config_path" ]; then
    usage
fi

cd /xdisk/nirav/zkdeng/SpiderML/scripts

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export MASTER_ADDR="localhost"
export MASTER_PORT=12345

### Run your work
time torchrun huggingFaceJSON.py --config "$config_path"

# Move the fine-tuned model to the models directory
mv "*-finetuned-*" ../models/

sleep 10
