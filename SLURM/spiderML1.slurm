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
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_2g.20gb

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

# Function to display help message
usage() {
    echo "Usage: $0 -s numSpecies -i numImages -m MODEL"
    exit 1
}

# Parse command-line arguments
while getopts ":s:i:m:" opt; do
  case $opt in
    s) numSpecies="$OPTARG"
    ;;
    i) numImages="$OPTARG"
    ;;
    m) MODEL="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
    ;;
    :) echo "Option -$OPTARG requires an argument." >&2
       usage
    ;;
  esac
done

# Check if the required arguments are provided
if [ -z "$numSpecies" ] || [ -z "$numImages" ] || [ -z "$MODEL" ]; then
    usage
fi

# Construct the dataset string
DATASET="zkdeng/spiderTraining${numSpecies}-${numImages}"

cd /xdisk/nirav/zkdeng/SpiderML/scripts

### Run your work
time python datasetUpload.py --numSpecies "$numSpecies" --numImages "$numImages"
time python huggingFaceTemplate.py --dataset "$DATASET" --model "$MODEL"

mv "${MODEL##*/}-finetuned-${DATASET##*/}" ../models/

sleep 10
