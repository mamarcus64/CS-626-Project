#!/bin/sh
#SBATCH --job-name=brainscore
#SBATCH --output=/scratch1/mjma/slurm_logs/%A/task_%a.out  # Output file for each task
#SBATCH --error=/scratch1/mjma/slurm_logs/%A/task_%a.err    # Error file for each task
#SBATCH --mem=48G                   # Memory per task (adjust as needed)
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=a40:1
#SBATCH --ntasks 1

source ~/.bashrc
cd /scratch1/mjma/CS-626-Project
conda activate brain
python scripts/brainscore_runner.py --model_name $1 --conditions "${@:2}"